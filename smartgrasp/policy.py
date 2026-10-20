"""SmartGrasp as a policy for the organisers' remote cell, through their client package.

The client package (https://github.com/SIDCLab/PhysicalAI-Hackathon-2026) calls act(obs) with
the cell's latest observation and plays the targets it returns. In that folder, policy.py holds
one line, `from smartgrasp.policy import Policy`, and run.py runs on this repo's environment:

    uv run --project ../<this repo> run.py --mock --checkpoint hsv

Each rollout, the policy warps the table image to the BEV, detects the cubes and plans the tower
(deploy.plan_frame), then streams deploy.cell_rows as UF0/UTOOL1 "tcp" targets, cut into straight
steps of SPEED_MM_S * DT. It plans once per rollout and holds the arm still before and after.
Each plan's images and files go to runs/cell/<trial id>/ (see docs/CELL_TEST.md).

This is the only code here that moves the real arm. The cell checks every target against its
limits, and an operator supervises every rollout (AGENTS.md).
"""

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .deploy import CellRow, plan_frame
from .detectors import Detector, load_detector
from .geometry import Bev, Homography, Image, cell_homography

ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR: str = f"yolo:{ROOT / 'artifacts/models/sim_v1/weights/best.pt'}"
MAX_CHUNK: int = 100  # the cell takes at most 100 targets per act() call
HOME_TCP: list[float] = [300.0, 0.0, -20.0, -180.0, 0.0, 0.0]  # fingertip at the cell's home joints
GRIPPER: dict[str, float] = {"open": 0.0, "close": 1.0}
log: logging.Logger = logging.getLogger(__name__)

type Target = list[float]  # [x, y, z mm, w, p, r deg, gripper 0 open / 1 closed]: UTOOL1 in UF0


class Observation(TypedDict):
    """The part of the cell's observation (cell_client.CellClient.observe) this policy reads.

    Attributes:
        images: RGB image per requested camera, shape (height, width, 3).
        joints: J1-J6 in degrees, then the gripper (0 open, 1 closed).
        tcp_pose: Fingertip [x, y, z mm, w, p, r deg] in UF0; None on the mock until a tcp target.
        trial: The current rollout, with its "trial_id".
    """

    images: dict[str, NDArray[np.uint8]]
    joints: NDArray[np.float32]
    tcp_pose: list[float] | None
    trial: dict[str, Any]


def _wrap(degrees: float, period: float) -> float:
    """Wrap an angle difference to the shortest turn.

    Args:
        degrees: Angle difference in degrees.
        period: Symmetry of the angle: 360 for a full turn, 90 for a square cube.

    Returns:
        The equivalent difference in [-period / 2, period / 2].
    """
    return degrees - period * round(degrees / period)


def trajectory(start: Target, rows: Sequence[CellRow], step_mm: float, dwell: int) -> list[Target]:
    """Turn waypoints into straight moves of targets at most step_mm apart.

    The wrist turns the short way: W modulo a full turn and R modulo a quarter turn, since a
    cube looks the same every 90 degrees. The gripper switches on a waypoint's last target,
    which is then held dwell more times so the jaws finish before the arm moves on.

    Args:
        start: The fingertip's current target, where the first move begins.
        rows: Waypoints from deploy.cell_rows.
        step_mm: Longest step between consecutive targets, in mm.
        dwell: Extra copies of each target where the gripper switches.

    Returns:
        The targets in playing order; empty if rows is.
    """
    targets: list[Target] = []
    here = start
    for row in rows:
        w = here[3] + _wrap(row["W_deg"] - here[3], 360)
        r = here[5] + _wrap(row["R_deg"] - here[5], 90)
        gripper = GRIPPER.get(row["gripper"], here[6])
        goal = [row["X_mm"], row["Y_mm"], row["Z_mm"], w, row["P_deg"], r, gripper]
        steps = max(1, math.ceil(math.dist(here[:3], goal[:3]) / step_mm))
        for k in range(1, steps):
            targets.append([a + (b - a) * k / steps for a, b in zip(here[:6], goal[:6], strict=True)] + [here[6]])
        targets += [goal] * (1 + (dwell if gripper != here[6] else 0))
        here = goal
    return targets


class Policy:
    """The SmartGrasp tower builder, in the shape the client package's run.py expects.

    Attributes:
        CAMERAS: Cameras requested from the cell: only the table camera.
        IMAGE_SIZE: Requested (width, height): the table camera's native size, as calibrated.
        IMAGE_FORMAT: JPEG, about 7x smaller than PNG to send.
        ACTION: act() returns fingertip targets, not joint angles.
        DT: Seconds each target plays (the cell accepts 0.1 to 1.0).
        PREFETCH: Off: the plan is fixed, so each chunk plays to its end before the next is sent.
        SPEED_MM_S: Fingertip speed along each straight move.
        DWELL_S: How long the arm holds still after each gripper switch, and between plan attempts.
        CALIBRATION: The organisers' calibration, relative to the client folder run.py runs in.
        RUNS: Folder for each rollout's BEV image, overlay and plan files.
        detector_name: Detector spec in use.
        detector: The detector it names.
        bev: View the detector works in.
        H: Homography from table-image pixels to robot-base XY on the cube-top plane.
        queue: Targets of the current plan not yet sent.
        planned: Whether this rollout already has a plan.
        hold: Where the arm holds still once the plan has been sent; None until then.
    """

    CAMERAS: list[str] = ["table"]
    IMAGE_SIZE: tuple[int, int] = (1920, 1080)
    IMAGE_FORMAT: str = "jpeg"
    ACTION: str = "tcp"
    DT: float = 0.1
    PREFETCH: bool = False
    SPEED_MM_S: float = 80.0
    DWELL_S: float = 1.0
    CALIBRATION: Path = Path("calibration/cameras.json")
    RUNS: Path = ROOT / "runs" / "cell"

    detector_name: str
    detector: Detector
    bev: Bev
    H: Homography
    queue: list[Target]
    planned: bool
    hold: Target | None

    def __init__(self, checkpoint: str | None = None) -> None:
        """Load the detector and the table camera's homography.

        Args:
            checkpoint: Detector spec (run.py --checkpoint): yolo:<weights>, hsv or vlm[:<dataset>].
                Default DEFAULT_DETECTOR.

        Raises:
            ValueError: If the calibration is for another image size than IMAGE_SIZE.
        """
        table = json.loads(self.CALIBRATION.read_text())["table"]
        if tuple(table["size"]) != self.IMAGE_SIZE:
            raise ValueError(f"{self.CALIBRATION} is for {table['size']} images, not IMAGE_SIZE {self.IMAGE_SIZE}")
        self.detector_name, self.bev = checkpoint or DEFAULT_DETECTOR, Bev()
        self.detector = load_detector(self.detector_name, self.bev)
        self.H = cell_homography(table, self.bev)
        self.reset()

    def reset(self) -> None:
        """Forget the last rollout's plan; run.py calls this at the start of every rollout."""
        self.queue, self.planned, self.hold = [], False, None

    def act(self, obs: Observation) -> list[Target]:
        """Return the next targets: the plan's next chunk, or the arm held still for DWELL_S.

        Args:
            obs: The cell's latest observation.

        Returns:
            Up to MAX_CHUNK targets, played DT seconds apart.
        """
        here = [*(obs["tcp_pose"] or HOME_TCP), float(obs["joints"][6] >= 0.5)]
        if not self.planned:
            self.queue = self._plan(obs, here)
            self.planned = bool(self.queue)
        if self.queue:
            chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
            self.hold = chunk[-1]
            return chunk
        return [self.hold or here] * round(self.DWELL_S / self.DT)

    def _plan(self, obs: Observation, here: Target) -> list[Target]:
        """Detect the cubes, plan the tower and turn it into targets that start where the arm is.

        Args:
            obs: The observation to plan from.
            here: The fingertip's current pose and gripper state.

        Returns:
            Every target of the plan; none if nothing can be planned.

        Raises:
            ValueError: If the table image is not IMAGE_SIZE.
        """
        rgb = obs["images"]["table"]
        if rgb.shape[1::-1] != self.IMAGE_SIZE:
            raise ValueError(f"table image is {rgb.shape[1::-1]}, but the calibration is for {self.IMAGE_SIZE}")
        out = self.RUNS / str(obs["trial"]["trial_id"])
        bgr = cast(Image, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        detections, picks, rows = plan_frame(bgr, self.H, self.detector, self.detector_name, out, self.bev)
        cv2.imwrite(str(out / "table.jpg"), bgr)  # a real frame for the held-out evaluation set
        log.info("%d detections, %d picks: %s", len(detections), len(picks), out)
        if not all(row["within_limits"] for row in rows):
            log.warning("not sending a plan with targets outside the limits: see %s", out / "cell_plan.csv")
            return []
        return trajectory(here, rows, self.SPEED_MM_S * self.DT, round(self.DWELL_S / self.DT))

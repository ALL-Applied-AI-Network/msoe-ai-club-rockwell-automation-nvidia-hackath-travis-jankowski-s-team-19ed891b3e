r"""Camera image -> cube detections -> pick-and-place plan, for the twin and the real cell.

    python -m smartgrasp.deploy --image runs/ep1/world_camera.png --calib runs/ep1/camera.json \
        --detector yolo:artifacts/models/sim_v1/weights/best.pt --out runs/ep1

Writes into --out:
    bev.png, overlay.png    what the detector saw and what it found
    detections.json         every detection in robot-base metres
    sort_plan.json          the plan in the twin's format: fanuc_er4ia_sim/sort_scene.py executes it
    cell_plan.csv/.json     the same plan as FANUC UF0/UTOOL1 XYZWPR targets for operator review

Nothing here talks to a robot controller. The real-cell plan is a dry run for a
human to check; the cell's own safety systems stay in charge.
"""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from .detectors import Detection, load_detector
from .geometry import Bev, BgrColour, Category, Image, load_homography, pose, read_image, twin

MIN_CONF: float = 0.8  # the twin's executor refuses grasps below this pose confidence
REACH: tuple[tuple[float, float], tuple[float, float]] = ((0.18, 0.52), (-0.28, 0.30))  # x, y box, robot-base metres
REACH_RADIUS_M: float = 0.54  # the twin's IK fails top-down hovers from 0.545 m out
SITE_CLEAR_M: float = 0.045  # a 6-cube tower's top shows up to 35 mm off-site in the BEV (parallax)
APPROACH_M: float = 0.13  # hover height above pick and place, as in sort_scene.py
UF0_Z_M: float = 0.330  # controller UF0 is 330 mm above the robot base (fanuc_er4ia_sim/docs/CALIBRATION_V05.md)
SAFE_Z: tuple[float, float] = (twin.SURFACE_DZ + 0.002, 0.25)  # robot-base z limits for every target, metres
BGR: dict[str, BgrColour] = {  # overlay outline colour for each detection class
    "cube_yellow": (0, 220, 255),
    "cube_blue": (255, 120, 0),
    "cube_red": (0, 0, 255),
    "cap": (200, 200, 200),
    "tube": (160, 160, 160),
}


class Pick(TypedDict):
    """One cube move in the twin's sort_plan.json format, which fanuc_er4ia_sim/sort_scene.py runs.

    Attributes:
        detection_id: Index of the cube in the deploy run's detections (detections.json).
        category: Always "cube" for the stacking task.
        label: Readable name such as "blue cube".
        colour: Cube colour ("yellow", "blue" or "red").
        xy: Cube centre to pick from, robot-base metres.
        yaw_deg: Cube yaw in degrees, folded to [-45, 45).
        pose_confidence: Detector confidence, 0 to 1; the twin refuses grasps below MIN_CONF.
        spread_dir: Unit vector [x, y, z] the jaws open along at the pick: the cube's local +Y,
            so the fingers close across two opposite faces.
        pick_z: Fingertip height for the grasp, robot-base metres.
        stack_level: Position in the tower, 0 at the bottom.
        place_xy: Tower site centre, robot-base metres.
        place_z: Fingertip height for the release, robot-base metres.
        expected_centre_z: Height of the placed cube's centre, robot-base metres.
        place_spread_dir: Unit vector [x, y, z] the jaws open along at the tower.
        place_yaw_deg: Cube yaw on the tower in degrees.
    """

    detection_id: int
    category: Category
    label: str
    colour: str
    xy: list[float]
    yaw_deg: float
    pose_confidence: float
    spread_dir: list[float]
    pick_z: float
    stack_level: int
    place_xy: list[float]
    place_z: float
    expected_centre_z: float
    place_spread_dir: list[float]
    place_yaw_deg: float


class SortPlan(TypedDict):
    """The sort_plan.json file that fanuc_er4ia_sim/sort_scene.py validates and executes.

    Attributes:
        source: Who made the plan; the twin accepts "SmartGrasp".
        detector: The --detector spec that found the cubes.
        gripper_model: Gripper the plan was made for.
        task: Always "cube_stack".
        stack_colours_bottom_to_top: The twin's tower colour order.
        open_gap_m: Jaw opening in metres; the twin rejects plans made for another opening.
        picks: The cube moves, bottom of the tower first.
    """

    source: str
    detector: str
    gripper_model: str
    task: str
    stack_colours_bottom_to_top: list[str]
    open_gap_m: float
    picks: list[Pick]


class CellRow(TypedDict):
    """One FANUC target for the real cell: an XYZWPR pose in UF0 with UTOOL1 (see CELL_ASSUMPTIONS).

    Attributes:
        cube: detection_id of the cube being moved.
        colour: Cube colour.
        step: approach, descend, grip, lift, carry, lower, release or retreat.
        X_mm: Target X in UF0, millimetres.
        Y_mm: Target Y in UF0, millimetres.
        Z_mm: Target Z in UF0, millimetres.
        W_deg: Tool rotation about X in degrees; 180 points the tool straight down.
        P_deg: Tool rotation about Y in degrees.
        R_deg: Tool rotation about Z in degrees: cube yaw at the pick, place_yaw_deg at the tower.
        gripper: "open" or "close" at this target, or "" to leave the gripper as it is.
        within_limits: Whether the target is in reach and inside SAFE_Z.
    """

    cube: int
    colour: str
    step: str
    X_mm: float
    Y_mm: float
    Z_mm: float
    W_deg: float
    P_deg: float
    R_deg: float
    gripper: str
    within_limits: bool


class CellPlan(TypedDict):
    """The cell_plan.json file: the real-cell dry run for an operator to review.

    Attributes:
        assumptions: CELL_ASSUMPTIONS, the frames the targets are given in.
        all_within_limits: Whether every row is within limits.
        rows: Every target, in motion order.
    """

    assumptions: str
    all_within_limits: bool
    rows: list[CellRow]


class _Cube(TypedDict):
    """A confident, reachable cube detection that could go on the tower.

    Attributes:
        detection_id: Index of the cube in the detections.
        colour: Cube colour ("yellow", "blue" or "red").
        xy: Centre in robot-base metres.
        yaw_deg: Yaw in degrees, folded to [-45, 45).
        conf: Detector confidence, 0 to 1.
    """

    detection_id: int
    colour: str
    xy: list[float]
    yaw_deg: float
    conf: float


def reachable(x: float, y: float) -> bool:
    """Check that the arm can hover over a point: inside the REACH box and within REACH_RADIUS_M.

    Args:
        x: Robot-base x in metres.
        y: Robot-base y in metres.

    Returns:
        True if the point is in reach.
    """
    return REACH[0][0] <= x <= REACH[0][1] and REACH[1][0] <= y <= REACH[1][1] and np.hypot(x, y) <= REACH_RADIUS_M


def _on_stack_site(detection: Detection, bev: Bev) -> bool:
    """Check whether a detection of any class sits on the tower site.

    Args:
        detection: Detection in BEV pixels.
        bev: View the detection was made in.

    Returns:
        True if its centre is within SITE_CLEAR_M of twin.CUBE_STACK_XY.
    """
    (x, y), _ = pose(detection.corners, bev)
    return np.hypot(x - twin.CUBE_STACK_XY[0], y - twin.CUBE_STACK_XY[1]) < SITE_CLEAR_M


def _stackable_cubes(detections: Sequence[Detection], bev: Bev) -> list[_Cube]:
    """Collect the cube detections that are confident (MIN_CONF) and reachable.

    Args:
        detections: Detections in BEV pixels.
        bev: View the detections were made in.

    Returns:
        The stackable cubes, in detection order.
    """
    cubes: list[_Cube] = []
    for detection_id, detection in enumerate(detections):
        (x, y), yaw = pose(detection.corners, bev)
        if detection.cls.startswith("cube_") and detection.conf >= MIN_CONF and reachable(x, y):
            cubes.append(
                _Cube(
                    detection_id=detection_id,
                    colour=detection.cls.removeprefix("cube_"),
                    xy=[float(x), float(y)],
                    yaw_deg=float(yaw),
                    conf=float(detection.conf),
                )
            )
    return cubes


def _pick(cube: _Cube, level: int) -> Pick:
    """Plan the move of one cube from the table onto the tower.

    Args:
        cube: The cube to move.
        level: Tower level it goes to, 0 at the bottom.

    Returns:
        The move in the twin's sort_plan format.
    """
    yaw_rad = np.radians(cube["yaw_deg"])
    return Pick(
        detection_id=cube["detection_id"],
        category="cube",
        label=f"{cube['colour']} cube",
        colour=cube["colour"],
        xy=cube["xy"],
        yaw_deg=cube["yaw_deg"],
        pose_confidence=cube["conf"],
        spread_dir=[float(-np.sin(yaw_rad)), float(np.cos(yaw_rad)), 0.0],
        pick_z=twin.SURFACE_DZ + twin.CUBE_TIP_CLEARANCE,
        stack_level=level,
        place_xy=list(twin.CUBE_STACK_XY),
        place_z=twin.SURFACE_DZ + level * twin.CUBE_EDGE + twin.CUBE_TIP_CLEARANCE,
        expected_centre_z=twin.SURFACE_DZ + (level + 0.5) * twin.CUBE_EDGE,
        place_spread_dir=[0.0, 1.0, 0.0],
        place_yaw_deg=0.0,
    )


def plan_stack(detections: Sequence[Detection], bev: Bev) -> list[Pick]:
    """Plan a tower of confident, reachable cubes in the twin's colour order.

    The order is twin.CUBE_STACK_COLOURS (yellow, yellow, blue, blue, red, red); each level takes
    the most confident unused cube of its colour. Missing colours are skipped, so two yellow cubes
    make a two-cube stack for a cell dry run; the twin's validate_plan accepts only the full six.
    Plans nothing unless the stack site is clear: one top-down frame cannot tell how tall a tower
    already is.

    Args:
        detections: Detections in BEV pixels; a pick's detection_id is its index here.
        bev: View the detections were made in.

    Returns:
        The moves, bottom of the tower first; empty if the stack site is occupied.
    """
    if any(_on_stack_site(detection, bev) for detection in detections):
        return []
    cubes = _stackable_cubes(detections, bev)
    picks: list[Pick] = []
    for colour in twin.CUBE_STACK_COLOURS:
        options = [cube for cube in cubes if cube["colour"] == colour]
        if not options:
            continue
        best = max(options, key=lambda cube: cube["conf"])
        cubes.remove(best)
        picks.append(_pick(best, level=len(picks)))
    return picks


def sort_plan(picks: list[Pick], detector: str) -> SortPlan:
    """Wrap the picks in the twin's plan format, which fanuc_er4ia_sim/sort_scene.py executes.

    Args:
        picks: Moves from plan_stack.
        detector: The --detector spec, recorded in the plan.

    Returns:
        The plan, as written to sort_plan.json.
    """
    return SortPlan(
        source="SmartGrasp",
        detector=detector,
        gripper_model=twin.GRIPPER_MODEL,
        task="cube_stack",
        stack_colours_bottom_to_top=list(twin.CUBE_STACK_COLOURS),
        open_gap_m=2 * twin.JAW_OPEN,
        picks=picks,
    )


CELL_ASSUMPTIONS: str = (
    f"Targets are UF0 (robot base + {1000 * UF0_Z_M:.0f} mm) with "
    f"UTOOL1 = (0, 0, {1000 * (twin.TOOL_LENGTH - twin.FLANGE_OFFSET):.0f}, 0, 0, 0) "  # faceplate -> fingertip
    "and the fingers closing along tool Y, so a top-down grasp is W=180, P=0, R=cube yaw. "
    "Confirm at CSI before any motion."
)


def _waypoints(pick: Pick) -> list[tuple[str, tuple[float, ...], float, str]]:
    """List the eight robot-base targets that carry one cube from the table to the tower.

    Args:
        pick: The cube's move.

    Returns:
        (step, (x, y, z) in robot-base metres, tool yaw in degrees, gripper action) per target.
    """
    at_pick, at_place = (*pick["xy"], pick["pick_z"]), (*pick["place_xy"], pick["place_z"])
    above_pick = (*pick["xy"], pick["pick_z"] + APPROACH_M)
    above_place = (*pick["place_xy"], pick["place_z"] + APPROACH_M)
    pick_yaw, place_yaw = pick["yaw_deg"], pick["place_yaw_deg"]
    return [
        ("approach", above_pick, pick_yaw, "open"),
        ("descend", at_pick, pick_yaw, ""),
        ("grip", at_pick, pick_yaw, "close"),
        ("lift", above_pick, pick_yaw, ""),
        ("carry", above_place, place_yaw, ""),
        ("lower", at_place, place_yaw, ""),
        ("release", at_place, place_yaw, "open"),
        ("retreat", above_place, place_yaw, ""),
    ]


def cell_rows(picks: Sequence[Pick]) -> list[CellRow]:
    """Turn the robot-base waypoints into FANUC XYZWPR rows (mm, deg); see CELL_ASSUMPTIONS.

    Args:
        picks: Moves from plan_stack.

    Returns:
        Eight rows per pick, in motion order.
    """
    rows: list[CellRow] = []
    for pick in picks:
        for step, (x, y, z), r_deg, gripper in _waypoints(pick):
            within_limits = reachable(x, y) and SAFE_Z[0] <= z <= SAFE_Z[1]
            rows.append(
                CellRow(
                    cube=pick["detection_id"],
                    colour=pick["colour"],
                    step=step,
                    X_mm=round(1000 * x, 1),
                    Y_mm=round(1000 * y, 1),
                    Z_mm=round(1000 * (z - UF0_Z_M), 1),
                    W_deg=180.0,
                    P_deg=0.0,
                    R_deg=round(r_deg, 1),
                    gripper=gripper,
                    within_limits=bool(within_limits),
                )
            )
    return rows


def draw(bev_img: Image, detections: Sequence[Detection], picks: Sequence[Pick], bev: Bev) -> Image:
    """Draw the detections, the tower levels and the tower site on a copy of the BEV image.

    Args:
        bev_img: BEV image the detector saw, BGR uint8.
        detections: Detections in BEV pixels, outlined in their class colour (BGR).
        picks: Planned moves, each labelled at the cube with its tower level, 1 at the bottom.
        bev: View the image was made in.

    Returns:
        The annotated copy, the same size as bev_img.
    """
    out = bev_img.copy()
    for detection in detections:
        corners_px = np.round(detection.corners).astype(np.int32)
        colour = BGR.get(detection.cls, (255, 255, 255))
        cv2.polylines(out, [corners_px], True, colour, 1, cv2.LINE_AA)
    for pick in picks:
        u, v = bev.to_px(pick["xy"])
        cv2.putText(
            out,
            str(pick["stack_level"] + 1),
            (int(u) + 12, int(v) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    u, v = bev.to_px(twin.CUBE_STACK_XY)
    cv2.drawMarker(out, (int(u), int(v)), (0, 160, 0), cv2.MARKER_CROSS, 24, 2)
    return out


def _write_cell_plan(out_dir: Path, rows: list[CellRow]) -> None:
    """Write the real-cell dry run as cell_plan.csv and cell_plan.json.

    Args:
        out_dir: Output folder.
        rows: FANUC targets from cell_rows.
    """
    with open(out_dir / "cell_plan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    cell_plan = CellPlan(
        assumptions=CELL_ASSUMPTIONS,
        all_within_limits=all(row["within_limits"] for row in rows),
        rows=rows,
    )
    (out_dir / "cell_plan.json").write_text(json.dumps(cell_plan, indent=2))


def deploy(
    image: str | Path, calib: str | Path, detector: str, out: str | Path, bev: Bev = Bev()
) -> tuple[list[Detection], list[Pick], list[CellRow]]:
    """Detect cubes in one camera frame, plan the tower and write every output file into out.

    Args:
        image: Table camera frame.
        calib: Sim camera.json or real calib.json for that camera.
        detector: Detector spec for load_detector: yolo:<weights>, hsv or vlm[:<dataset>].
        out: Output folder; created if missing.
        bev: View to detect in.

    Returns:
        (detections in BEV pixels, picks for the twin, FANUC rows for the real cell).

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bev_img = bev.warp(read_image(image), load_homography(calib, bev))
    detections = load_detector(detector, bev)(bev_img)
    picks = plan_stack(detections, bev)
    rows = cell_rows(picks)
    cv2.imwrite(str(out_dir / "bev.png"), bev_img)
    cv2.imwrite(str(out_dir / "overlay.png"), draw(bev_img, detections, picks, bev))
    (out_dir / "detections.json").write_text(json.dumps([d.to_json(bev) for d in detections], indent=2))
    (out_dir / "sort_plan.json").write_text(json.dumps(sort_plan(picks, detector), indent=2))
    _write_cell_plan(out_dir, rows)
    return detections, picks, rows


def main() -> None:
    """Run deploy from the command line and print the cell plan for operator review."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="table camera frame")
    parser.add_argument("--calib", required=True, help="sim camera.json or real calib.json")
    parser.add_argument("--detector", required=True, help="yolo:<weights>, hsv or vlm[:<dataset>]")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    detections, picks, rows = deploy(args.image, args.calib, args.detector, args.out)
    print(f"[deploy] {len(detections)} detections, {len(picks)} picks -> {args.out}")
    if not picks:
        print("  no plan: check overlay.png. Cubes must be confident and within reach, and the stack site clear.")
    print(f"  {CELL_ASSUMPTIONS}")
    for row in rows:
        flag = "" if row["within_limits"] else "  <-- OUTSIDE LIMITS"
        print(
            f"  cube {row['cube']} {row['step']:>8}  X {row['X_mm']:7.1f}  Y {row['Y_mm']:7.1f}"
            f"  Z {row['Z_mm']:7.1f}  R {row['R_deg']:6.1f}  {row['gripper']}{flag}"
        )


if __name__ == "__main__":
    main()

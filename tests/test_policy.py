"""Host tests for the cell policy: the real camera's homography and the targets it streams.

uv run pytest tests/test_policy.py
"""

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from fake_twin import STACK, camera_json, render
from smartgrasp import policy
from smartgrasp.deploy import CellRow
from smartgrasp.geometry import (
    CameraJson,
    CameraModel,
    Image,
    SceneObject,
    apply_homography,
    cell_homography,
    fold90,
    twin,
)

CELL_TCP_BOX: list[tuple[float, float]] = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]  # mock_cell.TCP_BOX
DWELL: int = round(policy.Policy.DWELL_S / policy.Policy.DT)


def cell_table(cam_json: CameraJson) -> dict[str, Any]:
    """Describe a twin camera as the organisers' cameras.json describes the real one.

    Their camera frame looks along +z with +y down; Isaac's looks along -z with +y up.

    Args:
        cam_json: A twin camera.json.

    Returns:
        A "table" entry for cameras.json, without lens distortion (k1 = 0).
    """
    model = CameraModel(cam_json)
    base_from_cam = np.eye(4)
    base_from_cam[:3, :3] = model.R.T @ np.diag([1.0, -1.0, -1.0])
    base_from_cam[:3, 3] = model.eye - (0.0, 0.0, twin.TABLE_TOP_Z)
    return {
        "T_base_cam": base_from_cam.tolist(),
        "fx": model.W / 2 / model.tan_h,
        "k1": 0.0,
        "size": [model.W, model.H],
    }


def observation(layout: list[SceneObject], tcp_pose: list[float] | None) -> policy.Observation:
    """Render a layout through the twin's table camera, as the cell would send it.

    Args:
        layout: Objects on the table.
        tcp_pose: Fingertip pose to report, or None as the mock does before any tcp target.

    Returns:
        An observation with an RGB table image, an open gripper and trial "t0".
    """
    bgr = render(layout, camera_json(), np.random.default_rng(0))
    return {
        "images": {"table": cast(Image, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))},
        "joints": np.zeros(7, np.float32),
        "tcp_pose": tcp_pose,
        "trial": {"trial_id": "t0"},
    }


@pytest.fixture
def cell_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> policy.Policy:
    """Build an HSV policy whose calibration is the twin camera and whose outputs go to tmp_path.

    Args:
        tmp_path: pytest's temporary folder for this test.
        monkeypatch: pytest's attribute patcher, undone after the test.

    Returns:
        The policy, ready for its first rollout.
    """
    (tmp_path / "cameras.json").write_text(json.dumps({"table": cell_table(camera_json())}))
    monkeypatch.setattr(policy.Policy, "CALIBRATION", tmp_path / "cameras.json")
    monkeypatch.setattr(policy.Policy, "RUNS", tmp_path / "runs")
    return policy.Policy("hsv")


def test_cell_homography_matches_the_twin_camera() -> None:
    """Without lens distortion, the organisers' camera format maps pixels as the twin does."""
    cam_json = camera_json()
    model = CameraModel(cam_json)
    H = cell_homography(cell_table(cam_json))
    xy = np.random.default_rng(0).uniform((0.2, -0.25), (0.5, 0.3), (20, 2))
    uv = [model.project((x, y, cam_json["table_top_z"] + twin.CUBE_EDGE)) for x, y in xy]
    assert np.allclose(apply_homography(H, uv), xy, atol=1e-6)


def test_policy_streams_the_tower_in_small_safe_steps(cell_policy: policy.Policy, tmp_path: Path) -> None:
    """The six-cube plan reaches the cell in small steps within its limits, gripping each cube.

    Args:
        cell_policy: HSV policy on the twin camera.
        tmp_path: pytest's temporary folder, where the policy writes its plan.
    """
    obs = observation(STACK, policy.HOME_TCP)
    targets: list[policy.Target] = []
    while not targets or cell_policy.queue:
        chunk = cell_policy.act(obs)
        assert 1 <= len(chunk) <= policy.MAX_CHUNK and all(len(target) == 7 for target in chunk)
        targets += chunk
    assert cell_policy.act(obs) == [targets[-1]] * DWELL, "holds still once the plan is sent"

    path = np.array([policy.HOME_TCP + [0.0]] + targets)
    for axis, (low, high) in enumerate(CELL_TCP_BOX):
        assert low <= path[:, axis].min() and path[:, axis].max() <= high
    assert np.abs(path[:, 3:6]).max() <= 360
    step_mm = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    assert step_mm.max() <= policy.Policy.SPEED_MM_S * policy.Policy.DT + 1e-9

    grips = np.flatnonzero(np.diff(path[:, 6]) > 0) + 1
    picks = json.loads((tmp_path / "runs" / "t0" / "sort_plan.json").read_text())["picks"]
    assert len(grips) == len(picks) == 6
    for index, pick in zip(grips, picks, strict=True):
        assert np.all(path[index : index + DWELL + 1] == path[index]), "holds while the jaws close"
        assert np.allclose(path[index, :2], np.multiply(pick["xy"], 1000), atol=0.05)  # rows are 0.1 mm
        assert abs(fold90(path[index, 5] - pick["yaw_deg"])) <= 0.05


def test_policy_holds_still_when_it_cannot_plan(cell_policy: policy.Policy) -> None:
    """An empty table gives no plan: the arm holds at home, and the next call tries again.

    Args:
        cell_policy: HSV policy on the twin camera.
    """
    assert cell_policy.act(observation([], None)) == [policy.HOME_TCP + [0.0]] * DWELL
    assert not cell_policy.planned


def test_trajectory_turns_the_wrist_the_short_way() -> None:
    """W turns modulo 360, R modulo 90, and an unchanged gripper adds no dwell."""
    row = CellRow(
        cube=0,
        colour="red",
        step="approach",
        X_mm=300.0,
        Y_mm=0.0,
        Z_mm=-20.0,
        W_deg=180.0,
        P_deg=0.0,
        R_deg=10.0,
        gripper="open",
        within_limits=True,
    )
    start = [300.0, 0.0, -20.0, -180.0, 0.0, 170.0, 0.0]
    assert policy.trajectory(start, [row], step_mm=8.0, dwell=DWELL) == [[300.0, 0.0, -20.0, -180.0, 0.0, 190.0, 0.0]]

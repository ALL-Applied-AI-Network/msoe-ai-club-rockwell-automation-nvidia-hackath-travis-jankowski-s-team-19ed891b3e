"""Host tests (no GPU, no training): geometry, labels, planning and metrics.

uv run pytest
"""

import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pytest

from fake_twin import STACK, camera_json, write_frames
from smartgrasp import dataset, deploy, evaluate
from smartgrasp.detectors import Detection
from smartgrasp.geometry import (
    Bev,
    CameraJson,
    CameraModel,
    Homography,
    Image,
    SceneObject,
    apply_homography,
    fold90,
    label_corners,
    load_homography,
    pixel,
    pose,
    twin,
)
from smartgrasp.scene import _clearance, sample_layout

BEV: Bev = Bev()


@pytest.fixture
def cam(tmp_path: Path) -> Path:
    """Write the fake twin's table camera to a camera.json.

    Args:
        tmp_path: pytest's temporary folder for this test.

    Returns:
        Path of the written camera.json.
    """
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(camera_json()))
    return path


def oracle(objects: Iterable[SceneObject], cam_json: CameraJson, H: Homography) -> list[Detection]:
    """Make perfect detections straight from ground truth.

    Args:
        objects: Ground-truth cubes.
        cam_json: Camera the BEV image was warped from.
        H: Homography from that camera's pixels to robot-base XY on the cube-top plane.

    Returns:
        One detection per object at its exact label box, with confidence 0.95.
    """
    return [Detection(f"cube_{obj['colour']}", label_corners(obj, cam_json, H, BEV), 0.95) for obj in objects]


def test_homography_matches_camera_model(cam: Path) -> None:
    """The fitted homography agrees with the camera model on the cube-top plane.

    Args:
        cam: Path of the fake twin's camera.json.
    """
    cam_json: CameraJson = json.loads(cam.read_text())
    model, H = CameraModel(cam_json), load_homography(cam, BEV)
    xy = np.random.default_rng(0).uniform((0.2, -0.25), (0.5, 0.3), (20, 2))
    uv = [pixel(model, (x, y, cam_json["table_top_z"] + twin.CUBE_EDGE)) for x, y in xy]
    assert np.allclose(apply_homography(H, uv), xy, atol=1e-6)


def test_bev_view_is_inside_the_camera_image(cam: Path) -> None:
    """Every corner of the BEV view on the table falls inside the camera image.

    Args:
        cam: Path of the fake twin's camera.json.
    """
    model = CameraModel(json.loads(cam.read_text()))
    for x in BEV.x_range:
        for y in BEV.y_range:
            u, v = model.project((x, y, model.table_z))
            assert 0 <= u < model.W and 0 <= v < model.H


def test_cube_label_is_the_true_square_and_pose_round_trips(cam: Path) -> None:
    """A cube's label box is an exact square of cube size, and pose() recovers its position and yaw.

    Args:
        cam: Path of the fake twin's camera.json.
    """
    cam_json: CameraJson = json.loads(cam.read_text())
    H = load_homography(cam, BEV)
    for obj in STACK:
        px = label_corners(obj, cam_json, H, BEV)
        sides = np.linalg.norm(np.roll(px, -1, 0) - px, axis=1)
        assert np.allclose(sides, BEV.cube_px, atol=1e-6)  # 20 px
        xy, yaw = pose(px, BEV)
        assert np.allclose(xy, obj["xy"], atol=1e-9)
        assert abs(fold90(yaw - obj["yaw_deg"])) < 1e-4


def test_layouts_never_overlap() -> None:
    """In 50 sampled layouts, no object's centre lies inside another object's footprint."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        layout = sample_layout(rng)
        for i, obj in enumerate(layout):
            for other in layout[i + 1 :]:
                assert _clearance(obj["xy"], other) > 0 and _clearance(other["xy"], obj) > 0


def test_dataset_labels_match_ground_truth(tmp_path: Path) -> None:
    """Every frame is built, and each label file holds exactly the frame's true cubes.

    Args:
        tmp_path: pytest's temporary folder for this test.
    """
    rng = np.random.default_rng(1)
    write_frames(tmp_path / "raw", [sample_layout(rng) for _ in range(8)])
    manifest = dataset.build(tmp_path / "raw", tmp_path / "bev")
    assert sum(manifest["frames"].values()) == 8
    for label in (tmp_path / "bev" / "labels").rglob("*.txt"):
        frame: dict[str, list[SceneObject]] = json.loads((tmp_path / "raw" / f"{label.stem}.json").read_text())
        truth = [obj for obj in frame["objects"] if obj["category"] == "cube"]
        found = [pose(px, BEV) for name, px in dataset.read_labels(label, BEV.size) if name.startswith("cube_")]
        assert len(found) == len(truth)
        for obj in truth:
            assert min(np.linalg.norm(xy - obj["xy"]) for xy, _ in found) < 1e-5


def test_plan_passes_the_twins_own_validator(cam: Path) -> None:
    """A six-cube plan from perfect detections passes the twin's validate_plan and cell limits.

    Args:
        cam: Path of the fake twin's camera.json.
    """
    from sort_scene import validate_plan  # fanuc_er4ia_sim/, on sys.path once geometry is imported

    picks = deploy.plan_stack(oracle(STACK, json.loads(cam.read_text()), load_homography(cam, BEV)), BEV)
    validate_plan(deploy.sort_plan(picks, "oracle"))  # exactly what deploy writes to sort_plan.json
    assert [pick["colour"] for pick in picks] == list(twin.CUBE_STACK_COLOURS)
    rows = deploy.cell_rows(picks)
    assert len(rows) == 8 * 6 and all(row["within_limits"] for row in rows)
    grips = [row for row in rows if row["step"] == "grip"]
    assert [row["R_deg"] for row in grips] == [round(pick["yaw_deg"], 1) for pick in picks]


def test_plan_skips_what_it_cannot_do_safely(cam: Path) -> None:
    """The planner skips missing colours and unreachable cubes, and plans nothing on a busy site.

    Args:
        cam: Path of the fake twin's camera.json.
    """
    cam_json: CameraJson = json.loads(cam.read_text())
    H = load_homography(cam, BEV)

    def plan(objects: list[SceneObject]) -> list[deploy.Pick]:
        """Plan a stack from perfect detections of the given objects.

        Args:
            objects: Ground-truth objects on the table.

        Returns:
            The planned picks, in stacking order.
        """
        return deploy.plan_stack(oracle(objects, cam_json, H), BEV)

    yellow = [obj for obj in STACK if obj["colour"] == "yellow"]
    assert [pick["stack_level"] for pick in plan(yellow)] == [0, 1], "missing colours are skipped"
    far: SceneObject = {
        "category": "cube",
        "colour": "blue",
        "xy": [0.51, 0.26],  # inside the reach box, but 0.57 m from the base
        "yaw_deg": 0.0,
    }
    assert [pick["colour"] for pick in plan(yellow + [far])] == ["yellow", "yellow"], "out of reach"
    tower: SceneObject = {
        "category": "cube",
        "colour": "red",
        "xy": [twin.CUBE_STACK_XY[0] - 0.03, twin.CUBE_STACK_XY[1]],
        "yaw_deg": 0.0,
    }
    assert plan(STACK + [tower]) == [], "an occupied stack site blocks the whole plan"


def test_perfect_detector_scores_perfectly(tmp_path: Path) -> None:
    """A detector that returns the labels themselves gets full recall, precision and graspability.

    Args:
        tmp_path: pytest's temporary folder for this test.
    """
    rng = np.random.default_rng(2)
    write_frames(tmp_path / "raw", [sample_layout(rng) for _ in range(12)])
    dataset.build(tmp_path / "raw", tmp_path / "bev")
    labels = iter(sorted((tmp_path / "bev" / "labels" / "train").glob("*.txt")))

    def perfect(image: Image) -> list[Detection]:
        """Return the next training label's boxes as detections, ignoring the image.

        evaluate_perception visits the split's images in sorted order, the order of `labels`.

        Args:
            image: BEV image (unused).

        Returns:
            One detection per labelled object, with confidence 1.0.
        """
        return [Detection(name, px, 1.0) for name, px in dataset.read_labels(next(labels), BEV.size)]

    metrics = evaluate.evaluate_perception(tmp_path / "bev", perfect, split="train")
    xy_error_mm = metrics["xy_error_mm"]
    assert xy_error_mm is not None
    assert (
        metrics["recall"] == 1
        and metrics["precision"] == 1
        and xy_error_mm["max"] < 1e-3
        and metrics["graspable_rate"] == 1
    )


def test_hsv_baseline_and_deploy_end_to_end(tmp_path: Path) -> None:
    """The HSV detector finds all six STACK cubes, and deploy writes every output file.

    Args:
        tmp_path: pytest's temporary folder for this test.
    """
    write_frames(tmp_path / "raw", [STACK])
    detections, picks, rows = deploy.deploy(
        tmp_path / "raw" / "000000.jpg", tmp_path / "raw" / "camera.json", "hsv", tmp_path / "run"
    )
    assert len(picks) == 6, "colour thresholding finds all six cubes in a clean frame"
    for pick in picks:  # side faces bias the classical estimate, but stay within a cube edge
        truth = min(STACK, key=lambda obj: np.linalg.norm(np.subtract(obj["xy"], pick["xy"])))
        assert np.linalg.norm(np.subtract(truth["xy"], pick["xy"])) < 0.008
    for name in ("bev.png", "overlay.png", "detections.json", "sort_plan.json", "cell_plan.csv", "cell_plan.json"):
        assert (tmp_path / "run" / name).exists()
    bev_png = cv2.imread(str(tmp_path / "run" / "bev.png"))
    assert bev_png is not None
    assert bev_png.shape[1::-1] == BEV.size


def test_trial_and_episode_metrics(tmp_path: Path) -> None:
    """Trial-log and closed-loop episode metrics match hand-computed values.

    Args:
        tmp_path: pytest's temporary folder for this test.
    """
    log = tmp_path / "trials.csv"
    log.write_text("trial,attempt,grasp_ok,place_ok,cycle_s,notes\n1,1,1,1,10,\n2,1,0,0,9,\n2,2,1,1,11,\n3,1,0,0,9,\n")
    trials = evaluate.evaluate_trials(log)
    assert trials == {
        "trials": 3,
        "first_attempt_grasp_success": 0.3333,
        "recovery_rate": 0.5,
        "cycle_s_per_pick": 10.5,
    }

    run = tmp_path / "ep"
    run.mkdir()
    (run / "sort_plan.json").write_text(json.dumps({"picks": [{}] * 6}))
    verified = {"pickup_verified": True, "placement_verified": True}
    failed = {"pickup_verified": False, "placement_verified": False}
    report = {"picks": [verified, verified, failed], "frames": 828, "task_complete": False}
    (run / "execution_report.json").write_text(json.dumps(report))
    episodes = evaluate.evaluate_episodes([run, tmp_path / "missing"])
    assert episodes["episodes"] == 2 and episodes["full_plan_rate"] == 0.5
    assert episodes["first_attempt_grasp_success"] == 0.6667
    assert episodes["placement_success"] == 0.6667 and episodes["recovery_rate"] is None

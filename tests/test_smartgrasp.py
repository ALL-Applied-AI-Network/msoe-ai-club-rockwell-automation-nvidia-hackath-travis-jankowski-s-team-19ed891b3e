"""Host tests (no GPU, no training): geometry, labels, planning and metrics.

    uv run pytest
"""

import json

import cv2
import numpy as np
import pytest
from fake_twin import STACK, camera_json, write_frames

from smartgrasp import dataset, deploy, evaluate
from smartgrasp.detectors import Detection
from smartgrasp.geometry import Bev, CameraModel, apply_homography, fold90, label_corners, load_homography, pixel, pose, twin
from smartgrasp.scene import _clearance, sample_layout

BEV = Bev()


@pytest.fixture
def cam(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(camera_json()))
    return path


def oracle(objects, cam_json, H):
    """Perfect detections straight from ground truth."""
    return [Detection(f"cube_{o['colour']}", label_corners(o, cam_json, H, BEV), 0.95) for o in objects]


def test_homography_matches_camera_model(cam):
    cam_json = json.loads(cam.read_text())
    model, H = CameraModel(cam_json), load_homography(cam, BEV)
    xy = np.random.default_rng(0).uniform((0.2, -0.25), (0.5, 0.3), (20, 2))
    uv = [pixel(model, (x, y, cam_json["table_top_z"] + twin.CUBE_EDGE)) for x, y in xy]
    assert np.allclose(apply_homography(H, uv), xy, atol=1e-6)


def test_bev_view_is_inside_the_camera_image(cam):
    model = CameraModel(json.loads(cam.read_text()))
    for x in BEV.x_range:
        for y in BEV.y_range:
            u, v = model.project((x, y, model.table_z))
            assert 0 <= u < model.W and 0 <= v < model.H


def test_cube_label_is_the_true_square_and_pose_round_trips(cam):
    cam_json, H = json.loads(cam.read_text()), load_homography(cam, BEV)
    for obj in STACK:
        px = label_corners(obj, cam_json, H, BEV)
        sides = np.linalg.norm(np.roll(px, -1, 0) - px, axis=1)
        assert np.allclose(sides, BEV.cube_px, atol=1e-6)  # 20 px
        xy, yaw = pose(px, BEV)
        assert np.allclose(xy, obj["xy"], atol=1e-9)
        assert abs(fold90(yaw - obj["yaw_deg"])) < 1e-4


def test_layouts_never_overlap():
    rng = np.random.default_rng(0)
    for _ in range(50):
        layout = sample_layout(rng)
        for i, a in enumerate(layout):
            for b in layout[i + 1 :]:
                assert _clearance(a["xy"], b) > 0 and _clearance(b["xy"], a) > 0


def test_dataset_labels_match_ground_truth(tmp_path):
    rng = np.random.default_rng(1)
    write_frames(tmp_path / "raw", [sample_layout(rng) for _ in range(8)])
    manifest = dataset.build(tmp_path / "raw", tmp_path / "bev")
    assert sum(manifest["frames"].values()) == 8
    for label in (tmp_path / "bev" / "labels").rglob("*.txt"):
        truth = [o for o in json.loads((tmp_path / "raw" / f"{label.stem}.json").read_text())["objects"] if o["category"] == "cube"]
        found = [pose(px, BEV) for c, px in dataset.read_labels(label, BEV.size) if c.startswith("cube_")]
        assert len(found) == len(truth)
        for o in truth:
            assert min(np.linalg.norm(xy - o["xy"]) for xy, _ in found) < 1e-5


def test_plan_passes_the_twins_own_validator(cam):
    from sort_scene import validate_plan  # fanuc_er4ia_sim/sort_scene.py

    picks = deploy.plan_stack(oracle(STACK, json.loads(cam.read_text()), load_homography(cam, BEV)), BEV)
    validate_plan(deploy.sort_plan(picks, "oracle"))  # exactly what deploy writes to sort_plan.json
    assert [p["colour"] for p in picks] == list(twin.CUBE_STACK_COLOURS)
    rows = deploy.cell_rows(picks)
    assert len(rows) == 8 * 6 and all(r["within_limits"] for r in rows)
    grips = [r for r in rows if r["step"] == "grip"]
    assert [r["R_deg"] for r in grips] == [round(p["yaw_deg"], 1) for p in picks]


def test_plan_skips_what_it_cannot_do_safely(cam):
    cam_json, H = json.loads(cam.read_text()), load_homography(cam, BEV)
    plan = lambda objects: deploy.plan_stack(oracle(objects, cam_json, H), BEV)  # noqa: E731
    yellow = [o for o in STACK if o["colour"] == "yellow"]
    assert [p["stack_level"] for p in plan(yellow)] == [0, 1], "missing colours are skipped"
    far = {"category": "cube", "colour": "blue", "xy": [0.51, 0.26], "yaw_deg": 0.0}  # in the box, 0.57 m out
    assert [p["colour"] for p in plan(yellow + [far])] == ["yellow", "yellow"], "out of reach"
    tower = {"category": "cube", "colour": "red", "xy": [twin.CUBE_STACK_XY[0] - 0.03, twin.CUBE_STACK_XY[1]], "yaw_deg": 0.0}
    assert plan(STACK + [tower]) == [], "an occupied stack site blocks the whole plan"


def test_perfect_detector_scores_perfectly(tmp_path):
    rng = np.random.default_rng(2)
    write_frames(tmp_path / "raw", [sample_layout(rng) for _ in range(12)])
    dataset.build(tmp_path / "raw", tmp_path / "bev")
    labels = iter(sorted((tmp_path / "bev" / "labels" / "train").glob("*.txt")))
    perfect = lambda image: [Detection(c, px, 1.0) for c, px in dataset.read_labels(next(labels), BEV.size)]  # noqa: E731
    m = evaluate.evaluate_perception(tmp_path / "bev", perfect, split="train")
    assert m["recall"] == 1 and m["precision"] == 1 and m["xy_error_mm"]["max"] < 1e-3 and m["graspable_rate"] == 1


def test_hsv_baseline_and_deploy_end_to_end(tmp_path):
    write_frames(tmp_path / "raw", [STACK])
    detections, picks, rows = deploy.deploy(tmp_path / "raw" / "000000.jpg", tmp_path / "raw" / "camera.json", "hsv", tmp_path / "run")
    assert len(picks) == 6, "colour thresholding finds all six cubes in a clean frame"
    for p in picks:  # side faces bias the classical estimate, but stay within a cube edge
        truth = min(STACK, key=lambda o: np.linalg.norm(np.subtract(o["xy"], p["xy"])))
        assert np.linalg.norm(np.subtract(truth["xy"], p["xy"])) < 0.008
    for name in ("bev.png", "overlay.png", "detections.json", "sort_plan.json", "cell_plan.csv", "cell_plan.json"):
        assert (tmp_path / "run" / name).exists()
    assert cv2.imread(str(tmp_path / "run" / "bev.png")).shape[1::-1] == BEV.size


def test_trial_and_episode_metrics(tmp_path):
    log = tmp_path / "trials.csv"
    log.write_text("trial,attempt,grasp_ok,place_ok,cycle_s,notes\n1,1,1,1,10,\n2,1,0,0,9,\n2,2,1,1,11,\n3,1,0,0,9,\n")
    m = evaluate.evaluate_trials(log)
    assert m == {"trials": 3, "first_attempt_grasp_success": 0.3333, "recovery_rate": 0.5, "cycle_s_per_pick": 10.5}

    run = tmp_path / "ep"
    run.mkdir()
    (run / "sort_plan.json").write_text(json.dumps({"picks": [{}] * 6}))
    picks = [{"pickup_verified": True, "placement_verified": True}] * 2 + [{"pickup_verified": False, "placement_verified": False}]
    (run / "execution_report.json").write_text(json.dumps({"picks": picks, "frames": 828, "task_complete": False}))
    m = evaluate.evaluate_episodes([run, tmp_path / "missing"])
    assert m["episodes"] == 2 and m["full_plan_rate"] == 0.5 and m["first_attempt_grasp_success"] == 0.6667
    assert m["placement_success"] == 0.6667 and m["recovery_rate"] is None

"""Measure perception on a labelled split, grasping in sim episodes, or real-cell trials.

    python -m smartgrasp.evaluate perception data/bev/sim_v1 --detector yolo:artifacts/models/sim_v1/weights/best.pt
    python -m smartgrasp.evaluate perception data/bev/sim_v1 --detector hsv
    python -m smartgrasp.evaluate episodes runs/episodes/*
    python -m smartgrasp.evaluate trials docs/trials_template.csv

Ground truth is only ever read here, after the detector has answered.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .dataset import read_labels
from .geometry import Bev, fold90, pose, twin

MATCH_MM = 10.0  # a prediction farther than one cube edge from every true cube is a false positive


def graspable(xy_error_mm, yaw_error_deg):
    """Would the gripper still close cleanly on a cube perceived with these errors?

    The twin's SMC MHZL2-16D2 opens 12 mm around a 10 mm cube (CSI measured 15 +/- 2 mm
    at the fingertips), and the twin's executor rejects placements 8 degrees or more off.
    """
    return xy_error_mm <= 2.0 and yaw_error_deg <= 8.0


def _stats(values):
    v = np.asarray(values, float)
    if not v.size:
        return None
    return {"p50": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2), "max": round(float(v.max()), 2)}


def evaluate_perception(dataset, detector, split="test", limit=None):
    dataset, bev = Path(dataset), Bev.load(dataset)
    matches, missed, false_pos, seconds, images = [], 0, 0, [], 0
    for image_path in sorted((dataset / "images" / split).glob("*.jpg"))[:limit]:
        label = dataset / "labels" / split / f"{image_path.stem}.txt"
        if not label.exists():  # real frames not labelled yet
            continue
        truth = [(c, pose(px, bev)) for c, px in read_labels(label, bev.size) if c.startswith("cube_")]
        image = cv2.imread(str(image_path))
        start = time.perf_counter()
        found = detector(image)
        seconds.append(time.perf_counter() - start)
        preds = [(d.cls, pose(d.corners, bev)) for d in found if d.cls.startswith("cube_")]
        images += 1
        mm = np.array([[1000 * np.linalg.norm(t[1][0] - p[1][0]) for p in preds] for t in truth]).reshape(len(truth), len(preds))
        pairs = [(i, j) for i, j in zip(*linear_sum_assignment(mm)) if mm[i, j] <= MATCH_MM] if mm.size else []
        for i, j in pairs:
            yaw_error = abs(fold90(preds[j][1][1] - truth[i][1][1]))
            matches.append((mm[i, j], yaw_error, truth[i][0] == preds[j][0]))
        missed += len(truth) - len(pairs)
        false_pos += len(preds) - len(pairs)

    cubes, predictions = len(matches) + missed, len(matches) + false_pos
    right = [(xy, yaw) for xy, yaw, colour_ok in matches if colour_ok]
    return {
        "split": split,
        "images": images,
        "cubes": cubes,
        "predictions": predictions,
        "recall": round(len(right) / max(cubes, 1), 4),  # found, right colour, within MATCH_MM
        "precision": round(len(right) / max(predictions, 1), 4),
        "colour_accuracy": round(len(right) / max(len(matches), 1), 4),
        "xy_error_mm": _stats([xy for xy, _ in right]),
        "yaw_error_deg": _stats([yaw for _, yaw in right]),
        "graspable_rate": round(sum(graspable(xy, yaw) for xy, yaw in right) / max(cubes, 1), 4),
        "ms_per_image": round(1000 * float(np.mean(seconds)), 1) if seconds else None,
    }


def evaluate_episodes(run_dirs):
    """Closed-loop sim runs made by isaac/run_episode.sh (sort_plan.json + execution_report.json).

    No cycle time: the twin's scripted motion takes the same 276 frames for every pick.
    """
    episodes, attempts = 0, []
    planned_full, complete = 0, 0
    for run in map(Path, run_dirs):
        episodes += 1
        plan = run / "sort_plan.json"
        picks = json.loads(plan.read_text())["picks"] if plan.exists() else []
        planned_full += len(picks) == len(twin.CUBE_STACK_COLOURS)
        report = run / "execution_report.json"
        if report.exists():  # missing: the plan was rejected before execution
            r = json.loads(report.read_text())
            attempts += r["picks"]
            complete += bool(r["task_complete"])
    n = max(len(attempts), 1)
    return {
        "episodes": episodes,
        "full_plan_rate": round(planned_full / max(episodes, 1), 4),
        "task_complete_rate": round(complete / max(episodes, 1), 4),
        "attempted_picks": len(attempts),
        "first_attempt_grasp_success": round(sum(p["pickup_verified"] for p in attempts) / n, 4),
        "placement_success": round(sum(p["placement_verified"] for p in attempts) / n, 4),
        "recovery_rate": None,  # the baseline never retries a failed grasp (ROADMAP task 10)
    }


def evaluate_trials(path):
    """Real-cell log, one row per attempt: trial, attempt, grasp_ok, place_ok, cycle_s, notes."""
    trials = defaultdict(list)
    for row in csv.DictReader(open(path)):
        trials[row["trial"]].append(row)
    attempts = [sorted(rows, key=lambda r: int(r["attempt"])) for rows in trials.values()]
    ok = lambda r: r["grasp_ok"].strip() == "1"  # noqa: E731
    failed_first = [a for a in attempts if not ok(a[0])]
    cycles = [float(r["cycle_s"]) for a in attempts for r in a if ok(r) and r["place_ok"].strip() == "1"]
    return {
        "trials": len(attempts),
        "first_attempt_grasp_success": round(sum(ok(a[0]) for a in attempts) / max(len(attempts), 1), 4),
        "recovery_rate": round(sum(any(map(ok, a[1:])) for a in failed_first) / len(failed_first), 4) if failed_first else None,
        "cycle_s_per_pick": round(float(np.mean(cycles)), 2) if cycles else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("perception")
    p.add_argument("dataset")
    p.add_argument("--detector", required=True, help="yolo:<weights>, hsv or vlm[:<dataset>]")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="evaluate only the first N images (VLM rate limits)")
    p.add_argument("--save", help="write the metrics JSON here as well")
    e = sub.add_parser("episodes")
    e.add_argument("runs", nargs="+")
    t = sub.add_parser("trials")
    t.add_argument("csv")
    a = ap.parse_args()

    if a.mode == "perception":
        from .detectors import load_detector

        result = {"detector": a.detector, **evaluate_perception(a.dataset, load_detector(a.detector, Bev.load(a.dataset)), a.split, a.limit)}
    elif a.mode == "episodes":
        result = evaluate_episodes(a.runs)
    else:
        result = evaluate_trials(a.csv)
    text = json.dumps(result, indent=2)
    print(text)
    if getattr(a, "save", None):
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(text + "\n")


if __name__ == "__main__":
    main()

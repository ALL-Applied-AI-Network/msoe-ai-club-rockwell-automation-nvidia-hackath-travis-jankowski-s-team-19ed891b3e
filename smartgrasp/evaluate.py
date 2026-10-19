r"""Measure perception on a labelled split, grasping in sim episodes, or real-cell trials.

    python -m smartgrasp.evaluate perception data/bev/sim_v1 \
        --detector yolo:artifacts/models/sim_v1/weights/best.pt
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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .dataset import Split, read_labels
from .deploy import Pick
from .detectors import Detector, load_detector
from .geometry import Bev, FloatArray, fold90, pose, read_image, twin

MATCH_MM: float = 1000 * twin.CUBE_EDGE  # a prediction farther than this from every true cube is a false positive

type _CubePose = tuple[str, tuple[FloatArray, float]]  # (class name, (centre XY in m, yaw in degrees))
type _Match = tuple[float, float, bool]  # (centre error in mm, yaw error in degrees, colour correct)
type _TrialRow = dict[str, str]  # one row of the real-cell trial log, keyed by column name


class ErrorStats(TypedDict):
    """Spread of one error over the correctly matched cubes, each value rounded to 2 decimals.

    Attributes:
        p50: Median.
        p95: 95th percentile.
        max: Largest value.
    """

    p50: float
    p95: float
    max: float


class PerceptionMetrics(TypedDict):
    """Detector scores on one labelled split (cubes only).

    Attributes:
        split: Dataset split that was scored ("train", "val" or "test").
        images: Labelled images scored.
        cubes: True cubes in those images.
        predictions: Cubes the detector reported.
        recall: Share of true cubes found with the right colour within MATCH_MM.
        precision: Share of predictions that are such a correct find.
        colour_accuracy: Share of position matches that also have the right colour.
        xy_error_mm: Centre error of the correct finds in mm, or None if there are none.
        yaw_error_deg: Yaw error of the correct finds in degrees, or None if there are none.
        graspable_rate: Share of true cubes found well enough for the gripper to close on.
        ms_per_image: Mean detector time per image in milliseconds, or None with no images.
    """

    split: Split
    images: int
    cubes: int
    predictions: int
    recall: float
    precision: float
    colour_accuracy: float
    xy_error_mm: ErrorStats | None
    yaw_error_deg: ErrorStats | None
    graspable_rate: float
    ms_per_image: float | None


class EpisodeMetrics(TypedDict):
    """Grasping scores over closed-loop sim episodes.

    Attributes:
        episodes: Run directories scored.
        full_plan_rate: Share of episodes whose plan picked every cube of the stack.
        task_complete_rate: Share of episodes the twin's executor reported complete.
        attempted_picks: Picks the executor attempted across all episodes.
        first_attempt_grasp_success: Share of attempted picks that lifted the cube.
        placement_success: Share of attempted picks that placed the cube correctly.
        recovery_rate: Always None: the baseline never retries a failed grasp (issue #57).
    """

    episodes: int
    full_plan_rate: float
    task_complete_rate: float
    attempted_picks: int
    first_attempt_grasp_success: float
    placement_success: float
    recovery_rate: float | None


class TrialMetrics(TypedDict):
    """Grasping scores from a real-cell trial log.

    Attributes:
        trials: Trials in the log.
        first_attempt_grasp_success: Share of trials whose first attempt grasped the cube.
        recovery_rate: Share of failed first attempts that a later attempt grasped,
            or None if no first attempt failed.
        cycle_s_per_pick: Mean cycle time in seconds of attempts that grasped and placed,
            or None if none did.
    """

    trials: int
    first_attempt_grasp_success: float
    recovery_rate: float | None
    cycle_s_per_pick: float | None


class _PickEvent(TypedDict):
    """One attempted pick in the twin's execution_report.json (only the fields read here).

    Attributes:
        pickup_verified: The twin saw the cube lifted by the gripper.
        placement_verified: The twin saw the cube placed on its stack site.
    """

    pickup_verified: bool
    placement_verified: bool


class _ExecutionReport(TypedDict):
    """The twin's execution_report.json (only the fields read here).

    Attributes:
        picks: Every attempted pick, in order.
        task_complete: Whether the twin finished the whole tower.
    """

    picks: list[_PickEvent]
    task_complete: bool


def graspable(xy_error_mm: float, yaw_error_deg: float) -> bool:
    """Tell whether the gripper would still close cleanly on a cube perceived with these errors.

    The twin's SMC MHZL2-16D2 opens 12 mm around a 10 mm cube (CSI measured 15 +/- 2 mm
    at the fingertips), and the twin's executor rejects placements 8 degrees or more off.

    Args:
        xy_error_mm: Distance between the perceived and true cube centres in mm.
        yaw_error_deg: Yaw error in degrees, folded to [0, 45].

    Returns:
        True if both errors are within what the gripper and executor tolerate.
    """
    return xy_error_mm <= 2.0 and yaw_error_deg <= 8.0


def _stats(values: Sequence[float]) -> ErrorStats | None:
    """Summarise a list of errors by median, 95th percentile and maximum.

    Args:
        values: Errors, all in one unit.

    Returns:
        The rounded statistics, or None if there are no values.
    """
    array = np.asarray(values, float)
    if not array.size:
        return None
    return {
        "p50": round(float(np.median(array)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "max": round(float(array.max()), 2),
    }


def _match_cubes(truth: list[_CubePose], preds: list[_CubePose]) -> list[_Match]:
    """Pair true and predicted cubes in one image one-to-one by centre distance.

    The Hungarian assignment minimises the total centre distance; pairs farther apart than
    MATCH_MM are dropped, leaving those cubes unmatched.

    Args:
        truth: Labelled cubes as (class name, (centre XY in metres, yaw in degrees)).
        preds: Detected cubes in the same layout.

    Returns:
        One (centre error in mm, yaw error in degrees, colour correct) per matched pair.
    """
    true_xy = np.reshape([xy for _, (xy, _) in truth], (-1, 2))  # (0, 2) when empty, as cdist needs
    pred_xy = np.reshape([xy for _, (xy, _) in preds], (-1, 2))
    distance_mm = 1000 * cdist(true_xy, pred_xy)
    pairs = (
        [(i, j) for i, j in zip(*linear_sum_assignment(distance_mm)) if distance_mm[i, j] <= MATCH_MM]
        if distance_mm.size
        else []
    )
    matches: list[_Match] = []
    for i, j in pairs:
        yaw_error = abs(fold90(preds[j][1][1] - truth[i][1][1]))
        matches.append((distance_mm[i, j], yaw_error, truth[i][0] == preds[j][0]))
    return matches


def evaluate_perception(
    dataset: str | Path, detector: Detector, split: Split = "test", limit: int | None = None
) -> PerceptionMetrics:
    """Score a detector's cubes against the labels of one dataset split.

    Images without a label file (real frames not labelled yet) are skipped.

    Args:
        dataset: BEV dataset directory with images/<split>/*.jpg and labels/<split>/*.txt.
        detector: Detector to score; it never sees the labels.
        split: Dataset split to score.
        limit: Score only the first N images in name order (VLM rate limits); None for all.

    Returns:
        The perception metrics for the split.

    Raises:
        FileNotFoundError: If an image cannot be read.
    """
    dataset_dir, bev = Path(dataset), Bev.load(dataset)
    matches: list[_Match] = []
    seconds: list[float] = []
    cubes, predictions = 0, 0
    for image_path in sorted((dataset_dir / "images" / split).glob("*.jpg"))[:limit]:
        label = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
        if not label.exists():
            continue
        truth = [(cls, pose(corners, bev)) for cls, corners in read_labels(label, bev.size) if cls.startswith("cube_")]
        image = read_image(image_path)
        start = time.perf_counter()
        found = detector(image)
        seconds.append(time.perf_counter() - start)
        preds = [(d.cls, pose(d.corners, bev)) for d in found if d.cls.startswith("cube_")]
        cubes += len(truth)
        predictions += len(preds)
        matches += _match_cubes(truth, preds)

    correct = [(xy, yaw) for xy, yaw, colour_ok in matches if colour_ok]
    return {
        "split": split,
        "images": len(seconds),
        "cubes": cubes,
        "predictions": predictions,
        "recall": round(len(correct) / max(cubes, 1), 4),
        "precision": round(len(correct) / max(predictions, 1), 4),
        "colour_accuracy": round(len(correct) / max(len(matches), 1), 4),
        "xy_error_mm": _stats([xy for xy, _ in correct]),
        "yaw_error_deg": _stats([yaw for _, yaw in correct]),
        "graspable_rate": round(sum(graspable(xy, yaw) for xy, yaw in correct) / max(cubes, 1), 4),
        "ms_per_image": round(1000 * float(np.mean(seconds)), 1) if seconds else None,
    }


def evaluate_episodes(run_dirs: Iterable[str | Path]) -> EpisodeMetrics:
    """Score closed-loop sim runs made by isaac/run_episode.sh.

    Each run directory holds sort_plan.json and, unless the twin rejected the plan before
    executing it, execution_report.json. There is no cycle time: the twin's scripted motion
    takes the same 276 frames for every pick.

    Args:
        run_dirs: Episode run directories; missing files count as an empty plan or no execution.

    Returns:
        The episode metrics over all runs.
    """
    episodes, full_plans, complete = 0, 0, 0
    attempts: list[_PickEvent] = []
    for run in map(Path, run_dirs):
        episodes += 1
        plan_path = run / "sort_plan.json"
        picks: list[Pick] = json.loads(plan_path.read_text())["picks"] if plan_path.exists() else []
        full_plans += len(picks) == len(twin.CUBE_STACK_COLOURS)
        report_path = run / "execution_report.json"
        if report_path.exists():
            report: _ExecutionReport = json.loads(report_path.read_text())
            attempts += report["picks"]
            complete += bool(report["task_complete"])
    divisor = max(len(attempts), 1)
    return {
        "episodes": episodes,
        "full_plan_rate": round(full_plans / max(episodes, 1), 4),
        "task_complete_rate": round(complete / max(episodes, 1), 4),
        "attempted_picks": len(attempts),
        "first_attempt_grasp_success": round(sum(pick["pickup_verified"] for pick in attempts) / divisor, 4),
        "placement_success": round(sum(pick["placement_verified"] for pick in attempts) / divisor, 4),
        "recovery_rate": None,
    }


def _grasped(row: _TrialRow) -> bool:
    """Tell whether the gripper held the cube in one logged attempt.

    Args:
        row: One attempt from the trial log.

    Returns:
        True if its grasp_ok column is "1".
    """
    return row["grasp_ok"].strip() == "1"


def evaluate_trials(path: str | Path) -> TrialMetrics:
    """Score a real-cell trial log.

    The CSV has one row per attempt with the columns trial, attempt, grasp_ok, place_ok,
    cycle_s and notes; grasp_ok and place_ok are "1" for success.

    Args:
        path: Trial log CSV.

    Returns:
        The trial metrics.
    """
    rows_by_trial: defaultdict[str, list[_TrialRow]] = defaultdict(list)
    with open(path) as log:
        for row in csv.DictReader(log):
            rows_by_trial[row["trial"]].append(row)
    trials = [sorted(rows, key=lambda row: int(row["attempt"])) for rows in rows_by_trial.values()]
    failed_first = [attempts for attempts in trials if not _grasped(attempts[0])]
    recovered = sum(any(map(_grasped, attempts[1:])) for attempts in failed_first)
    cycles = [
        float(row["cycle_s"])
        for attempts in trials
        for row in attempts
        if _grasped(row) and row["place_ok"].strip() == "1"
    ]
    return {
        "trials": len(trials),
        "first_attempt_grasp_success": round(
            sum(_grasped(attempts[0]) for attempts in trials) / max(len(trials), 1), 4
        ),
        "recovery_rate": round(recovered / len(failed_first), 4) if failed_first else None,
        "cycle_s_per_pick": round(float(np.mean(cycles)), 2) if cycles else None,
    }


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser, with one sub-command per evaluation.

    Returns:
        Parser whose "mode" is perception, episodes or trials.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    perception = modes.add_parser("perception")
    perception.add_argument("dataset")
    perception.add_argument("--detector", required=True, help="yolo:<weights>, hsv or vlm[:<dataset>]")
    perception.add_argument("--split", default="test")
    perception.add_argument("--limit", type=int, help="evaluate only the first N images (VLM rate limits)")
    episodes = modes.add_parser("episodes")
    episodes.add_argument("runs", nargs="+")
    trials = modes.add_parser("trials")
    trials.add_argument("csv")
    for mode in (perception, episodes, trials):
        mode.add_argument("--save", help="write the metrics JSON here as well")
    return parser


def main() -> None:
    """Run one evaluation from the command line and print its metrics as JSON."""
    args = _parser().parse_args()
    result: Mapping[str, object]
    if args.mode == "perception":
        detector = load_detector(args.detector, Bev.load(args.dataset))
        result = {"detector": args.detector, **evaluate_perception(args.dataset, detector, args.split, args.limit)}
    elif args.mode == "episodes":
        result = evaluate_episodes(args.runs)
    else:
        result = evaluate_trials(args.csv)
    text = json.dumps(result, indent=2)
    print(text)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(text + "\n")


if __name__ == "__main__":
    main()

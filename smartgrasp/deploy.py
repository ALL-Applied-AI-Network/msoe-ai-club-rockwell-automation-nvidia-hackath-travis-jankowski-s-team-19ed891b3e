"""Camera image -> cube detections -> pick-and-place plan, for the twin and the real cell.

    python -m smartgrasp.deploy --image runs/ep1/world_camera.png --calib runs/ep1/camera.json \\
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
from pathlib import Path

import cv2
import numpy as np

from .detectors import load_detector
from .geometry import Bev, load_homography, pose, twin

MIN_CONF = 0.8  # the twin's executor refuses grasps below this pose confidence
REACH = ((0.18, 0.52), (-0.28, 0.30))  # x, y box in front of the arm, robot-base metres
REACH_RADIUS_M = 0.54  # the twin's IK fails top-down hovers from 0.545 m out
SITE_CLEAR_M = 0.045  # a 6-cube tower's top shows up to 35 mm off-site in the BEV (parallax)
APPROACH_M = 0.13  # hover height above pick and place, as in sort_scene.py
UF0_Z_M = 0.330  # controller UF0 sits 330 mm above the robot base plane (fanuc_er4ia_sim/docs/CALIBRATION_V05.md)
SAFE_Z = (twin.SURFACE_DZ + 0.002, 0.25)  # robot-base z limits for every target, metres
BGR = {"cube_yellow": (0, 220, 255), "cube_blue": (255, 120, 0), "cube_red": (0, 0, 255), "cap": (200, 200, 200), "tube": (160, 160, 160)}


def reachable(x, y):
    return REACH[0][0] <= x <= REACH[0][1] and REACH[1][0] <= y <= REACH[1][1] and np.hypot(x, y) <= REACH_RADIUS_M


def plan_stack(detections, bev):
    """Stack confident, reachable cubes in the twin's colour order (yellow, yellow, blue, blue, red, red).

    Missing colours are skipped, so two yellow cubes make a two-cube stack for a cell dry run;
    the twin's validate_plan accepts only the full six. Plans nothing unless the stack site is
    clear: one top-down frame cannot tell how tall a tower already is.
    """
    free = []
    for i, d in enumerate(detections):
        (x, y), yaw = pose(d.corners, bev)
        if np.hypot(x - twin.CUBE_STACK_XY[0], y - twin.CUBE_STACK_XY[1]) < SITE_CLEAR_M:
            return []
        if d.cls.startswith("cube_") and d.conf >= MIN_CONF and reachable(x, y):
            free.append(dict(detection_id=i, colour=d.cls[5:], xy=[float(x), float(y)], yaw_deg=float(yaw), conf=float(d.conf)))
    picks = []
    for colour in twin.CUBE_STACK_COLOURS:
        options = [c for c in free if c["colour"] == colour]
        if not options:
            continue
        c = max(options, key=lambda c: c["conf"])
        free.remove(c)
        level, t = len(picks), np.radians(c["yaw_deg"])
        picks.append(dict(
            detection_id=c["detection_id"],
            category="cube",
            label=f"{colour} cube",
            colour=colour,
            xy=c["xy"],
            yaw_deg=c["yaw_deg"],
            pose_confidence=c["conf"],
            spread_dir=[float(-np.sin(t)), float(np.cos(t)), 0.0],  # fingers close across the cube's faces
            pick_z=twin.SURFACE_DZ + twin.CUBE_TIP_CLEARANCE,
            stack_level=level,
            place_xy=list(twin.CUBE_STACK_XY),
            place_z=twin.SURFACE_DZ + level * twin.CUBE_EDGE + twin.CUBE_TIP_CLEARANCE,
            expected_centre_z=twin.SURFACE_DZ + (level + 0.5) * twin.CUBE_EDGE,
            place_spread_dir=[0.0, 1.0, 0.0],
            place_yaw_deg=0.0,
        ))
    return picks


CELL_ASSUMPTIONS = (
    "Targets are UF0 (robot base + 330 mm) with UTOOL1 = (0, 0, 223, 0, 0, 0) and the fingers "
    "closing along tool Y, so a top-down grasp is W=180, P=0, R=cube yaw. Confirm at CSI before any motion."
)


def cell_rows(picks):
    """Robot-base waypoints -> FANUC XYZWPR rows (mm, deg); see CELL_ASSUMPTIONS."""
    rows = []
    for p in picks:
        pick, place = (*p["xy"], p["pick_z"]), (*p["place_xy"], p["place_z"])
        up = lambda xyz: (xyz[0], xyz[1], xyz[2] + APPROACH_M)  # noqa: E731
        for step, (x, y, z), r, gripper in (
            ("approach", up(pick), p["yaw_deg"], "open"),
            ("descend", pick, p["yaw_deg"], ""),
            ("grip", pick, p["yaw_deg"], "close"),
            ("lift", up(pick), p["yaw_deg"], ""),
            ("carry", up(place), p["place_yaw_deg"], ""),
            ("lower", place, p["place_yaw_deg"], ""),
            ("release", place, p["place_yaw_deg"], "open"),
            ("retreat", up(place), p["place_yaw_deg"], ""),
        ):
            ok = reachable(x, y) and SAFE_Z[0] <= z <= SAFE_Z[1]
            rows.append(dict(
                cube=p["detection_id"], colour=p["colour"], step=step,
                X_mm=round(1000 * x, 1), Y_mm=round(1000 * y, 1), Z_mm=round(1000 * (z - UF0_Z_M), 1),
                W_deg=180.0, P_deg=0.0, R_deg=round(r, 1), gripper=gripper, within_limits=bool(ok),
            ))
    return rows


def draw(bev_img, detections, picks, bev):
    out = bev_img.copy()
    for d in detections:
        cv2.polylines(out, [np.int32(np.round(d.corners))], True, BGR.get(d.cls, (255, 255, 255)), 1, cv2.LINE_AA)
    for p in picks:
        u, v = bev.to_px(p["xy"])
        cv2.putText(out, str(p["stack_level"] + 1), (int(u) + 12, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    u, v = bev.to_px(twin.CUBE_STACK_XY)
    cv2.drawMarker(out, (int(u), int(v)), (0, 160, 0), cv2.MARKER_CROSS, 24, 2)
    return out


def deploy(image, calib, detector, out, bev=Bev()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bev_img = bev.warp(cv2.imread(str(image)), load_homography(calib, bev))
    detections = load_detector(detector, bev)(bev_img)
    picks = plan_stack(detections, bev)
    plan = {
        "source": "SmartGrasp",
        "detector": detector,
        "gripper_model": twin.GRIPPER_MODEL,
        "task": "cube_stack",
        "stack_colours_bottom_to_top": list(twin.CUBE_STACK_COLOURS),
        "open_gap_m": 2 * twin.JAW_OPEN,
        "picks": picks,
    }
    rows = cell_rows(picks)
    cv2.imwrite(str(out / "bev.png"), bev_img)
    cv2.imwrite(str(out / "overlay.png"), draw(bev_img, detections, picks, bev))
    (out / "detections.json").write_text(json.dumps([d.to_json(bev) for d in detections], indent=2))
    (out / "sort_plan.json").write_text(json.dumps(plan, indent=2))
    with open(out / "cell_plan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    cell = {"assumptions": CELL_ASSUMPTIONS, "all_within_limits": all(r["within_limits"] for r in rows), "rows": rows}
    (out / "cell_plan.json").write_text(json.dumps(cell, indent=2))
    return detections, picks, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="table camera frame")
    ap.add_argument("--calib", required=True, help="sim camera.json or real calib.json")
    ap.add_argument("--detector", required=True, help="yolo:<weights>, hsv or vlm[:<dataset>]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    detections, picks, rows = deploy(a.image, a.calib, a.detector, a.out)
    print(f"[deploy] {len(detections)} detections, {len(picks)} picks -> {a.out}")
    if not picks:
        print("  no plan: check overlay.png. Cubes must be confident and within reach, and the stack site clear.")
    print(f"  {CELL_ASSUMPTIONS}")
    for r in rows:
        flag = "" if r["within_limits"] else "  <-- OUTSIDE LIMITS"
        print(f"  cube {r['cube']} {r['step']:>8}  X {r['X_mm']:7.1f}  Y {r['Y_mm']:7.1f}  Z {r['Z_mm']:7.1f}  R {r['R_deg']:6.1f}  {r['gripper']}{flag}")


if __name__ == "__main__":
    main()

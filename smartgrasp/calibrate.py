"""Calibrate a real camera to the robot: fit its image -> cube-top-plane homography.

    python -m smartgrasp.calibrate --image frame.png --points points.csv --out calib/table_cam.json
    python -m smartgrasp.calibrate --image frame.png --points markers.csv --aruco --out calib/table_cam.json

points.csv columns: u,v,x,y   (pixel; robot-base metres)
with --aruco:       id,x,y    (ArUco DICT_4X4_50 marker id; robot-base metres of its centre)

Every point must sit at cube-top height (10 mm above the mat), e.g. markers glued
to 10 mm blocks whose centres were touched off with the robot. See docs/CELL_TEST.md.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import apply_homography, fit_homography


def aruco_centres(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    return {} if ids is None else {int(i): c.reshape(4, 2).mean(0) for c, i in zip(corners, ids.ravel())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--aruco", action="store_true", help="find pixel positions from ArUco markers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    image = cv2.imread(a.image)
    rows = list(csv.DictReader(open(a.points)))
    if a.aruco:
        found = aruco_centres(image)
        missing = [r["id"] for r in rows if int(r["id"]) not in found]
        if missing:
            raise SystemExit(f"Markers not found in the image: {missing}")
        uv = [found[int(r["id"])] for r in rows]
    else:
        uv = [(float(r["u"]), float(r["v"])) for r in rows]
    xy = np.array([(float(r["x"]), float(r["y"])) for r in rows])

    H = fit_homography(uv, xy)
    error_mm = 1000 * np.linalg.norm(apply_homography(H, uv) - xy, axis=1)
    calib = {
        "H": H.tolist(),  # image pixels -> robot-base XY metres on the cube-top plane
        "image_size": list(image.shape[1::-1]),
        "points": len(rows),
        "rms_mm": round(float(np.sqrt(np.mean(error_mm**2))), 3),
        "max_mm": round(float(error_mm.max()), 3),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps({k: v for k, v in calib.items() if k != "H"}, indent=2))
    if len(rows) < 6:
        print("Use 6+ points spread over the work area: 4 points fit exactly and hide errors.")
    if calib["max_mm"] > 2.0:
        print("Residual above 2 mm: check point coordinates, lens distortion, or marker height.")


if __name__ == "__main__":
    main()

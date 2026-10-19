r"""Calibrate a real camera to the robot: fit its image -> cube-top-plane homography.

    python -m smartgrasp.calibrate --image frame.png --points points.csv --out calib/table_cam.json
    python -m smartgrasp.calibrate --image frame.png --points markers.csv --aruco \
        --out calib/table_cam.json

points.csv columns: u,v,x,y   (pixel; robot-base metres)
with --aruco:       id,x,y    (ArUco DICT_4X4_50 marker id; robot-base metres of its centre)

Every point must sit at cube-top height (10 mm above the mat), e.g. markers glued
to 10 mm blocks whose centres were touched off with the robot. See docs/CELL_TEST.md.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from .geometry import FloatArray, Image, apply_homography, fit_homography, read_image

type _PixelPoints = list[FloatArray] | list[tuple[float, float]]  # (u, v) per point: ArUco or CSV


class Calibration(TypedDict):
    """A real camera's calib.json, as written by this module.

    Attributes:
        H: 3x3 homography from image pixels to robot-base XY metres on the cube-top plane.
        image_size: [width, height] of the calibration image in pixels.
        points: Number of calibration points.
        rms_mm: Root-mean-square distance between fitted and measured points, in mm.
        max_mm: Largest distance between a fitted and a measured point, in mm.
    """

    H: list[list[float]]
    image_size: list[int]
    points: int
    rms_mm: float
    max_mm: float


def aruco_centres(image: Image) -> dict[int, FloatArray]:
    """Find the ArUco DICT_4X4_50 markers in an image.

    Args:
        image: BGR camera image, shape (height, width, 3).

    Returns:
        Marker id -> marker centre in image pixels, shape (2,). Empty if no marker is found.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    if ids is None:
        return {}
    return {int(marker_id): marker.reshape(4, 2).mean(0) for marker, marker_id in zip(corners, ids.ravel())}


def _pixel_points(rows: list[dict[str, str]], image: Image, aruco: bool) -> _PixelPoints:
    """Get each calibration point's pixel position: from its u,v columns or its ArUco marker.

    Args:
        rows: Calibration points as read from the CSV.
        image: Calibration image, used to find the markers.
        aruco: Find each point's pixel position from the marker named in its id column.

    Returns:
        (u, v) in image pixels for each row, in row order.

    Raises:
        SystemExit: If a marker listed in the CSV is not found in the image.
    """
    if not aruco:
        return [(float(row["u"]), float(row["v"])) for row in rows]
    found = aruco_centres(image)
    missing = [row["id"] for row in rows if int(row["id"]) not in found]
    if missing:
        raise SystemExit(f"Markers not found in the image: {missing}")
    return [found[int(row["id"])] for row in rows]


def _fit_calibration(uv: _PixelPoints, xy: FloatArray, image: Image) -> Calibration:
    """Fit the homography and measure how far it maps each point from its measured position.

    Args:
        uv: Pixel position of each point.
        xy: Measured robot-base XY of each point in metres, shape (N, 2).
        image: Calibration image (only its size is recorded).

    Returns:
        The calib.json contents.
    """
    H = fit_homography(uv, xy)
    error_mm = 1000 * np.linalg.norm(apply_homography(H, uv) - xy, axis=1)
    return {
        "H": H.tolist(),
        "image_size": list(image.shape[1::-1]),
        "points": len(xy),
        "rms_mm": round(float(np.sqrt(np.mean(error_mm**2))), 3),
        "max_mm": round(float(error_mm.max()), 3),
    }


def _report(calib: Calibration) -> None:
    """Print the fit quality, with advice when there are too few points or the error is large.

    Args:
        calib: The calib.json contents.
    """
    print(json.dumps({k: v for k, v in calib.items() if k != "H"}, indent=2))
    if calib["points"] < 6:
        print("Use 6+ points spread over the work area: 4 points fit exactly and hide errors.")
    if calib["max_mm"] > 2.0:
        print("Residual above 2 mm: check point coordinates, lens distortion, or marker height.")


def main() -> None:
    """Calibrate from the command line: write calib.json and print the fit quality."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--aruco", action="store_true", help="find pixel positions from ArUco markers")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image = read_image(args.image)
    with open(args.points) as file:
        rows = list(csv.DictReader(file))
    uv = _pixel_points(rows, image, args.aruco)
    xy = np.array([(float(row["x"]), float(row["y"])) for row in rows])
    calib = _fit_calibration(uv, xy, image)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(calib, indent=2))
    _report(calib)


if __name__ == "__main__":
    main()

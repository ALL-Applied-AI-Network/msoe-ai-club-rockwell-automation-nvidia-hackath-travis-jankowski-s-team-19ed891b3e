"""Geometry shared by every stage: twin constants, the top-down (BEV) view and box labels.

Every model sees one kind of image: a bird's-eye view (BEV) of the plane at
cube-top height, 0.5 mm per pixel. In it a cube's top face is a true 20x20 px
square, the box angle is the world yaw, and a pixel maps to robot XY with one
multiply-add. Sim and real cameras differ only in the homography that makes it.

Frames: world = Isaac Sim world. Robot-base XY equals world XY (no rotation);
robot-base z = world z - twin.TABLE_TOP_Z.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TWIN_DIR = Path(__file__).resolve().parents[1] / "fanuc_er4ia_sim"
sys.path.insert(0, str(TWIN_DIR))
import scene_config as twin  # noqa: E402  cell dimensions: the single source of truth
from backproject import CameraModel  # noqa: E402  Isaac camera.json -> pinhole model

CLASSES = ["cube_yellow", "cube_blue", "cube_red", "cap", "tube"]
COLOURS = [c.removeprefix("cube_") for c in CLASSES if c.startswith("cube_")]


@dataclass(frozen=True)
class Bev:
    """Top-down view of the cube-top plane. Image right = +Y and down = +X,
    so it looks like the table camera's view with the perspective removed."""

    x_range: tuple = (0.16, 0.56)  # robot-base metres: cubes, caps, tubes and the tower site
    y_range: tuple = (-0.30, 0.34)
    mm_per_px: float = 0.5

    @classmethod
    def load(cls, dataset_dir):
        """The view a dataset was built with (its bev.json)."""
        j = json.loads((Path(dataset_dir) / "bev.json").read_text())
        return cls(tuple(j["x_range"]), tuple(j["y_range"]), j["mm_per_px"])

    @property
    def size(self):
        """(width, height) in pixels."""
        s = self.mm_per_px / 1000
        return (
            round((self.y_range[1] - self.y_range[0]) / s),
            round((self.x_range[1] - self.x_range[0]) / s),
        )

    @property
    def cube_px(self):
        """A cube's edge length in pixels."""
        return twin.CUBE_EDGE * 1000 / self.mm_per_px

    @property
    def world_to_px(self):
        """3x3 matrix taking homogeneous world XY to BEV pixels."""
        s = self.mm_per_px / 1000
        x0, y0 = self.x_range[0], self.y_range[0]
        return np.array([[0, 1 / s, -y0 / s], [1 / s, 0, -x0 / s], [0, 0, 1]])

    def to_px(self, xy):
        return apply_homography(self.world_to_px, xy)

    def to_world(self, uv):
        return apply_homography(np.linalg.inv(self.world_to_px), uv)

    def warp(self, image, H):
        """Camera image -> BEV image. H maps image pixels to world XY on the cube-top plane."""
        return cv2.warpPerspective(image, self.world_to_px @ H, self.size, flags=cv2.INTER_LINEAR)


def apply_homography(M, points):
    points = np.asarray(points, float)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), M).reshape(points.shape)


def fit_homography(image_px, world_xy):
    """Least-squares homography from >= 4 point pairs (pixels -> metres)."""
    H, _ = cv2.findHomography(np.asarray(image_px, float), np.asarray(world_xy, float))
    if H is None:
        raise ValueError("Homography fit failed: need >= 4 points, not all on one line")
    return H


def pixel(cam, point):
    """World point -> OpenCV pixel coordinates in the sim image.

    CameraModel.project is continuous: the renderer's pixel i spans [i, i + 1).
    OpenCV (warpPerspective, ArUco) puts pixel i's centre at i, so shift by half a pixel.
    """
    return cam.project(point) - 0.5


def load_homography(path, bev=Bev()):
    """Image pixels -> world XY on the cube-top plane.

    Accepts a sim camera.json (written by Isaac capture/generation) or a real
    calib.json (written by smartgrasp.calibrate, which stores "H" directly).
    """
    j = json.loads(Path(path).read_text())
    if "H" in j:
        return np.array(j["H"], float)
    cam, z = CameraModel(j), j["table_top_z"] + twin.CUBE_EDGE
    xy = [(x, y) for x in bev.x_range for y in bev.y_range]
    return fit_homography([pixel(cam, (x, y, z)) for x, y in xy], xy)


def fold90(degrees):
    """A square looks the same every 90 degrees: fold yaw to [-45, 45)."""
    return (degrees + 45) % 90 - 45


def pose(corners_px, bev):
    """BEV box corners -> (world XY centre, yaw in degrees folded to [-45, 45))."""
    w = bev.to_world(corners_px)
    edge = w[1] - w[0]
    return w.mean(0), fold90(np.degrees(np.arctan2(edge[1], edge[0])))


def outline(obj, surface_z):
    """World points on the top outline of a ground-truth object (as seen from above)."""
    x, y = obj["xy"]
    yaw = np.radians(obj.get("yaw_deg", 0.0))
    if obj["category"] == "cube":
        h, z = twin.CUBE_EDGE / 2, surface_z + twin.CUBE_EDGE
        local = [(-h, -h), (h, -h), (h, h), (-h, h)]
    elif obj["category"] == "cap":
        r, z = twin.CAP_RADIUS, surface_z + twin.CAP_HEIGHT
        local = [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    elif obj["category"] == "tube":  # lying on its side: top line is one diameter up
        h, r, z = twin.VIAL_LENGTH / 2, twin.VIAL_RADIUS, surface_z + 2 * twin.VIAL_RADIUS
        local = [(-h, -r), (h, -r), (h, r), (-h, r)]
    else:
        raise ValueError(f"Unknown category {obj['category']}")
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([(x + c * a - s * b, y + s * a + c * b, z) for a, b in local])


def label_corners(obj, cam_json, H, bev):
    """BEV-pixel corners of the box where the object's top appears in the BEV image.

    Cube tops lie on the BEV plane, so their box is the exact square. Caps and
    tubes are taller, so parallax shifts them; projecting through the camera
    and back puts their box where the model will actually see them.
    """
    cam = CameraModel(cam_json)
    image_px = [pixel(cam, p) for p in outline(obj, cam_json["table_top_z"])]
    px = apply_homography(bev.world_to_px @ H, image_px)
    return px if obj["category"] == "cube" else cv2.boxPoints(cv2.minAreaRect(px.astype(np.float32)))

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
from typing import Any, Literal, Self, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

TWIN_DIR: Path = Path(__file__).resolve().parents[1] / "fanuc_er4ia_sim"
sys.path.insert(0, str(TWIN_DIR))
import scene_config as twin  # noqa: E402  cell dimensions: the single source of truth
from backproject import CameraModel  # noqa: E402  Isaac camera.json -> pinhole model

type FloatArray = NDArray[np.floating[Any]]  # float32 or float64; each docstring gives the shape
type Image = NDArray[np.uint8]  # BGR, shape (height, width, 3)
type Corners = FloatArray  # (4, 2) box corners in pixels
type Homography = NDArray[np.float64]  # 3x3 projective transform
type CameraJson = dict[str, Any]  # a sim camera.json or a real calib.json, as read from disk
type Category = Literal["cube", "cap", "tube"]
type BgrColour = tuple[int, int, int]  # OpenCV colour order: blue, green, red


class SceneObject(TypedDict):
    """One ground-truth object on the table, as stored in each frame's NNNNNN.json.

    Attributes:
        category: What the object is.
        colour: Cube colour ("yellow", "blue" or "red"); caps are "yellow" and tubes "clear".
        xy: Centre in robot-base metres.
        yaw_deg: Rotation about +Z in degrees.
    """

    category: Category
    colour: str
    xy: list[float]
    yaw_deg: float


CLASSES: list[str] = ["cube_yellow", "cube_blue", "cube_red", "cap", "tube"]
COLOURS: list[str] = [c.removeprefix("cube_") for c in CLASSES if c.startswith("cube_")]


@dataclass(frozen=True)
class Bev:
    """Top-down view of the cube-top plane.

    Image right = +Y and down = +X, so it looks like the table camera's view with the
    perspective removed. The default ranges cover the cubes, caps, tubes and the tower site.

    Attributes:
        x_range: (min, max) robot-base X in metres, spanning the image top to bottom.
        y_range: (min, max) robot-base Y in metres, spanning the image left to right.
        mm_per_px: Size of one pixel in millimetres.
    """

    x_range: tuple[float, float] = (0.16, 0.56)
    y_range: tuple[float, float] = (-0.30, 0.34)
    mm_per_px: float = 0.5

    @classmethod
    def load(cls, dataset_dir: str | Path) -> Self:
        """Load the view a dataset was built with, from its bev.json.

        Args:
            dataset_dir: Dataset folder written by smartgrasp.dataset.

        Returns:
            The view stored in dataset_dir/bev.json.
        """
        saved: dict[str, Any] = json.loads((Path(dataset_dir) / "bev.json").read_text())
        return cls(tuple(saved["x_range"]), tuple(saved["y_range"]), saved["mm_per_px"])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the BEV image in pixels."""
        metres_per_px = self.mm_per_px / 1000
        return (
            round((self.y_range[1] - self.y_range[0]) / metres_per_px),
            round((self.x_range[1] - self.x_range[0]) / metres_per_px),
        )

    @property
    def cube_px(self) -> float:
        """A cube's edge length in pixels."""
        return twin.CUBE_EDGE * 1000 / self.mm_per_px

    @property
    def world_to_px(self) -> Homography:
        """3x3 matrix taking homogeneous robot-base XY (metres) to BEV pixels."""
        metres_per_px = self.mm_per_px / 1000
        x0, y0 = self.x_range[0], self.y_range[0]
        return np.array(
            [[0, 1 / metres_per_px, -y0 / metres_per_px], [1 / metres_per_px, 0, -x0 / metres_per_px], [0, 0, 1]]
        )

    def to_px(self, xy: ArrayLike) -> FloatArray:
        """Map robot-base XY to BEV pixels.

        Args:
            xy: One point (2,) or many points (..., 2) in robot-base metres.

        Returns:
            BEV pixel coordinates (u, v), same shape as xy.
        """
        return apply_homography(self.world_to_px, xy)

    def to_world(self, uv: ArrayLike) -> FloatArray:
        """Map BEV pixels to robot-base XY.

        Args:
            uv: One point (2,) or many points (..., 2) in BEV pixels.

        Returns:
            Robot-base XY in metres, same shape as uv.
        """
        return apply_homography(np.linalg.inv(self.world_to_px), uv)

    def warp(self, image: Image, H: Homography) -> Image:
        """Warp a camera image into the BEV image.

        Args:
            image: Camera image (BGR).
            H: Homography taking image pixels to robot-base XY (metres) on the cube-top plane.

        Returns:
            BEV image (BGR) of size `self.size`.
        """
        warped = cv2.warpPerspective(image, self.world_to_px @ H, self.size, flags=cv2.INTER_LINEAR)
        return cast(Image, warped)


def read_image(path: str | Path) -> Image:
    """Read a colour image from disk.

    Args:
        path: Image file (.jpg or .png).

    Returns:
        BGR image, shape (height, width, 3).

    Raises:
        FileNotFoundError: If OpenCV cannot read the file.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return cast(Image, image)


def apply_homography(M: Homography, points: ArrayLike) -> FloatArray:
    """Apply a projective transform to 2D points.

    Args:
        M: 3x3 homography.
        points: One point (2,) or many points (..., 2).

    Returns:
        Transformed points, same shape as points.
    """
    array = np.asarray(points, float)
    transformed = cast(FloatArray, cv2.perspectiveTransform(array.reshape(-1, 1, 2), M))
    return transformed.reshape(array.shape)


def fit_homography(image_px: ArrayLike, world_xy: ArrayLike) -> Homography:
    """Fit a least-squares homography from at least 4 point pairs.

    Args:
        image_px: (N, 2) image points in pixels.
        world_xy: (N, 2) matching robot-base XY in metres.

    Returns:
        3x3 homography taking image pixels to robot-base metres.

    Raises:
        ValueError: If fewer than 4 points are given or they all lie on one line.
    """
    H, _ = cv2.findHomography(np.asarray(image_px, float), np.asarray(world_xy, float))
    if H is None:
        raise ValueError("Homography fit failed: need >= 4 points, not all on one line")
    return cast(Homography, H)


def pixel(cam: CameraModel, point: ArrayLike) -> FloatArray:
    """Project a world point to OpenCV pixel coordinates in the sim image.

    CameraModel.project is continuous: the renderer's pixel i spans [i, i + 1).
    OpenCV (warpPerspective, ArUco) puts pixel i's centre at i, so shift by half a pixel.

    Args:
        cam: The sim camera.
        point: (x, y, z) world point in metres, in front of the camera.

    Returns:
        (u, v) pixel coordinates.
    """
    return cam.project(point) - 0.5


def load_homography(path: str | Path, bev: Bev = Bev()) -> Homography:
    """Fit the homography from image pixels to robot-base XY on the cube-top plane.

    Accepts a sim camera.json (written by Isaac capture/generation), fitted through the BEV
    corners, or a real calib.json (written by smartgrasp.calibrate), which stores "H" directly.

    Args:
        path: camera.json or calib.json.
        bev: View whose corners anchor the fit for a sim camera.

    Returns:
        3x3 homography taking image pixels to robot-base metres.

    Raises:
        ValueError: If the corner points of a sim camera cannot be fitted.
    """
    camera: CameraJson = json.loads(Path(path).read_text())
    if "H" in camera:
        return np.array(camera["H"], float)
    cam, cube_top_z = CameraModel(camera), camera["table_top_z"] + twin.CUBE_EDGE
    corners_xy = [(x, y) for x in bev.x_range for y in bev.y_range]
    return fit_homography([pixel(cam, (x, y, cube_top_z)) for x, y in corners_xy], corners_xy)


def fold90(degrees: float) -> float:
    """Fold a yaw to [-45, 45), since a square looks the same every 90 degrees.

    Args:
        degrees: Yaw in degrees.

    Returns:
        The equivalent yaw in [-45, 45) degrees.
    """
    return (degrees + 45) % 90 - 45


def pose(corners_px: Corners, bev: Bev) -> tuple[FloatArray, float]:
    """Turn BEV box corners into a world pose.

    Args:
        corners_px: (4, 2) box corners in BEV pixels.
        bev: The view the corners are in.

    Returns:
        (centre, yaw): centre (2,) in robot-base metres and yaw in degrees folded to [-45, 45).
    """
    corners_xy = bev.to_world(corners_px)
    edge = corners_xy[1] - corners_xy[0]
    return corners_xy.mean(0), fold90(np.degrees(np.arctan2(edge[1], edge[0])))


def outline(obj: SceneObject, surface_z: float) -> FloatArray:
    """List world points on the top outline of a ground-truth object, as seen from above.

    Args:
        obj: The object.
        surface_z: World z of the surface it rests on, in metres.

    Returns:
        (N, 3) world points in metres: 4 corners for a cube or tube, 16 rim points for a cap.

    Raises:
        ValueError: If the object's category is unknown.
    """
    x, y = obj["xy"]
    yaw = np.radians(obj.get("yaw_deg", 0.0))
    if obj["category"] == "cube":
        half, top_z = twin.CUBE_EDGE / 2, surface_z + twin.CUBE_EDGE
        local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    elif obj["category"] == "cap":
        radius, top_z = twin.CAP_RADIUS, surface_z + twin.CAP_HEIGHT
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        local = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    elif obj["category"] == "tube":  # lying on its side: top line is one diameter up
        half, radius = twin.VIAL_LENGTH / 2, twin.VIAL_RADIUS
        top_z = surface_z + 2 * twin.VIAL_RADIUS
        local = [(-half, -radius), (half, -radius), (half, radius), (-half, radius)]
    else:
        raise ValueError(f"Unknown category {obj['category']}")
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.array([(x + cos_yaw * u - sin_yaw * v, y + sin_yaw * u + cos_yaw * v, top_z) for u, v in local])


def label_corners(obj: SceneObject, cam_json: CameraJson, H: Homography, bev: Bev) -> Corners:
    """Find the BEV-pixel corners of the box where the object's top appears in the BEV image.

    Cube tops lie on the BEV plane, so their box is the exact square. Caps and
    tubes are taller, so parallax shifts them; projecting through the camera
    and back puts their box where the model will actually see them.

    Args:
        obj: Ground-truth object.
        cam_json: The sim camera.json the frame was rendered with.
        H: Homography taking image pixels to robot-base XY (metres) on the cube-top plane.
        bev: The view to label in.

    Returns:
        (4, 2) box corners in BEV pixels.
    """
    cam = CameraModel(cam_json)
    image_px = [pixel(cam, point) for point in outline(obj, cam_json["table_top_z"])]
    bev_px = apply_homography(bev.world_to_px @ H, image_px)
    if obj["category"] == "cube":
        return bev_px
    return cast(Corners, cv2.boxPoints(cv2.minAreaRect(bev_px.astype(np.float32))))

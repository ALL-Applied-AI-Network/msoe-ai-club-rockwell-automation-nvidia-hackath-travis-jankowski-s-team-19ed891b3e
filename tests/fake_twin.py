"""OpenCV stand-in for isaac/generate_data.py: same files, cartoon images, no GPU.

    uv run python tests/fake_twin.py data/raw/fake --frames 200

Use it to exercise the pipeline on a laptop. Its images are flat-shaded boxes
seen through the calibrated table camera, so a model trained on them proves the
plumbing works, not that it will see real cubes. Use Isaac renders for that.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from smartgrasp.geometry import BgrColour, CameraJson, CameraModel, FloatArray, Image, SceneObject, outline, pixel, twin
from smartgrasp.scene import sample_layout

# Fill colour per cube colour, and per category for caps and tubes.
BGR: dict[str, BgrColour] = {
    "yellow": (30, 200, 235),
    "blue": (190, 100, 30),
    "red": (40, 40, 200),
    "cap": (120, 210, 230),
    "tube": (225, 225, 215),
}
TABLE_BGR: BgrColour = (204, 200, 196)
# Six reachable cubes clear of the tower site: a complete yellow/blue/red stack task (--stack).
STACK: list[SceneObject] = [
    {"category": "cube", "colour": colour, "xy": xy, "yaw_deg": yaw}
    for colour, xy, yaw in [
        ("yellow", [0.30, -0.15], 10.0),
        ("yellow", [0.42, -0.05], -20.0),
        ("blue", [0.36, 0.20], 33.0),
        ("blue", [0.46, 0.15], 0.0),
        ("red", [0.25, -0.05], -40.0),
        ("red", [0.40, 0.25], 44.0),
    ]
]


def camera_json() -> CameraJson:
    """Build the twin's table camera in capture_scene.py's camera.json schema.

    The camera is made as in the twin's own host tests (test_support.camera_model), whose
    CameraModel keeps the camera.json dict it was built from in its `.j` attribute.

    Returns:
        The table camera's camera.json dict, plus "camera" and "robot_base_xy" keys.
    """
    from test_support import camera_model  # fanuc_er4ia_sim/test_support.py, on sys.path via geometry

    return {"camera": "table", **camera_model(twin.TABLE_CAM).j, "robot_base_xy": [0.0, 0.0]}


def _fill_pixels(img: Image, px: FloatArray, colour: BgrColour, shade: float) -> None:
    """Fill a convex polygon given in sub-pixel image coordinates, anti-aliased.

    OpenCV takes fixed-point vertices: shift=4 means 4 fractional bits, so pixels are scaled by 16.

    Args:
        img: Camera image to draw into, changed in place.
        px: Polygon vertices in image pixels, shape (N, 2) or (N, 1, 2).
        colour: Fill colour before shading.
        shade: Brightness factor applied to the colour (1.0 = unshaded).
    """
    cv2.fillConvexPoly(img, np.round(px * 16).astype(np.int32), [v * shade for v in colour], cv2.LINE_AA, shift=4)


def _fill(img: Image, cam: CameraModel, points: FloatArray, colour: BgrColour, shade: float) -> None:
    """Project a convex world polygon into the image and fill it.

    Args:
        img: Camera image to draw into, changed in place.
        cam: Table camera model.
        points: Polygon vertices in world metres, shape (N, 3).
        colour: Fill colour before shading.
        shade: Brightness factor applied to the colour (1.0 = unshaded).
    """
    _fill_pixels(img, np.array([pixel(cam, p) for p in points]), colour, shade)  # rasterise like Isaac


def _draw_object(img: Image, cam: CameraModel, obj: SceneObject, surface_z: float) -> None:
    """Draw one object: its camera-facing sides shaded darker, then its top face.

    A cap is an upright cylinder, so its side is the convex hull of both rims. Cubes and
    tubes are boxes, so only the side faces that point towards the camera are drawn
    (back-face culling).

    Args:
        img: Camera image to draw into, changed in place.
        cam: Table camera model.
        obj: Ground-truth object.
        surface_z: Table-top height in world metres.
    """
    top = outline(obj, surface_z)  # the same outline the labels are made from
    bottom = top.copy()
    bottom[:, 2] = surface_z
    colour = BGR[obj["colour"] if obj["category"] == "cube" else obj["category"]]
    if obj["category"] == "cap":
        rims = np.array([pixel(cam, p) for p in [*bottom, *top]], dtype=np.float32)
        hull = cast(FloatArray, cv2.convexHull(rims))  # float32 points in, float32 points out
        _fill_pixels(img, hull, colour, 0.7)
    else:
        for i in range(4):
            face = np.array([bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]])
            normal = np.cross(face[1] - face[0], face[3] - face[0])
            if normal @ (cam.eye - face.mean(0)) > 0:
                _fill(img, cam, face, colour, 0.6)
    _fill(img, cam, top, colour, 1.0)


def render(layout: list[SceneObject], cam_json: CameraJson, rng: np.random.Generator) -> Image:
    """Render one cartoon frame of a layout as the table camera sees it.

    Draws a noisy grey table, then every object from the farthest to the nearest
    (painter's algorithm), then applies a random brightness and white-balance gain and a
    slight blur.

    Args:
        layout: Ground-truth objects to draw.
        cam_json: Table camera (camera.json dict).
        rng: Random source for the sensor noise and the colour gain.

    Returns:
        BGR image at the camera's resolution.
    """
    cam = CameraModel(cam_json)
    width, height = cam_json["resolution"]
    noise = rng.normal(0, 4, (height, width, 3))
    img = np.clip(np.full((height, width, 3), TABLE_BGR) + noise, 0, 255).astype(np.uint8)
    surface_z = cam_json["table_top_z"]
    for obj in sorted(layout, key=lambda o: -np.linalg.norm(np.array([*o["xy"], surface_z]) - cam.eye)):
        _draw_object(img, cam, obj, surface_z)

    gain = rng.uniform(0.75, 1.25) * rng.uniform(0.95, 1.05, 3)  # brightness and white balance
    return cv2.GaussianBlur(np.clip(img * gain, 0, 255).astype(np.uint8), (0, 0), 0.6)


def write_frames(out: str | Path, layouts: Sequence[list[SceneObject]], seed: int = 0) -> None:
    """Write a raw dataset folder in isaac/generate_data.py's layout.

    Writes camera.json, manifest.json, and per frame an NNNNNN.jpg image with its
    NNNNNN.json ground truth.

    Args:
        out: Output folder, created if missing.
        layouts: Ground-truth objects for each frame.
        seed: Seed for the render noise, recorded in manifest.json.
    """
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    cam_json, rng = camera_json(), np.random.default_rng(seed)
    (folder / "camera.json").write_text(json.dumps(cam_json, indent=2))
    manifest = {"generator": "tests/fake_twin.py", "frames": len(layouts), "seed": seed}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for i, layout in enumerate(layouts):
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), render(layout, cam_json, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
        (folder / f"{i:06d}.json").write_text(json.dumps({"objects": layout}))


def main() -> None:
    """Parse the command line and write the frames."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack", action="store_true", help="every frame shows the fixed six-cube STACK layout")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    write_frames(args.out, [STACK if args.stack else sample_layout(rng) for _ in range(args.frames)], args.seed)


if __name__ == "__main__":
    main()

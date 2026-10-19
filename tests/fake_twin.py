"""OpenCV stand-in for isaac/generate_data.py: same files, cartoon images, no GPU.

    uv run python tests/fake_twin.py data/raw/fake --frames 200

Use it to exercise the pipeline on a laptop. Its images are flat-shaded boxes
seen through the calibrated table camera, so a model trained on them proves the
plumbing works, not that it will see real cubes. Use Isaac renders for that.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from smartgrasp.geometry import CameraModel, outline, pixel, twin
from smartgrasp.scene import sample_layout

BGR = {"yellow": (30, 200, 235), "blue": (190, 100, 30), "red": (40, 40, 200), "cap": (120, 210, 230), "tube": (225, 225, 215)}
# Six reachable cubes clear of the tower site: a complete yellow/blue/red stack task (--stack).
STACK = [
    {"category": "cube", "colour": c, "xy": xy, "yaw_deg": yaw}
    for c, xy, yaw in [
        ("yellow", [0.30, -0.15], 10.0), ("yellow", [0.42, -0.05], -20.0),
        ("blue", [0.36, 0.20], 33.0), ("blue", [0.46, 0.15], 0.0),
        ("red", [0.25, -0.05], -40.0), ("red", [0.40, 0.25], 44.0),
    ]
]


def camera_json():
    """The twin's table camera (built as in its own host tests) in capture_scene.py's camera.json schema."""
    from test_support import camera_model  # fanuc_er4ia_sim/test_support.py

    # .j is the camera.json dict the twin's CameraModel was built from
    return {"camera": "table", **camera_model(twin.TABLE_CAM).j, "robot_base_xy": [0.0, 0.0]}


def render(layout, cam_json, rng):
    cam = CameraModel(cam_json)
    w, h = cam_json["resolution"]
    img = np.clip(np.full((h, w, 3), (204, 200, 196)) + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    surface = cam_json["table_top_z"]

    def fill(points, colour, shade):
        px = np.array([pixel(cam, p) for p in points])  # rasterise like Isaac
        cv2.fillConvexPoly(img, np.round(px * 16).astype(np.int32), [v * shade for v in colour], cv2.LINE_AA, shift=4)

    # painter's algorithm: far objects first
    for obj in sorted(layout, key=lambda o: -np.linalg.norm(np.array([*o["xy"], surface]) - cam.eye)):
        top = outline(obj, surface)  # the same outline the labels are made from
        bottom = top.copy()
        bottom[:, 2] = surface
        colour = BGR[obj["colour"] if obj["category"] == "cube" else obj["category"]]
        if obj["category"] == "cap":  # an upright cylinder: its side is the hull of both rims
            hull = cv2.convexHull(np.float32([pixel(cam, p) for p in [*bottom, *top]]))
            cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32), [v * 0.7 for v in colour], cv2.LINE_AA, shift=4)
        else:  # a box: draw only the sides that face the camera (back-face culling)
            for i in range(4):
                face = np.array([bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]])
                normal = np.cross(face[1] - face[0], face[3] - face[0])
                if normal @ (cam.eye - face.mean(0)) > 0:
                    fill(face, colour, 0.6)
        fill(top, colour, 1.0)

    gain = rng.uniform(0.75, 1.25) * rng.uniform(0.95, 1.05, 3)  # brightness and white balance
    return cv2.GaussianBlur(np.clip(img * gain, 0, 255).astype(np.uint8), (0, 0), 0.6)


def write_frames(out, layouts, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cam_json, rng = camera_json(), np.random.default_rng(seed)
    (out / "camera.json").write_text(json.dumps(cam_json, indent=2))
    (out / "manifest.json").write_text(json.dumps({"generator": "tests/fake_twin.py", "frames": len(layouts), "seed": seed}))
    for i, layout in enumerate(layouts):
        cv2.imwrite(str(out / f"{i:06d}.jpg"), render(layout, cam_json, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / f"{i:06d}.json").write_text(json.dumps({"objects": layout}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack", action="store_true", help="every frame shows the fixed six-cube STACK layout")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    write_frames(a.out, [STACK if a.stack else sample_layout(rng) for _ in range(a.frames)], a.seed)


if __name__ == "__main__":
    main()

"""Turn rendered frames into a YOLO-OBB dataset of top-down (BEV) images.

    python -m smartgrasp.dataset data/raw/sim_v1 data/bev/sim_v1
    python -m smartgrasp.dataset data/raw/csi_day1 data/bev/csi_day1 --calib calib/table_cam.json --real

Input (written by isaac/generate_data.py or tests/fake_twin.py):
    camera.json, manifest.json, and per frame NNNNNN.jpg + NNNNNN.json ({"objects": [...]})
Output:
    images/{train,val,test}/*.jpg, labels/{train,val,test}/*.txt, data.yaml, bev.json, manifest.json

--real frames are only warped (no labels) and all go to the test split: real
data is for evaluation, never training. Label them in CVAT ("Ultralytics YOLO
Oriented Bounding Boxes" export) and drop the .txt files into labels/test/.
"""

import argparse
import json
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import CLASSES, Bev, label_corners, load_homography


def split_of(name, val=0.1, test=0.1):
    """Deterministic split from the frame name, so it is stable as the dataset grows."""
    r = zlib.crc32(name.encode()) / 2**32
    return "test" if r < test else "val" if r < test + val else "train"


def class_id(obj):
    return CLASSES.index(f"cube_{obj['colour']}" if obj["category"] == "cube" else obj["category"])


def read_labels(path, size):
    """YOLO-OBB label file -> [(class name, (4, 2) corners in pixels)] for an image of size (w, h)."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return [(CLASSES[int(r[0])], np.array(r[1:9], float).reshape(4, 2) * size) for r in rows]


def label_lines(objects, cam_json, H, bev):
    w, h = bev.size
    lines = []
    for obj in objects:
        px = label_corners(obj, cam_json, H, bev)
        cx, cy = px.mean(0)
        if 0 <= cx < w and 0 <= cy < h:  # centre inside the view
            lines.append(f"{class_id(obj)} " + " ".join(f"{v:.6f}" for v in (px / (w, h)).ravel()))
    return lines


def build(raw, out, calib=None, real=False, bev=Bev()):
    raw, out = Path(raw), Path(out)
    calib = Path(calib) if calib else raw / "camera.json"
    H, cam_json = load_homography(calib, bev), json.loads(calib.read_text())
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)

    frames, classes = Counter(), Counter()
    for image_path in sorted(p for p in raw.iterdir() if p.suffix in (".jpg", ".png")):
        split = "test" if real else split_of(image_path.stem)
        cv2.imwrite(str(out / "images" / split / f"{image_path.stem}.jpg"), bev.warp(cv2.imread(str(image_path)), H))
        frames[split] += 1
        if not real:
            objects = json.loads(image_path.with_suffix(".json").read_text())["objects"]
            lines = label_lines(objects, cam_json, H, bev)
            (out / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(lines))
            classes.update(CLASSES[int(line.split()[0])] for line in lines)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    # No "path:" key, so Ultralytics resolves the splits next to this file.
    (out / "data.yaml").write_text(f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    (out / "bev.json").write_text(json.dumps({**asdict(bev), "calib": str(calib)}, indent=2))
    source = raw / "manifest.json"
    manifest = {
        "source": str(raw),
        "real": real,
        "frames": dict(frames),
        "objects": dict(classes),
        "generator": json.loads(source.read_text()) if source.exists() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="directory of rendered or captured frames")
    ap.add_argument("out", help="dataset directory to create")
    ap.add_argument("--calib", help="camera.json or calib.json (default: RAW/camera.json)")
    ap.add_argument("--real", action="store_true", help="real camera frames: warp only, all to the test split")
    a = ap.parse_args()
    print(json.dumps(build(a.raw, a.out, a.calib, a.real), indent=2))


if __name__ == "__main__":
    main()

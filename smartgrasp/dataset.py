r"""Turn rendered frames into a YOLO-OBB dataset of top-down (BEV) images.

    python -m smartgrasp.dataset data/raw/sim_v1 data/bev/sim_v1
    python -m smartgrasp.dataset data/raw/csi_day1 data/bev/csi_day1 \
        --calib calib/table_cam.json --real

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
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np

from .geometry import (
    CLASSES,
    Bev,
    CameraJson,
    Corners,
    FloatArray,
    Homography,
    SceneObject,
    label_corners,
    load_homography,
    read_image,
)

type Split = Literal["train", "val", "test"]


class DatasetManifest(TypedDict):
    """Summary of a built dataset, written to its manifest.json.

    Attributes:
        source: Directory of raw frames the dataset was built from.
        real: Whether the frames came from a real camera (all in the test split, no labels).
        frames: Number of images in each split.
        objects: Number of labelled objects per class name (empty for real frames).
        generator: The raw directory's own manifest.json, or None if it has none.
    """

    source: str
    real: bool
    frames: dict[Split, int]
    objects: dict[str, int]
    generator: dict[str, object] | None


def split_of(name: str, val: float = 0.1, test: float = 0.1) -> Split:
    """Pick a frame's split from a hash of its name, so it is stable as the dataset grows.

    Args:
        name: Frame name, e.g. "000042".
        val: Fraction of frames that go to the validation split.
        test: Fraction of frames that go to the test split.

    Returns:
        "train", "val" or "test".
    """
    hash_fraction = zlib.crc32(name.encode()) / 2**32
    return "test" if hash_fraction < test else "val" if hash_fraction < test + val else "train"


def class_id(obj: SceneObject) -> int:
    """Look up an object's YOLO class index (cubes are split by colour).

    Args:
        obj: Ground-truth object.

    Returns:
        Index into CLASSES.
    """
    return CLASSES.index(f"cube_{obj['colour']}" if obj["category"] == "cube" else obj["category"])


def read_labels(path: str | Path, size: tuple[float, float] | FloatArray) -> list[tuple[str, Corners]]:
    """Read a YOLO-OBB label file.

    Args:
        path: Label .txt file: per line a class id and 8 corner coordinates normalised to 0..1.
        size: (width, height) of the labelled image in pixels.

    Returns:
        (class name, (4, 2) box corners in pixels) for each labelled object.
    """
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return [(CLASSES[int(fields[0])], np.array(fields[1:9], float).reshape(4, 2) * size) for fields in rows]


def label_lines(objects: Iterable[SceneObject], cam_json: CameraJson, H: Homography, bev: Bev) -> list[str]:
    """Make YOLO-OBB label lines for the objects whose box centre lies inside the view.

    Args:
        objects: Ground-truth objects in one frame.
        cam_json: The sim camera.json that rendered the frame.
        H: Homography from camera image pixels to robot-base XY metres on the cube-top plane.
        bev: View the labels are for.

    Returns:
        One line per object: class id, then the 4 box corners as x y pairs normalised by the
        BEV image size (6 decimals).
    """
    w, h = bev.size
    lines: list[str] = []
    for obj in objects:
        corners = label_corners(obj, cam_json, H, bev)
        cx, cy = corners.mean(0)
        if 0 <= cx < w and 0 <= cy < h:
            lines.append(f"{class_id(obj)} " + " ".join(f"{v:.6f}" for v in (corners / (w, h)).ravel()))
    return lines


def _write_data_yaml(out: Path) -> None:
    """Write the Ultralytics data.yaml for a dataset.

    It has no "path:" key, so Ultralytics resolves the image folders next to the file.

    Args:
        out: Dataset directory.
    """
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")


def build(
    raw: str | Path, out: str | Path, calib: str | Path | None = None, real: bool = False, bev: Bev = Bev()
) -> DatasetManifest:
    """Warp every frame in raw to the BEV view and write the dataset to out.

    A frame that OpenCV cannot read stops the build with FileNotFoundError.

    Args:
        raw: Directory of rendered or captured frames.
        out: Dataset directory to create.
        calib: camera.json or calib.json; defaults to raw/camera.json.
        real: Real camera frames: warp only (no labels) and put them all in the test split.
        bev: View to warp the frames to.

    Returns:
        The manifest, also written to out/manifest.json.
    """
    raw, out = Path(raw), Path(out)
    calib = Path(calib) if calib else raw / "camera.json"
    H = load_homography(calib, bev)
    cam_json: CameraJson = json.loads(calib.read_text())
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)

    frames: Counter[Split] = Counter()
    classes: Counter[str] = Counter()
    for image_path in sorted(p for p in raw.iterdir() if p.suffix in (".jpg", ".png")):
        split = "test" if real else split_of(image_path.stem)
        cv2.imwrite(str(out / "images" / split / f"{image_path.stem}.jpg"), bev.warp(read_image(image_path), H))
        frames[split] += 1
        if not real:
            objects: list[SceneObject] = json.loads(image_path.with_suffix(".json").read_text())["objects"]
            lines = label_lines(objects, cam_json, H, bev)
            (out / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(lines))
            classes.update(CLASSES[int(line.split()[0])] for line in lines)

    _write_data_yaml(out)
    (out / "bev.json").write_text(json.dumps({**asdict(bev), "calib": str(calib)}, indent=2))
    source = raw / "manifest.json"
    manifest: DatasetManifest = {
        "source": str(raw),
        "real": real,
        "frames": dict(frames),
        "objects": dict(classes),
        "generator": json.loads(source.read_text()) if source.exists() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    """Build a dataset from the command line and print its manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", help="directory of rendered or captured frames")
    parser.add_argument("out", help="dataset directory to create")
    parser.add_argument("--calib", help="camera.json or calib.json (default: RAW/camera.json)")
    parser.add_argument("--real", action="store_true", help="real camera frames: warp only, all to the test split")
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.out, args.calib, args.real), indent=2))


if __name__ == "__main__":
    main()

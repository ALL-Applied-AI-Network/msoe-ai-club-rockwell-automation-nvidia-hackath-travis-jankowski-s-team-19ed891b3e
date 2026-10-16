"""Fine-tune YOLO26-OBB on a BEV dataset, export ONNX and write a model card.

    python -m smartgrasp.train data/bev/sim_v1 --name sim_v1 --epochs 100

Output: artifacts/models/<name>/weights/{best.pt,best.onnx}, model_card.json, training plots.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from .geometry import Bev

AUGMENT = dict(
    fliplr=0.0,  # a mirror image puts the cube's visible side faces on the wrong side
    scale=0.1,  # a cube is always ~20 px in the metric BEV view
    hsv_h=0.01,  # colour is the class: keep hue shifts small
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="directory made by smartgrasp.dataset")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo26n-obb.pt", help="pretrained start point (yolo26s-obb.pt is larger)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", help="0 (CUDA), mps, cpu; default: best available")
    a = ap.parse_args()

    dataset = Path(a.dataset).resolve()
    imgsz = max(Bev.load(dataset).size)
    device = a.device or ("0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(a.model)
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=device,
        project=str(Path("artifacts/models").resolve()),
        name=a.name,
        exist_ok=True,
        deterministic=False,  # MPS has no deterministic scatter; harmless elsewhere
        **AUGMENT,
    )
    best = Path(model.trainer.best)
    onnx = YOLO(best).export(format="onnx", imgsz=imgsz)

    git = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    card = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git or None,
        "device": device,
        "ultralytics": ultralytics.__version__,
        "base_model": a.model,
        "dataset": a.dataset,
        "dataset_manifest": json.loads((dataset / "manifest.json").read_text()),
        "train_args": {"imgsz": imgsz, "epochs": a.epochs, "batch": a.batch, **AUGMENT},
        "val_metrics": {k: round(float(v), 4) for k, v in model.trainer.metrics.items()},
        "weights": [f"weights/{best.name}", f"weights/{Path(onnx).name}"],  # relative, so the folder can be shared
    }
    path = best.parent.parent / "model_card.json"
    path.write_text(json.dumps(card, indent=2))
    print(f"[train] weights {best}\n[train] onnx    {onnx}\n[train] card    {path}")


if __name__ == "__main__":
    main()

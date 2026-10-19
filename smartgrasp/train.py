"""Fine-tune YOLO26-OBB on a BEV dataset, export ONNX and write a model card.

    python -m smartgrasp.train data/bev/sim_v1 --name sim_v1 --epochs 100

Output: artifacts/models/<name>/weights/{best.pt,best.onnx}, model_card.json, training plots.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

from .dataset import DatasetManifest
from .geometry import Bev

AUGMENT: dict[str, float] = {
    "fliplr": 0.0,  # a mirror image puts the cube's visible side faces on the wrong side
    "scale": 0.1,  # a cube is always ~20 px in the metric BEV view
    "hsv_h": 0.01,  # colour is the class: keep hue shifts small
}


class ModelCard(TypedDict):
    """Provenance written next to the weights as model_card.json.

    Attributes:
        created: UTC time the card was written, ISO 8601 to the second.
        git_commit: `git describe --always --dirty` of this checkout, or None if it printed nothing.
        device: Ultralytics device the model trained on ("0", "mps" or "cpu").
        ultralytics: Ultralytics version.
        base_model: Pretrained weights the training started from.
        dataset: Dataset directory as given on the command line.
        dataset_manifest: The dataset's manifest.json.
        train_args: Image size, epochs and batch size, plus the AUGMENT settings.
        val_metrics: Final validation metrics, rounded to 4 decimals.
        weights: best.pt and best.onnx relative to the card, so the folder can be shared.
    """

    created: str
    git_commit: str | None
    device: str
    ultralytics: str
    base_model: str
    dataset: str
    dataset_manifest: DatasetManifest
    train_args: dict[str, float]
    val_metrics: dict[str, float]
    weights: list[str]


def _parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The dataset, run name, base model, epochs, batch size and device.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="directory made by smartgrasp.dataset")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="yolo26n-obb.pt", help="pretrained start point (yolo26s-obb.pt is larger)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", help="0 (CUDA), mps, cpu; default: best available")
    return parser.parse_args()


def _default_device() -> str:
    """Pick the fastest device this machine has.

    Returns:
        "0" (the first CUDA GPU), "mps" (Apple GPU) or "cpu".
    """
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _train(args: argparse.Namespace, dataset: Path, imgsz: int, device: str) -> BaseTrainer:
    """Fine-tune the pretrained model on the dataset.

    Args:
        args: Command line, for the base model, run name, epochs and batch size.
        dataset: Absolute dataset directory holding data.yaml.
        imgsz: Training image size in pixels (the BEV image's longer side).
        device: Ultralytics device string.

    Returns:
        The finished trainer, which holds the best checkpoint's path and the validation metrics.
    """
    model = YOLO(args.model)
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project=str(Path("artifacts/models").resolve()),
        name=args.name,
        exist_ok=True,
        deterministic=False,  # MPS has no deterministic scatter; harmless elsewhere
        **AUGMENT,
    )
    return cast(BaseTrainer, model.trainer)


def _git_commit() -> str | None:
    """Describe the commit this checkout is on.

    Returns:
        The `git describe --always --dirty` output, or None if it printed nothing (not a repo).
    """
    describe = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True)
    return describe.stdout.strip() or None


def main() -> None:
    """Train on the dataset, export the best weights to ONNX and write model_card.json."""
    args = _parse_args()
    dataset = Path(args.dataset).resolve()
    imgsz = max(Bev.load(dataset).size)
    device = args.device or _default_device()
    trainer = _train(args, dataset, imgsz, device)
    best = Path(trainer.best)
    onnx = YOLO(best).export(format="onnx", imgsz=imgsz)

    git_commit = _git_commit()
    metrics = cast(dict[str, float], trainer.metrics)
    card: ModelCard = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "device": device,
        "ultralytics": ultralytics.__version__,
        "base_model": args.model,
        "dataset": args.dataset,
        "dataset_manifest": json.loads((dataset / "manifest.json").read_text()),
        "train_args": {"imgsz": imgsz, "epochs": args.epochs, "batch": args.batch, **AUGMENT},
        "val_metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "weights": [f"weights/{best.name}", f"weights/{Path(onnx).name}"],
    }
    card_path = best.parent.parent / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    print(f"[train] weights {best}\n[train] onnx    {onnx}\n[train] card    {card_path}")


if __name__ == "__main__":
    main()

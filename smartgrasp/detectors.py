"""Cube detectors. All share one interface: BEV image in, list of Detection out.

    yolo:<weights>    YOLO26-OBB trained on synthetic twin images (.pt or exported .onnx)
    hsv               colour threshold + rotated rectangle, no training (reference baseline)
    vlm[:<dataset>]   vision-language model through an OpenAI-compatible API (in-context
                      learning). With a dataset, its first two labelled training images
                      are shown as worked examples before the question.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .dataset import read_labels
from .geometry import COLOURS, Bev, pose, twin


@dataclass
class Detection:
    cls: str  # one of geometry.CLASSES
    corners: np.ndarray  # (4, 2) box corners in BEV pixels
    conf: float

    def to_json(self, bev):
        xy, yaw = pose(self.corners, bev)
        return {
            "class": self.cls,
            "conf": round(float(self.conf), 3),
            "xy_m": [round(float(v), 5) for v in xy],
            "yaw_deg": round(float(yaw), 2),
            "corners_px": np.round(self.corners, 1).tolist(),
        }


class YoloDetector:
    def __init__(self, weights, conf=0.25):
        from ultralytics import YOLO  # imported here so hsv/vlm work without torch

        self.model, self.conf = YOLO(weights, task="obb"), conf

    def __call__(self, image):
        r = self.model.predict(image, imgsz=max(image.shape[:2]), conf=self.conf, verbose=False)[0]
        return [
            Detection(r.names[int(k)], c, p)
            for c, k, p in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.tolist(), r.obb.conf.tolist())
        ]


# OpenCV hue is 0-180. Same bands as the twin's image-geometry fitter (fanuc_er4ia_sim/cube_pose.py).
HSV_RANGES = {
    "yellow": [((12, 45, 35), (42, 255, 255))],
    "blue": [((95, 45, 35), (135, 255, 255))],
    "red": [((0, 45, 35), (12, 255, 255)), ((170, 45, 35), (180, 255, 255))],
}


def colour_blobs(image, colour, min_area, max_area):
    """Rotated boxes around blobs of one colour, largest first."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[colour]:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = sorted(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    return [cv2.boxPoints(cv2.minAreaRect(c)) for c in reversed(contours) if min_area <= cv2.contourArea(c) <= max_area]


class HsvDetector:
    """No learning. Visible side faces join the colour blob and bias its centre
    and yaw; that is the error a trained model should remove."""

    def __init__(self, bev):
        top = (twin.CUBE_EDGE * 1000 / bev.mm_per_px) ** 2  # top-face area in pixels
        self.area = (0.4 * top, 3.0 * top)  # caps (yellow) are ~8x larger and drop out

    def __call__(self, image):
        return [Detection(f"cube_{c}", box, 1.0) for c in COLOURS for box in colour_blobs(image, c, *self.area)]


class VlmDetector:
    PROMPT = (
        "Top-down image of a robot work table. Find every small cube (about 20 pixels wide); "
        "cubes are yellow, blue or red. Ignore round caps and long tubes. Reply with JSON only: "
        '[{"bbox_2d": BOX, "label": "red cube"}], coordinates on a 0-1000 grid.'
    )

    def __init__(self, bev, examples=None):
        _load_dotenv()
        base = os.environ.get("VLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = os.environ.get("VLM_MODEL", "google/gemma-4-31b-it")
        # The build.nvidia.com key goes only to NVIDIA; a self-hosted vLLM needs no key.
        self.key = os.environ.get("NVIDIA_BUILD_API_KEY", "") if "api.nvidia.com" in base else ""
        # Grounding models disagree on box order: Cosmos/Qwen answer [x1, y1, x2, y2], Gemma/Gemini [y1, x1, y2, x2].
        self.yx = os.environ.get("VLM_BOX_ORDER", "yxyx" if "gemma" in self.model else "xyxy") == "yxyx"
        self.prompt = self.PROMPT.replace("BOX", "[y1, x1, y2, x2]" if self.yx else "[x1, y1, x2, y2]")
        self.area = HsvDetector(bev).area
        self.shots = _examples(examples, self.yx) if examples else []
        if examples and not self.shots:
            raise SystemExit(f"vlm:{examples}: no labelled images in labels/train")

    def __call__(self, image):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for shot, answer in self.shots:
            messages += [_user(shot, self.prompt), {"role": "assistant", "content": answer}]
        messages.append(_user(image, self.prompt))
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": 2048, "temperature": 0.0})
        headers = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {self.key}"} if self.key else {})
        reply = self._ask(body.encode(), headers)
        if reply is None:
            return []
        try:
            items = json.loads(re.search(r"\[.*\]", reply, re.S).group(0))
        except (AttributeError, ValueError):
            print(f"[vlm] unparseable reply: {reply[:200]!r}")
            return []
        h, w = image.shape[:2]
        out = []
        for item in items:
            colour = next((c for c in COLOURS if c in str(item.get("label", ""))), None)
            if colour and len(item.get("bbox_2d", [])) == 4:
                box = np.array(item["bbox_2d"], float)[[1, 0, 3, 2] if self.yx else [0, 1, 2, 3]] / 1000 * (w, h, w, h)
                out.append(Detection(f"cube_{colour}", *self._fit(image, colour, *box)))
        return out

    def _ask(self, body, headers, attempts=2):
        """POST the chat request. A timeout or server error is retried once; after that the
        image counts as no detections, so one flaky call does not end a whole evaluation."""
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=300) as r:
                    return json.load(r)["choices"][0]["message"]["content"] or ""
            except (TimeoutError, urllib.error.URLError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    raise  # a bad key, model name or request: fix the configuration
                print(f"[vlm] request {attempt + 1}/{attempts} failed: {e}")
        return None

    def _fit(self, image, colour, x1, y1, x2, y2):
        """VLMs localise but cannot measure yaw: fit a rotated box to the colour blob in their box.

        Returns (corners, conf). Without a blob the VLM's box is only a guess, so conf 0 keeps it out of plans.
        """
        pad = 0.5 * max(x2 - x1, y2 - y1)
        x0, y0 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        crop = image[y0 : int(y2 + pad), x0 : int(x2 + pad)]
        boxes = colour_blobs(crop, colour, *self.area) if crop.size else []
        if boxes:
            return boxes[0] + (x0, y0), 1.0
        return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), 0.0


def _user(image, text):
    jpg = base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg}"}},
        {"type": "text", "text": text},
    ]}


def _examples(dataset, yx, n=2):
    """Worked examples for in-context learning: (image, correct JSON answer) from the train split."""
    pairs = []
    for label in sorted((Path(dataset) / "labels" / "train").glob("*.txt"))[:n]:
        image = cv2.imread(str(Path(dataset) / "images" / "train" / f"{label.stem}.jpg"))
        size = np.array(image.shape[1::-1], float)
        answer = []
        for cls, c in read_labels(label, size):
            if cls.startswith("cube_"):
                (x1, y1), (x2, y2) = c.min(0) / size * 1000, c.max(0) / size * 1000
                box = (y1, x1, y2, x2) if yx else (x1, y1, x2, y2)
                answer.append({"bbox_2d": [round(v) for v in box], "label": f"{cls[5:]} cube"})
        pairs.append((image, json.dumps(answer)))
    return pairs


def _load_dotenv(path=Path(__file__).resolve().parents[1] / ".env"):
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_detector(spec, bev=Bev()):
    kind, _, arg = spec.partition(":")
    if kind == "yolo" and arg:
        return YoloDetector(arg)
    if kind == "hsv":
        return HsvDetector(bev)
    if kind == "vlm":
        return VlmDetector(bev, arg or None)
    raise ValueError(f"Unknown detector {spec!r}; use yolo:<weights>, hsv or vlm[:<dataset>]")

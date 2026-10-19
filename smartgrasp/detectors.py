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
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

import cv2
import numpy as np

from .dataset import read_labels
from .geometry import COLOURS, Bev, Corners, FloatArray, Image, pose, read_image

if TYPE_CHECKING:  # type hints only: torch and ultralytics load lazily, see YoloDetector
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import OBB, Results

# Functional TypedDict syntax because "class" is a Python keyword; see Detection.to_json.
DetectionJson = TypedDict(
    "DetectionJson",
    {"class": str, "conf": float, "xy_m": list[float], "yaw_deg": float, "corners_px": list[list[float]]},
)


@dataclass
class Detection:
    """One detected object.

    Attributes:
        cls: Class name, one of geometry.CLASSES.
        corners: (4, 2) box corners in BEV pixels.
        conf: Confidence in [0, 1].
    """

    cls: str
    corners: Corners
    conf: float

    def to_json(self, bev: Bev) -> DetectionJson:
        """Describe the detection as rounded, JSON-ready values.

        Args:
            bev: The view the corners are in.

        Returns:
            Class, confidence, centre in robot-base metres, yaw in degrees folded to [-45, 45)
            and the corners in BEV pixels.
        """
        xy, yaw = pose(self.corners, bev)
        return {
            "class": self.cls,
            "conf": round(float(self.conf), 3),
            "xy_m": [round(float(v), 5) for v in xy],
            "yaw_deg": round(float(yaw), 2),
            "corners_px": np.round(self.corners, 1).tolist(),
        }


type Detector = Callable[[Image], list[Detection]]  # BEV image in, detections out


class YoloDetector:
    """YOLO26-OBB trained on synthetic twin images.

    Attributes:
        model: The network, from .pt weights or an exported .onnx file.
        conf: Lowest confidence a box needs to be returned.
    """

    model: "YOLO"
    conf: float

    def __init__(self, weights: str | Path, conf: float = 0.25) -> None:
        """Load the weights.

        Args:
            weights: .pt weights or an exported .onnx model.
            conf: Lowest confidence a box needs to be returned.
        """
        from ultralytics import YOLO  # imported here so hsv/vlm work without torch

        self.model, self.conf = YOLO(weights, task="obb"), conf

    def __call__(self, image: Image) -> list[Detection]:
        """Detect the objects in one image.

        Args:
            image: BEV image (BGR).

        Returns:
            One Detection per oriented box at or above the confidence threshold.
        """
        predictions = self.model.predict(image, imgsz=max(image.shape[:2]), conf=self.conf, verbose=False)
        result = cast("list[Results]", predictions)[0]
        boxes = cast("OBB", result.obb)
        corners = cast("torch.Tensor", boxes.xyxyxyxy).cpu().numpy()
        return [
            Detection(result.names[int(class_id)], box_corners, conf)
            for box_corners, class_id, conf in zip(corners, boxes.cls.tolist(), boxes.conf.tolist())
        ]


type _Hsv = tuple[int, int, int]

# (low, high) HSV bounds; OpenCV hue runs 0-180, so red wraps around and needs two bands.
HSV_RANGES: dict[str, list[tuple[_Hsv, _Hsv]]] = {
    "yellow": [((12, 45, 35), (42, 255, 255))],
    "blue": [((95, 45, 35), (135, 255, 255))],
    "red": [((0, 45, 35), (12, 255, 255)), ((170, 45, 35), (180, 255, 255))],
}


def colour_blobs(image: Image, colour: str, bev: Bev) -> list[Corners]:
    """Find rotated boxes around cube-sized blobs of one colour.

    The bands match the twin's image-geometry fitter (fanuc_er4ia_sim/cube_pose.py).

    Args:
        image: BEV image (BGR).
        colour: A key of HSV_RANGES.
        bev: The view the image is in, which sets the expected cube size.

    Returns:
        (4, 2) box corners in pixels for each blob between 0.4 and 3 cube top faces in area,
        largest first.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in HSV_RANGES[colour]:
        mask |= cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = sorted(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    top = bev.cube_px**2  # top-face area; caps (yellow) are ~8x larger and drop out
    return [
        cast(Corners, cv2.boxPoints(cv2.minAreaRect(contour)))
        for contour in reversed(contours)
        if 0.4 * top <= cv2.contourArea(contour) <= 3.0 * top
    ]


class HsvDetector:
    """Colour threshold plus rotated rectangle: the reference baseline, with no learning.

    Visible side faces join the colour blob and bias its centre and yaw; that is the error
    a trained model should remove.

    Attributes:
        bev: The view the images are in, which sets the expected cube size.
    """

    bev: Bev

    def __init__(self, bev: Bev) -> None:
        """Keep the view.

        Args:
            bev: The view the images are in.
        """
        self.bev = bev

    def __call__(self, image: Image) -> list[Detection]:
        """Detect the cubes in one image.

        Args:
            image: BEV image (BGR).

        Returns:
            One Detection (conf 1.0) per cube-sized blob, colour by colour.
        """
        return [
            Detection(f"cube_{colour}", box, 1.0) for colour in COLOURS for box in colour_blobs(image, colour, self.bev)
        ]


class _ImageUrl(TypedDict):
    """An image inside a chat message.

    Attributes:
        url: data: URL holding the base64 JPEG.
    """

    url: str


class _ImagePart(TypedDict):
    """The image part of a user message.

    Attributes:
        type: Always "image_url".
        image_url: The image.
    """

    type: Literal["image_url"]
    image_url: _ImageUrl


class _TextPart(TypedDict):
    """The text part of a user message.

    Attributes:
        type: Always "text".
        text: The question.
    """

    type: Literal["text"]
    text: str


class _Message(TypedDict):
    """One turn of an OpenAI-compatible chat request.

    Attributes:
        role: Who speaks.
        content: Plain text, or an image and a question for a user turn.
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[_ImagePart | _TextPart]


class _VlmBox(TypedDict, total=False):
    """One box in a VLM answer. A reply may omit or garble either field.

    Attributes:
        bbox_2d: Box corners on a 0-1000 grid, in the order the prompt asked for.
        label: Free text such as "red cube".
    """

    bbox_2d: list[float]
    label: str


class VlmDetector:
    """Vision-language model behind an OpenAI-compatible chat API, shown in-context examples.

    Configured by the environment (or the repo's .env): VLM_BASE_URL, VLM_MODEL,
    NVIDIA_BUILD_API_KEY and VLM_BOX_ORDER ("xyxy" or "yxyx").

    Attributes:
        PROMPT: Question template; BOX and PX are filled in per model and view.
        url: Chat completions endpoint.
        model: Model name sent with each request.
        key: API key, or "" when none is sent.
        yx: True when the model answers boxes as [y1, x1, y2, x2] instead of [x1, y1, x2, y2].
        prompt: The question asked about every image.
        bev: The view the images are in, which sets the cube size quoted in the prompt.
        shots: Worked examples as chat turns, encoded once: each question with its image, then
            the correct answer.
    """

    PROMPT: ClassVar[str] = (
        "Top-down image of a robot work table. Find every small cube (about PX pixels wide); "
        "cubes are yellow, blue or red. Ignore round caps and long tubes. Reply with JSON only: "
        '[{"bbox_2d": BOX, "label": "red cube"}], coordinates on a 0-1000 grid.'
    )
    url: str
    model: str
    key: str
    yx: bool
    prompt: str
    bev: Bev
    shots: list[_Message]

    def __init__(self, bev: Bev, examples: str | Path | None = None) -> None:
        """Read the endpoint settings and encode the worked examples.

        Args:
            bev: The view the images are in.
            examples: Dataset whose first two labelled training images become worked examples.

        Raises:
            SystemExit: If examples has no labelled training images.
        """
        _load_dotenv()
        base_url = os.environ.get("VLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = os.environ.get("VLM_MODEL", "google/gemma-4-31b-it")
        # The build.nvidia.com key goes only to NVIDIA; a self-hosted vLLM needs no key.
        self.key = os.environ.get("NVIDIA_BUILD_API_KEY", "") if "api.nvidia.com" in base_url else ""
        # Box order varies: Cosmos/Qwen answer [x1, y1, x2, y2], Gemma/Gemini [y1, x1, y2, x2].
        self.yx = os.environ.get("VLM_BOX_ORDER", "yxyx" if "gemma" in self.model else "xyxy") == "yxyx"
        box_order = "[y1, x1, y2, x2]" if self.yx else "[x1, y1, x2, y2]"
        self.prompt = self.PROMPT.replace("BOX", box_order).replace("PX", f"{bev.cube_px:.0f}")
        self.bev = bev
        self.shots = []
        if examples:
            for image, answer in _examples(examples, self.yx):
                self.shots += [_user_turn(image, self.prompt), {"role": "assistant", "content": answer}]
            if not self.shots:
                raise SystemExit(f"vlm:{examples}: no labelled images in labels/train")

    def __call__(self, image: Image) -> list[Detection]:
        """Ask the VLM where the cubes are, then fit each answer box to its colour blob.

        Args:
            image: BEV image (BGR).

        Returns:
            One Detection per answer box with a cube colour and four coordinates; [] if the
            request failed or the reply held no JSON list.
        """
        reply = self._ask(self._request_body(image))
        if reply is None:
            return []
        items = _parse_reply(reply)
        if items is None:
            return []
        height, width = image.shape[:2]
        detections: list[Detection] = []
        for item in items:
            label = str(item.get("label", ""))
            colour = next((c for c in COLOURS if c in label), None)
            bbox = item.get("bbox_2d", [])
            if colour and len(bbox) == 4:
                corners, conf = self._fit(image, colour, self._box_px(bbox, width, height))
                detections.append(Detection(f"cube_{colour}", corners, conf))
        return detections

    def _request_body(self, image: Image) -> bytes:
        """Encode the chat request: system prompt, worked examples, then the question about image.

        Args:
            image: BEV image (BGR).

        Returns:
            The JSON request body.
        """
        messages: list[_Message] = [
            {"role": "system", "content": "You are a helpful assistant."},
            *self.shots,
            _user_turn(image, self.prompt),
        ]
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": 2048, "temperature": 0.0})
        return body.encode()

    def _ask(self, body: bytes, attempts: int = 2) -> str | None:
        """POST the chat request and return the reply text.

        A timeout or server error is retried once; after that the image counts as no
        detections, so one flaky call does not end a whole evaluation.

        Args:
            body: JSON request body.
            attempts: How many times to try.

        Returns:
            The reply text, or None if every attempt failed.

        Raises:
            urllib.error.URLError: For a configuration mistake that a retry cannot fix: an HTTP
                4xx (bad key, model name or request) or an unknown or refusing host.
        """
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=300) as response:
                    return json.load(response)["choices"][0]["message"]["content"] or ""
            except TimeoutError as error:
                print(f"[vlm] request {attempt + 1}/{attempts} failed: {error}")
            except urllib.error.URLError as error:
                if _is_config_error(error):
                    raise
                print(f"[vlm] request {attempt + 1}/{attempts} failed: {error}")
        return None

    def _box_px(self, bbox_2d: list[float], width: int, height: int) -> FloatArray:
        """Scale an answer box from the 0-1000 grid to BEV pixels.

        Args:
            bbox_2d: Four coordinates on the 0-1000 grid, in the model's box order.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            (4,) array x1, y1, x2, y2 in pixels.
        """
        order = [1, 0, 3, 2] if self.yx else [0, 1, 2, 3]
        return np.array(bbox_2d, float)[order] / 1000 * (width, height, width, height)

    def _fit(self, image: Image, colour: str, box: FloatArray) -> tuple[Corners, float]:
        """Fit a rotated box to the colour blob inside the VLM's box.

        VLMs localise but cannot measure yaw, so the yaw comes from the blob. Without a blob
        the VLM's box is only a guess, and conf 0 keeps it out of plans.

        Args:
            image: BEV image (BGR).
            colour: Cube colour the VLM named.
            box: (4,) array x1, y1, x2, y2 in pixels.

        Returns:
            (corners, conf): (4, 2) corners in pixels and 1.0 if a blob was found, else the
            VLM's own box and 0.0.
        """
        x1, y1, x2, y2 = box
        pad = 0.5 * max(x2 - x1, y2 - y1)
        x0, y0 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        crop = image[y0 : int(y2 + pad), x0 : int(x2 + pad)]
        boxes = colour_blobs(crop, colour, self.bev) if crop.size else []
        if boxes:
            return boxes[0] + (x0, y0), 1.0
        return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), 0.0


def _is_config_error(error: urllib.error.URLError) -> bool:
    """Tell a configuration mistake, which a retry cannot fix, from a passing failure.

    Args:
        error: What urlopen raised.

    Returns:
        True for an HTTP 4xx (bad key, model name or request) or an unknown or refusing host
        (a wrong VLM_BASE_URL); False for a server error or a timeout.
    """
    if isinstance(error, urllib.error.HTTPError):
        return error.code < 500
    return isinstance(error.reason, (socket.gaierror, ConnectionRefusedError))


def _parse_reply(reply: str) -> list[_VlmBox] | None:
    """Pull the JSON list of boxes out of a VLM reply.

    Args:
        reply: The reply text, which may wrap the JSON in prose or a code fence.

    Returns:
        The parsed list, or None (after printing the reply) if it holds no valid JSON list.
    """
    match = re.search(r"\[.*\]", reply, re.S)
    if match is not None:
        try:
            return json.loads(match.group(0))
        except ValueError:
            pass
    print(f"[vlm] unparseable reply: {reply[:200]!r}")
    return None


def _user_turn(image: Image, text: str) -> _Message:
    """Build a user chat turn: the image (JPEG, quality 85), then the text.

    Args:
        image: BEV image (BGR).
        text: The question.

    Returns:
        The message, with the image inlined as a base64 data URL.
    """
    jpg = base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg}"}},
            {"type": "text", "text": text},
        ],
    }


def _examples(dataset: str | Path, yx: bool, count: int = 2) -> list[tuple[Image, str]]:
    """Build worked examples for in-context learning from a dataset's train split.

    Args:
        dataset: Dataset folder written by smartgrasp.dataset.
        yx: Write boxes as [y1, x1, y2, x2] instead of [x1, y1, x2, y2].
        count: How many labelled images to use (the first ones by name).

    Returns:
        (image, correct JSON answer) pairs, answers in the format the prompt asks for.

    Raises:
        FileNotFoundError: If a label file has no readable image.
    """
    root = Path(dataset)
    pairs: list[tuple[Image, str]] = []
    for label in sorted((root / "labels" / "train").glob("*.txt"))[:count]:
        image_path = root / "images" / "train" / f"{label.stem}.jpg"
        image = read_image(image_path)
        size = np.array(image.shape[1::-1], float)
        answer: list[_VlmBox] = []
        for name, corners in read_labels(label, size):
            if name.startswith("cube_"):
                (x1, y1), (x2, y2) = corners.min(0) / size * 1000, corners.max(0) / size * 1000
                box = (y1, x1, y2, x2) if yx else (x1, y1, x2, y2)
                answer.append({"bbox_2d": [round(v) for v in box], "label": f"{name.removeprefix('cube_')} cube"})
        pairs.append((image, json.dumps(answer)))
    return pairs


def _load_dotenv() -> None:
    """Copy KEY=value lines from the repo's .env into os.environ.

    Variables already set win. Lines starting with # are skipped and quotes around a value
    are stripped.
    """
    path = Path(__file__).resolve().parents[1] / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_detector(spec: str, bev: Bev = Bev()) -> Detector:
    """Build a detector from its command-line spec (see the module docstring).

    Args:
        spec: "yolo:<weights>", "hsv" or "vlm[:<dataset>]".
        bev: The view the images are in.

    Returns:
        A callable that takes a BEV image and returns its detections.

    Raises:
        ValueError: If spec names no known detector.
    """
    kind, _, arg = spec.partition(":")
    if kind == "yolo" and arg:
        return YoloDetector(arg)
    if kind == "hsv":
        return HsvDetector(bev)
    if kind == "vlm":
        return VlmDetector(bev, arg or None)
    raise ValueError(f"Unknown detector {spec!r}; use yolo:<weights>, hsv or vlm[:<dataset>]")

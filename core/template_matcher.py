"""YOLOv8-based template verification against the Praroop layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Union

import numpy as np

BBox = Sequence[float]
Detection = Mapping[str, Any]
ImageInput = Union[str, Path, np.ndarray]

DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "data" / "templates" / "praroop_template.json"
)
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "praroop_yolov8.pt"

# Canonical class names expected by the Praroop verifier
TARGET_LABELS = ("Signature", "Photo", "QR Code")


class TemplateMatcher:
    """
    Detect document fields with YOLOv8 and compare them to the Praroop template.

    Model loading is path-strict: missing weights raise ``FileNotFoundError``
    immediately. Weights are never auto-downloaded.
    """

    def __init__(
        self,
        template_path: str | Path = DEFAULT_TEMPLATE,
        model_path: str | Path | None = DEFAULT_WEIGHTS,
        model: Any | None = None,
        min_iou: float | None = None,
        max_center_offset: float | None = None,
    ) -> None:
        self.template_path = Path(template_path)
        self.template = self._load_template(self.template_path)

        thresholds = self.template.get("thresholds", {})
        self.min_iou = float(
            min_iou if min_iou is not None else thresholds.get("min_iou", 0.45)
        )
        self.max_center_offset = float(
            max_center_offset
            if max_center_offset is not None
            else thresholds.get("max_center_offset", 50.0)
        )

        self.model_path = Path(model_path) if model_path is not None else None
        self.model = model
        if self.model is None:
            self.model = self._load_yolo(self.model_path)

    @staticmethod
    def _load_template(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Praroop template JSON not found at: {path.resolve()}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if "elements" not in data:
            raise ValueError(f"Template missing 'elements': {path}")
        return data

    @staticmethod
    def _load_yolo(model_path: Path | None) -> Any:
        """Load local YOLOv8 weights once. Do not download on failure."""
        if model_path is None:
            raise FileNotFoundError(
                "No YOLO model_path provided and no injected model. "
                f"Expected weights at: {DEFAULT_WEIGHTS.resolve()}"
            )
        resolved = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at: {resolved}. "
                "Place a trained .pt file there or pass model_path explicitly. "
                "Auto-download is disabled."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for TemplateMatcher. Install project requirements."
            ) from exc

        # Pass an absolute existing path so Ultralytics does not treat it as a hub id.
        return YOLO(str(resolved))

    @staticmethod
    def _bbox_center(bbox: BBox) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _normalize_label(label: str) -> str:
        cleaned = " ".join(str(label).strip().split())
        aliases = {
            "signature": "Signature",
            "photo": "Photo",
            "photograph": "Photo",
            "qr": "QR Code",
            "qrcode": "QR Code",
            "qr_code": "QR Code",
            "qr code": "QR Code",
        }
        return aliases.get(cleaned.lower(), cleaned)

    def detect(self, image: ImageInput) -> list[dict[str, Any]]:
        """
        Run YOLOv8 and return detections as
        ``{"label", "confidence", "bbox": [x1, y1, x2, y2]}``.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")

        results = self.model.predict(source=str(image) if not isinstance(image, np.ndarray) else image, verbose=False)
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[dict[str, Any]] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        if xyxy is None:
            return detections

        # Torch tensors or numpy — coerce via tolist when available
        coords = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
        conf_list = confs.tolist() if confs is not None and hasattr(confs, "tolist") else [1.0] * len(coords)
        cls_list = clss.tolist() if clss is not None and hasattr(clss, "tolist") else [0] * len(coords)

        for bbox, conf, cls_id in zip(coords, conf_list, cls_list):
            raw_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
            label = self._normalize_label(raw_name)
            detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                }
            )
        return detections

    def compare_to_template(
        self,
        detections: Iterable[Detection],
    ) -> dict[str, Any]:
        """
        Compare detections to the Praroop template.

        Returns a report with per-element alignment stats and an ``issues`` list
        (e.g. ``\"Misaligned Photo\"``, ``\"Missing Signature\"``).
        """
        by_label: MutableMapping[str, list[Detection]] = {label: [] for label in TARGET_LABELS}
        for det in detections:
            label = self._normalize_label(str(det.get("label", "")))
            if label in by_label:
                by_label[label].append(det)

        elements = self.template["elements"]
        issues: list[str] = []
        details: dict[str, Any] = {}

        for label in TARGET_LABELS:
            expected = elements.get(label)
            if expected is None:
                continue
            expected_bbox = expected["bbox"]
            candidates = by_label.get(label, [])

            if not candidates:
                issues.append(f"Missing {label}")
                details[label] = {
                    "status": "missing",
                    "expected_bbox": list(expected_bbox),
                    "detected_bbox": None,
                    "iou": 0.0,
                    "center_offset": None,
                }
                continue

            # Best IoU match among detections of this class
            best = max(
                candidates,
                key=lambda d: self._bbox_iou(expected_bbox, d["bbox"]),
            )
            detected_bbox = best["bbox"]
            iou = self._bbox_iou(expected_bbox, detected_bbox)
            exp_c = self._bbox_center(expected_bbox)
            det_c = self._bbox_center(detected_bbox)
            offset = float(np.hypot(exp_c[0] - det_c[0], exp_c[1] - det_c[1]))

            aligned = iou >= self.min_iou and offset <= self.max_center_offset
            if not aligned:
                issues.append(f"Misaligned {label}")

            details[label] = {
                "status": "aligned" if aligned else "misaligned",
                "expected_bbox": list(expected_bbox),
                "detected_bbox": list(detected_bbox),
                "confidence": float(best.get("confidence", 0.0)),
                "iou": iou,
                "center_offset": offset,
            }

        return {
            "template": self.template.get("name", "Praroop"),
            "aligned": len(issues) == 0,
            "issues": issues,
            "elements": details,
        }

    def verify(self, image: ImageInput) -> dict[str, Any]:
        """Detect fields on ``image`` and compare against the Praroop template."""
        detections = self.detect(image)
        report = self.compare_to_template(detections)
        report["detections"] = detections
        return report

import argparse
import base64
import os
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"small_object_detection_worker_{worker_id}.log")


class SmallObjectDetectionWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_path=None,
        model_name="SmallObjectDetection",
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        limit_model_concurrency=2,
        host="0.0.0.0",
        port=None,
        model_semaphore=None,
        wait_timeout=180.0,
        task_timeout=180.0,
    ):
        self.class_names: List[str] = []
        self.class_name_to_id: Dict[str, int] = {}
        self.synonym_to_class: Dict[str, str] = {}
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            None,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info(f"Initializing model {self.model_name} from {self.model_path}")
        if not self.model_path:
            raise ValueError("model_path must be provided for SmallObjectDetection")
        self.model = YOLO(self.model_path)
        self.class_names = [self.model.names[idx] for idx in sorted(self.model.names.keys())]
        self.class_name_to_id = {name.lower(): idx for idx, name in enumerate(self.class_names)}
        self.synonym_to_class = self._build_synonym_mapping()
        logger.info(f"Loaded SmallObjectDetection classes: {self.class_names}")

    def _build_synonym_mapping(self) -> Dict[str, str]:
        mapping = {
            "plane": "plane",
            "airplane": "plane",
            "aircraft": "plane",
            "ship": "ship",
            "boat": "ship",
            "vessel": "ship",
            "storage tank": "storage tank",
            "tank": "storage tank",
            "baseball diamond": "baseball diamond",
            "baseball field": "baseball diamond",
            "tennis court": "tennis court",
            "basketball court": "basketball court",
            "ground track field": "ground track field",
            "groundtrack field": "ground track field",
            "track field": "ground track field",
            "harbor": "harbor",
            "harbour": "harbor",
            "bridge": "bridge",
            "large vehicle": "large vehicle",
            "truck": "large vehicle",
            "bus": "large vehicle",
            "heavy vehicle": "large vehicle",
            "small vehicle": "small vehicle",
            "car": "small vehicle",
            "vehicle": "small vehicle",
            "helicopter": "helicopter",
            "roundabout": "roundabout",
            "soccer ball field": "soccer ball field",
            "soccer field": "soccer ball field",
            "football field": "soccer ball field",
            "swimming pool": "swimming pool",
            "pool": "swimming pool",
        }
        return mapping

    def _load_image(self, image_input: str) -> Image.Image:
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        return Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().replace("_", " ").split())

    def _coerce_optional_bool(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    def _resolve_filter_class_ids(self, text: Optional[str]) -> Optional[List[int]]:
        if text is None:
            return None

        normalized = self._normalize_text(text)
        if not normalized:
            return None

        normalized = normalized.replace(",", " . ")
        candidates = [self._normalize_text(part) for part in normalized.split(".") if self._normalize_text(part)]
        resolved_ids: List[int] = []
        for candidate in candidates:
            canonical = self.synonym_to_class.get(candidate)
            if canonical is None and candidate in self.class_name_to_id:
                canonical = candidate
            if canonical is None:
                continue
            class_id = self.class_name_to_id.get(canonical)
            if class_id is not None and class_id not in resolved_ids:
                resolved_ids.append(class_id)
        if not resolved_ids:
            return []
        return resolved_ids

    def _generate_tile_windows(
        self,
        width: int,
        height: int,
        tile_size: int,
        tile_overlap: int,
    ) -> List[Tuple[int, int, int, int]]:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if tile_overlap < 0:
            raise ValueError("tile_overlap must be >= 0")
        if tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")

        if width <= tile_size and height <= tile_size:
            return [(0, 0, width, height)]

        stride = tile_size - tile_overlap

        def compute_starts(length: int) -> List[int]:
            if length <= tile_size:
                return [0]
            starts = list(range(0, max(length - tile_size, 0) + 1, stride))
            last_start = max(length - tile_size, 0)
            if starts[-1] != last_start:
                starts.append(last_start)
            return starts

        x_starts = compute_starts(width)
        y_starts = compute_starts(height)
        return [
            (x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height))
            for y0 in y_starts
            for x0 in x_starts
        ]

    def _predict_on_image(
        self,
        image_input: Any,
        score_threshold: float,
        filter_class_ids: Optional[List[int]],
    ):
        predict_kwargs = {
            "source": image_input,
            "conf": score_threshold,
            "verbose": False,
            "classes": filter_class_ids if filter_class_ids is not None else None,
        }
        if self.device and self.device != "auto":
            predict_kwargs["device"] = self.device
        return self.model.predict(**predict_kwargs)

    def _extract_hbb_detections(
        self,
        result,
        x_offset: int = 0,
        y_offset: int = 0,
        filter_class_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if result.obb is None or len(result.obb) == 0:
            return detections

        allowed_ids = None if filter_class_ids is None else set(int(x) for x in filter_class_ids)
        xyxyxyxy = result.obb.xyxyxyxy.detach().cpu().numpy()
        cls_ids = result.obb.cls.detach().cpu().numpy().astype(int)
        scores = result.obb.conf.detach().cpu().numpy()

        for polygon, cls_id, score in zip(xyxyxyxy, cls_ids, scores):
            if allowed_ids is not None and cls_id not in allowed_ids:
                continue

            xs = polygon[:, 0]
            ys = polygon[:, 1]
            x1 = float(xs.min()) + x_offset
            y1 = float(ys.min()) + y_offset
            x2 = float(xs.max()) + x_offset
            y2 = float(ys.max()) + y_offset

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": self.class_names[cls_id],
                    "score": float(score),
                    "class_id": int(cls_id),
                }
            )
        return detections

    def _run_full_image(
        self,
        image: Image.Image,
        score_threshold: float,
        filter_class_ids: Optional[List[int]],
    ) -> List[Dict[str, Any]]:
        result = self._predict_on_image(image, score_threshold, filter_class_ids)[0]
        return self._extract_hbb_detections(result, filter_class_ids=filter_class_ids)

    def _run_tiled(
        self,
        image: Image.Image,
        tile_size: int,
        tile_overlap: int,
        score_threshold: float,
        filter_class_ids: Optional[List[int]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        width, height = image.size
        windows = self._generate_tile_windows(width, height, tile_size, tile_overlap)
        detections: List[Dict[str, Any]] = []

        for x1, y1, x2, y2 in windows:
            tile = image.crop((x1, y1, x2, y2))
            result = self._predict_on_image(tile, score_threshold, filter_class_ids)[0]
            detections.extend(
                self._extract_hbb_detections(
                    result,
                    x_offset=x1,
                    y_offset=y1,
                    filter_class_ids=filter_class_ids,
                )
            )
        return detections, len(windows)

    def _apply_global_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []

        boxes = torch.tensor([det["bbox"] for det in detections], dtype=torch.float32)
        scores = torch.tensor([det["score"] for det in detections], dtype=torch.float32)
        class_ids = torch.tensor([det["class_id"] for det in detections], dtype=torch.float32)

        keep = torchvision.ops.batched_nms(boxes, scores, class_ids, iou_threshold)
        kept: List[Dict[str, Any]] = []
        for idx in keep.tolist():
            kept.append(detections[idx])
        return kept

    def _finalize_detections(
        self,
        detections: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        top1: bool,
        max_detections: Optional[int],
    ) -> List[Dict[str, Any]]:
        width, height = image_size
        finalized: List[Dict[str, Any]] = []

        for det in sorted(detections, key=lambda item: item["score"], reverse=True):
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, min(int(round(x1)), width))
            y1 = max(0, min(int(round(y1)), height))
            x2 = max(0, min(int(round(x2)), width))
            y2 = max(0, min(int(round(y2)), height))
            if x2 <= x1 or y2 <= y1:
                continue
            finalized.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": det["label"],
                    "score": round(float(det["score"]), 4),
                }
            )

        if top1 and finalized:
            finalized = finalized[:1]
        elif max_detections is not None and max_detections > 0:
            finalized = finalized[:max_detections]

        return finalized

    @torch.inference_mode()
    def generate(self, params):
        if "image" not in params:
            txt_e = "Missing required parameter: image"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        image_input = params.get("image")
        if os.path.exists(image_input) and not os.path.isfile(image_input):
            txt_e = f"Image path is not a file: {image_input}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 3}

        try:
            image = self._load_image(image_input)
        except Exception as e:
            txt_e = f"Failed to load image: {e}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 3}

        top1 = bool(params.get("top1", False))
        score_threshold = float(params.get("score_threshold", 0.25))
        use_tiling = self._coerce_optional_bool(params.get("use_tiling"))
        tile_size = int(params.get("tile_size", 1024))
        tile_overlap = int(params.get("tile_overlap", 128))
        max_detections = params.get("max_detections")
        max_detections = None if max_detections is None else int(max_detections)
        text = params.get("text")

        if score_threshold < 0 or score_threshold > 1:
            txt_e = "score_threshold must be in [0, 1]"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        try:
            filter_class_ids = self._resolve_filter_class_ids(text)
        except Exception as e:
            txt_e = f"Failed to parse text filter: {e}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 2}

        width, height = image.size
        auto_use_tiling = max(width, height) > 1600
        used_tiling = auto_use_tiling if use_tiling is None else use_tiling

        meta = {
            "used_tiling": used_tiling,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "score_threshold": score_threshold,
            "image_size": [width, height],
            "class_filter": None if filter_class_ids is None else [self.class_names[idx] for idx in filter_class_ids],
        }

        if filter_class_ids == []:
            logger.info(f"text filter '{text}' did not map to any supported class")
            return {
                "text": "",
                "error_code": 0,
                "detections": [],
                "meta": meta,
            }

        try:
            if used_tiling:
                raw_detections, tile_count = self._run_tiled(
                    image=image,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    score_threshold=score_threshold,
                    filter_class_ids=filter_class_ids,
                )
                meta["tile_count"] = tile_count
            else:
                raw_detections = self._run_full_image(
                    image=image,
                    score_threshold=score_threshold,
                    filter_class_ids=filter_class_ids,
                )
                meta["tile_count"] = 1

            merged_detections = self._apply_global_nms(raw_detections, iou_threshold=0.5)
            detections = self._finalize_detections(
                merged_detections,
                image_size=image.size,
                top1=top1,
                max_detections=max_detections,
            )

            text_lines = [
                f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), {det['label']}, score {det['score']:.2f}"
                for det in detections
                for bbox in [det["bbox"]]
            ]

            return {
                "text": "\n".join(text_lines),
                "error_code": 0,
                "detections": detections,
                "meta": meta,
            }
        except Exception as e:
            txt_e = f"Error in SmallObjectDetection: {e}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "SmallObjectDetection",
                "description": (
                    "Detect small remote-sensing objects using a YOLO11-OBB backend. "
                    "The worker internally converts oriented boxes to horizontal bounding boxes (HBB) "
                    "and returns only HBB outputs. `text` is a closed-set category filter, not open-vocabulary grounding."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image path on disk.",
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Optional closed-set class filter such as 'ship', 'plane', "
                                "'small vehicle', or 'large vehicle'. Synonyms like 'boat' and 'airplane' are supported."
                            ),
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, return only the highest-confidence detection.",
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Score threshold for filtering detections. Default is 0.25.",
                        },
                        "use_tiling": {
                            "type": "boolean",
                            "description": (
                                "Whether to force tiled inference. If omitted, tiling is enabled automatically "
                                "when the image long side is greater than 1600 pixels."
                            ),
                        },
                        "tile_size": {
                            "type": "integer",
                            "description": "Tile size for tiled inference. Default is 1024.",
                        },
                        "tile_overlap": {
                            "type": "integer",
                            "description": "Overlap between adjacent tiles in pixels. Default is 128.",
                        },
                        "max_detections": {
                            "type": "integer",
                            "description": "Optional cap on the number of returned detections after NMS and sorting.",
                        },
                    },
                    "required": ["image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20033)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="SmallObjectDetection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SmallObjectDetectionWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

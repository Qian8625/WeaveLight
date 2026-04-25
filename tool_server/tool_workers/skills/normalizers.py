import ast
import json
import re
from typing import Any, Dict, List, Tuple

def normalize_skill_params(
    skill_name: str,
    spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    params = dict(params or {})
    normalizer_name = spec.get("normalizer")

    if not normalizer_name:
        return params, []

    normalizer_registry = {
        "target_locate_measure": normalize_target_locate_measure,
        "sar_target_locate_measure": normalize_sar_target_locate_measure,
        "target_attribute": normalize_target_attribute,
        "conditional_count": normalize_conditional_count,
        "mult_confirm": normalize_mult_confirm,
        "change_summary": normalize_change_summary,
        "geotiff_poi_explore": normalize_geotiff_poi_explore,
        "geotiff_poi_distance": normalize_geotiff_poi_distance,
    }

    normalizer = normalizer_registry.get(normalizer_name)
    if normalizer is None:
        return params, [
            {
                "stage": "normalize_payload",
                "status": "skipped",
                "reason": f"Unknown normalizer: {normalizer_name}",
            }
        ]

    normalized, trace = normalizer(params)
    trace.insert(
        0,
        {
            "stage": "normalize_payload",
            "status": "start",
            "skill_name": skill_name,
            "normalizer": normalizer_name,
        },
    )
    return normalized, trace

def normalize_poi_specs(specs: Any) -> Any:
    """
    Normalize common LLM-produced POI specs into:
    [
      {"query": {"tourism": "museum"}, "layer_name": "museum"},
      {"query": {"shop": "mall"}, "layer_name": "mall"}
    ]

    Supported malformed inputs:
    - "[{'class': 'museum'}, {'class': 'mall'}]"
    - '[{"class": "museum"}, {"class": "mall"}]'
    - {"museum": {}, "mall": {}}
    - [{"museum": {}}, {"mall": {}}]
    - [{"type": "museum"}, {"type": "mall"}]
    - [{"class": "museum"}, {"class": "mall"}]
    - [{"category": "museum"}, {"category": "mall"}]
    - [{"query": {"tourism": "museum"}, "name": "museum"}]
    """

    # 关键新增：处理字符串形式的 list/dict
    if isinstance(specs, str):
        parsed = _parse_poi_specs_string(specs)
        if parsed is not specs:
            return normalize_poi_specs(parsed)
        return specs

    # {"museum": {}, "mall": {}}
    if isinstance(specs, dict):
        return [poi_type_to_spec(key) for key in specs.keys()]

    # [{"museum": {}}, {"mall": {}}]
    # [{"type": "museum"}, {"type": "mall"}]
    # [{"class": "museum"}, {"class": "mall"}]
    # [{"query": {"tourism": "museum"}, "name": "museum"}]
    if isinstance(specs, list):
        normalized = []

        for item in specs:
            if not isinstance(item, dict):
                normalized.append(item)
                continue

            # Already canonical or near-canonical
            if "query" in item:
                new_item = dict(item)

                if not new_item.get("layer_name") and new_item.get("name"):
                    new_item["layer_name"] = sanitize_name(new_item["name"], "poi_layer")

                if not new_item.get("layer_name"):
                    new_item["layer_name"] = layer_name_from_query(new_item.get("query"))

                normalized.append(new_item)
                continue

            # 关键新增：支持 class/category/name 作为 POI 类别字段
            poi_type = (
                item.get("type")
                or item.get("class")
                or item.get("category")
                or item.get("poi_class")
                or item.get("poi_type")
                or item.get("name")
            )
            if poi_type:
                normalized.append(poi_type_to_spec(poi_type))
                continue

            # {"museum": {}}
            if len(item) == 1:
                normalized.append(poi_type_to_spec(next(iter(item.keys()))))
                continue

            normalized.append(item)

        return normalized

    return specs

# -----------------------------
# Common helpers
# -----------------------------

def pick_first(params: Dict[str, Any], names: List[str], default=None):
    for name in names:
        value = params.get(name)
        if value not in [None, ""]:
            return value
    return default


def set_if_missing(params: Dict[str, Any], key: str, value: Any):
    if params.get(key) in [None, ""] and value not in [None, ""]:
        params[key] = value


def to_bool(value: Any, default: bool = False) -> bool:
    if value in [None, ""]:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "需要"}:
        return True
    if s in {"false", "0", "no", "n", "否", "不需要"}:
        return False
    return default


def to_float_or_none(value: Any):
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int_or_none(value: Any):
    if value in [None, ""]:
        return None
    try:
        return int(value)
    except Exception:
        return None


def sanitize_name(name: Any, default: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:30] or default


def normalize_target_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if x not in [None, ""])
    if isinstance(value, dict):
        for key in ["target", "object", "name", "text", "class"]:
            if value.get(key):
                return str(value[key])
        return str(value)
    return str(value).strip() if value not in [None, ""] else ""


def infer_locate_measure_mode(params: Dict[str, Any]) -> str:
    raw = pick_first(params, ["mode", "task_type", "operation", "intent", "action"])
    text = " ".join(str(x).lower() for x in [raw, params.get("query"), params.get("instruction")] if x)

    if any(k in text for k in ["distance", "距离"]):
        return "distance"
    if any(k in text for k in ["area", "面积"]):
        return "area"
    if any(k in text for k in ["segment", "segmentation", "分割"]):
        return "segment"
    if any(k in text for k in ["locate", "detect", "box", "bbox", "定位", "检测", "框选"]):
        return "locate"

    return str(raw).strip() if raw not in [None, ""] else "locate"

def _parse_poi_specs_string(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return value

    # 去掉可能的 markdown code fence
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        elif text.startswith("python"):
            text = text[6:].strip()

    # 先尝试标准 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试 Python literal，支持 "[{'class': 'museum'}]"
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return value

# -----------------------------
# 1. TargetLocateMeasureSkill
# -----------------------------

def normalize_target_locate_measure(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "mode", infer_locate_measure_mode(p))

    gsd = to_float_or_none(pick_first(p, ["gsd_m_per_pixel", "gsd", "resolution", "meter_per_pixel", "meters_per_pixel"]))
    if gsd is not None:
        p["gsd_m_per_pixel"] = gsd

    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "top1" in p:
        p["top1"] = to_bool(p["top1"], False)
    if "max_draw" in p:
        max_draw = to_int_or_none(p["max_draw"])
        if max_draw is not None:
            p["max_draw"] = max_draw

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["image", "target", "mode", "gsd_m_per_pixel"],
            "target": p.get("target"),
            "mode": p.get("mode"),
        }
    )
    return p, trace


# -----------------------------
# 2. SARTargetLocateMeasureSkill
# -----------------------------

def normalize_sar_target_locate_measure(params: Dict[str, Any]):
    p, trace = normalize_target_locate_measure(params)

    set_if_missing(p, "preprocess_mode", pick_first(p, ["preprocess_mode", "sar_preprocess_mode"], "sar_preprocess"))
    set_if_missing(p, "preprocessed_output_path", pick_first(p, ["preprocessed_output_path", "sar_preprocessed_output_path"]))

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["preprocess_mode", "preprocessed_output_path"],
            "preprocess_mode": p.get("preprocess_mode"),
        }
    )
    return p, trace


# -----------------------------
# 3. TargetAttributeSkill
# -----------------------------

def infer_attribute_task_type(params: Dict[str, Any]) -> str:
    raw = pick_first(params, ["task_type", "mode", "operation", "intent"])
    text = " ".join(str(x).lower() for x in [
        raw,
        params.get("query"),
        params.get("instruction"),
        params.get("attribute"),
        params.get("condition"),
    ] if x)

    if any(k in text for k in ["same", "different", "compare", "是否相同", "比较"]):
        return "compare"
    if any(k in text for k in ["count", "how many", "统计", "多少", "数量"]):
        return "count"
    if any(k in text for k in ["filter", "red", "blue", "颜色", "筛选"]):
        return "filter"
    if any(k in text for k in ["describe", "attribute", "属性", "描述"]):
        return "describe"

    return str(raw).strip() if raw not in [None, ""] else "describe"


def normalize_target_attribute(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "task_type", infer_attribute_task_type(p))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "attribute", pick_first(p, ["attribute", "attr", "property", "color", "orientation"]))
    set_if_missing(p, "attribute_value", pick_first(p, ["attribute_value", "value", "color_value", "target_value"]))
    set_if_missing(p, "compare_mode", pick_first(p, ["compare_mode", "comparison", "relation"]))

    if "top1" in p:
        p["top1"] = to_bool(p["top1"], False)
    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "use_segmentation" in p:
        p["use_segmentation"] = to_bool(p["use_segmentation"], False)
    if "max_draw" in p:
        max_draw = to_int_or_none(p["max_draw"])
        if max_draw is not None:
            p["max_draw"] = max_draw

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["image", "task_type", "target", "attribute"],
            "task_type": p.get("task_type"),
            "target": p.get("target"),
        }
    )
    return p, trace


# -----------------------------
# 4. ConditionalCountSkill
# -----------------------------

def normalize_condition(value: Any, params: Dict[str, Any]) -> str:
    if value not in [None, ""]:
        return str(value)

    pieces = []
    for key in ["condition", "status", "attribute", "attribute_value", "direction", "relation"]:
        if params.get(key):
            pieces.append(str(params[key]))

    if pieces:
        return " ".join(pieces)

    query = params.get("query") or params.get("instruction")
    return str(query).strip() if query not in [None, ""] else ""


def normalize_conditional_count(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "condition", normalize_condition(pick_first(p, ["condition", "filter"]), p))

    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "verify_examples" in p:
        p["verify_examples"] = to_bool(p["verify_examples"], False)
    if "max_draw" in p:
        max_draw = to_int_or_none(p["max_draw"])
        if max_draw is not None:
            p["max_draw"] = max_draw
    if "max_verify" in p:
        max_verify = to_int_or_none(p["max_verify"])
        if max_verify is not None:
            p["max_verify"] = max_verify

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["image", "target", "condition"],
            "target": p.get("target"),
            "condition": p.get("condition"),
        }
    )
    return p, trace


# -----------------------------
# 5. MultConfirmSkill
# -----------------------------

def normalize_mult_confirm(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "rgb_image", pick_first(p, ["rgb_image", "optical_image", "image", "primary_image"]))
    set_if_missing(p, "sar_image", pick_first(p, ["sar_image", "radar_image", "time1_image", "secondary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "task_type", pick_first(p, ["task_type", "mode", "operation"], "confirm"))
    set_if_missing(p, "preprocess_mode", pick_first(p, ["preprocess_mode", "sar_preprocess_mode"], "sar_preprocess"))

    if "iou_threshold" in p:
        iou = to_float_or_none(p["iou_threshold"])
        if iou is not None:
            p["iou_threshold"] = iou
    if "top1" in p:
        p["top1"] = to_bool(p["top1"], False)
    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "max_draw" in p:
        max_draw = to_int_or_none(p["max_draw"])
        if max_draw is not None:
            p["max_draw"] = max_draw

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["rgb_image", "sar_image", "target", "task_type"],
            "task_type": p.get("task_type"),
            "target": p.get("target"),
        }
    )
    return p, trace


# -----------------------------
# 6. ChangeSummarySkill
# -----------------------------

def infer_change_task_type(params: Dict[str, Any]) -> str:
    raw = pick_first(params, ["task_type", "mode", "operation", "intent"])
    text = " ".join(str(x).lower() for x in [
        raw,
        params.get("query"),
        params.get("instruction"),
        params.get("target"),
    ] if x)

    if any(k in text for k in ["damage", "damaged", "损毁", "受损"]):
        return "damage"
    if any(k in text for k in ["building", "建筑"]):
        return "building_change"
    if any(k in text for k in ["ship", "vehicle", "aircraft", "船", "车", "飞机"]):
        return "target_change"
    if any(k in text for k in ["change", "before", "after", "变化", "前后", "两期"]):
        return "generic"

    return str(raw).strip() if raw not in [None, ""] else "generic"


def normalize_change_summary(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "pre_image", pick_first(p, ["pre_image", "before_image", "time1_image", "image_before", "old_image"]))
    set_if_missing(p, "post_image", pick_first(p, ["post_image", "after_image", "time2_image", "image_after", "new_image"]))
    set_if_missing(p, "task_type", infer_change_task_type(p))
    set_if_missing(p, "query", pick_first(p, ["query", "instruction", "question"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["pre_image", "post_image", "task_type", "query", "target"],
            "task_type": p.get("task_type"),
            "target": p.get("target"),
        }
    )
    return p, trace


# -----------------------------
# 7/8. GeoTIFF POI skills
# -----------------------------

def normalize_geotiff_poi_explore(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "geotiff", pick_first(p, ["geotiff", "image", "primary_image", "tif", "tiff"]))
    p["poi_specs"] = normalize_poi_specs(p.get("poi_specs"))

    if "show_names" in p:
        p["show_names"] = to_bool(p["show_names"], True)
    if "describe_rendered" in p:
        p["describe_rendered"] = to_bool(p["describe_rendered"], False)

    set_if_missing(p, "task_type", pick_first(p, ["task_type", "mode", "operation"], "visualize"))

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["geotiff", "poi_specs", "task_type"],
            "poi_layers": extract_layer_names(p.get("poi_specs")),
        }
    )
    return p, trace


def normalize_geotiff_poi_distance(params: Dict[str, Any]):
    p, trace = normalize_geotiff_poi_explore(params)

    layers = extract_layer_names(p.get("poi_specs"))
    if len(layers) >= 2:
        set_if_missing(p, "src_layer", layers[0])
        set_if_missing(p, "tar_layer", layers[1])
        set_if_missing(p, "layers_to_render", layers)

    top = to_int_or_none(pick_first(p, ["top", "nearest_top", "k"]))
    if top is not None:
        p["top"] = top

    if "render_distance_layer" in p:
        p["render_distance_layer"] = to_bool(p["render_distance_layer"], True)

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["src_layer", "tar_layer", "top", "layers_to_render"],
            "src_layer": p.get("src_layer"),
            "tar_layer": p.get("tar_layer"),
            "poi_specs_type": type(p.get("poi_specs")).__name__,
            "poi_layers": layers,
        }
    )
    return p, trace


def poi_type_to_spec(poi_type: Any) -> Dict[str, Any]:
    value = str(poi_type or "").strip().lower()

    if value in {"museum", "museums", "博物馆"}:
        return {"query": {"tourism": "museum"}, "layer_name": "museum"}

    if value in {"mall", "malls", "shopping_mall", "shopping mall", "商场"}:
        return {"query": {"shop": "mall"}, "layer_name": "mall"}

    if value in {"hospital", "hospitals", "医院"}:
        return {"query": {"amenity": "hospital"}, "layer_name": "hospital"}

    if value in {"school", "schools", "学校"}:
        return {"query": {"amenity": "school"}, "layer_name": "school"}

    if value in {"restaurant", "restaurants", "餐厅"}:
        return {"query": {"amenity": "restaurant"}, "layer_name": "restaurant"}

    return {"query": value, "layer_name": sanitize_name(value, "poi_layer")}


def layer_name_from_query(query: Any) -> str:
    if isinstance(query, dict) and len(query) == 1:
        key, value = next(iter(query.items()))
        value_lower = str(value).lower()

        if value_lower in {"museum", "mall", "hospital", "school", "restaurant"}:
            return sanitize_name(value_lower, "poi_layer")

        return sanitize_name(f"{key}_{value}", "poi_layer")

    return sanitize_name(str(query), "poi_layer")


def extract_layer_names(specs: Any) -> List[str]:
    if not isinstance(specs, list):
        return []
    return [
        item.get("layer_name")
        for item in specs
        if isinstance(item, dict) and item.get("layer_name")
    ]


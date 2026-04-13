import gradio as gr
import json
import re
import torch
import os
import html
from PIL import Image
import rasterio
import numpy as np
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "app", "gradio_tmp")


# ---------------------------------------------------------
# Config: extensible multi-image registry
# ---------------------------------------------------------
PRETRAINED_PATH = "/home/ubuntu/00_CPK/OpenEarthAgent"
MAX_TOOL_ROUNDS = 16

INPUT_SPECS = [
    {
        "key": "primary_image",
        "label": "Upload General Image / GeoTIFF",
        "tab": "General",
        "description": "General remote sensing image, photo, or GeoTIFF for single-image tasks.",
        "height": 320,
        "sample_aliases": {
            "S_10_preview.png": "./assets/S_10.tif",
        },
    },
    {
        "key": "lst_image",
        "label": "Upload LST Raster",
        "tab": "TVDI",
        "description": "Land Surface Temperature raster used by TVDIAnalysis.",
        "height": 250,
    },
    {
        "key": "ndvi_image",
        "label": "Upload NDVI Raster",
        "tab": "TVDI",
        "description": "NDVI raster used by TVDIAnalysis.",
        "height": 250,
    },
    {
        "key": "time1_image",
        "label": "Upload Time-1 Image",
        "tab": "Time Series",
        "description": "Earlier timestamp image, often used as pre_image.",
        "height": 250,
    },
    {
        "key": "time2_image",
        "label": "Upload Time-2 Image",
        "tab": "Time Series",
        "description": "Later timestamp image, often used as post_image.",
        "height": 250,
    },
]

INPUT_GROUPS = [
    {
        "tab": "General",
        "title": "General single-image tasks",
        "note": "Use this for detection, OCR, GeoTIFF overlay, map reasoning, or general remote sensing analysis.",
        "input_keys": ["primary_image"],
    },
    {
        "tab": "TVDI",
        "title": "Dual-raster TVDI analysis",
        "note": "Upload both LST and NDVI rasters. The runtime will inject them into TVDIAnalysis(ndvi_path, lst_path, ...).",
        "input_keys": ["lst_image", "ndvi_image"],
    },
    {
        "tab": "Time Series",
        "title": "Time-series / bi-temporal comparison",
        "note": "Upload two temporal images here. The runtime can bind them to pre_image / post_image for change analysis tools.",
        "input_keys": ["time1_image", "time2_image"],
    },
]

# Declarative tool argument binding:
# tool_name -> {tool_argument_name: registry_key}
TOOL_ARG_BINDINGS = {
    "OCR": {"image": "primary_image"},
    "DrawBox": {"image": "primary_image"},
    "AddText": {"image": "primary_image"},
    "TextToBbox": {"image": "primary_image"},
    "CountGivenObject": {"image": "primary_image"},
    "ImageDescription": {"image": "primary_image"},
    "RegionAttributeDescription": {"image": "primary_image"},
    "SegmentObjectPixels": {"image": "primary_image"},
    "ObjectDetection": {"image": "primary_image"},
    "GetBboxFromGeotiff": {"geotiff": "primary_image"},
    "DisplayOnGeotiff": {"geotiff": "primary_image"},
    "ChangeDetection": {
        "pre_image": "time1_image",
        "post_image": "time2_image",
    },
    "TVDIAnalysis": {
        "ndvi_path": "ndvi_image",
        "lst_path": "lst_image",
    },
    "SARToRGB": {
        "image": "primary_image",
    },
    "SARPreprocessing": {
        "image": "primary_image",
    },
}

TOOL_DEFAULT_ARGUMENTS = {
    "AddText": {"color": "green"},
    "TVDIAnalysis": {"output_path": "tvdi_result.tif"},
}

GPKG_REQUIRED_TOOLS = {
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnMap",
    "AddIndexLayer",
    "AddDEMLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "DisplayOnGeotiff",
}


# ---------------------------------------------------------
# Model Class
# ---------------------------------------------------------
class LLM:
    def __init__(self, pretrained):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            use_fast=True,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.system_prompt = RS_AGENT_PROMPT
        self.max_new_tokens = 256

    def prepend_system_prompt(self, conversation):
        if not conversation or conversation[0]["role"] != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def generate(self, conversation):
        conversation = self.prepend_system_prompt(conversation)
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------
def extract_actions(text: str):
    try:
        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None, "No action found."
        actions_str = actions_match.group(1)
        actions_list = json.loads(actions_str)
        return actions_list, None
    except Exception:
        return None, "Invalid action format."


def try_parse_agent_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


def extract_thought_and_actions(text: str):
    data = try_parse_agent_json(text)
    thought = data.get("thought", "")
    actions = data.get("actions", None)

    if actions is None:
        actions, _ = extract_actions(text)
        if actions is None:
            actions = []

    return thought, actions


# ---------------------------------------------------------
# File / Preview Helpers
# ---------------------------------------------------------
def resolve_existing_file(path):
    if not path:
        return None

    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getcwd(), path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def make_preview_if_tif(path):
    if not path:
        return path

    if path.lower().endswith((".tif", ".tiff")):
        png_path = re.sub(r"\.(tif|tiff)$", "_preview.png", path, flags=re.IGNORECASE)

        with rasterio.open(path) as src:
            bands = src.count

            if bands >= 3:
                data = src.read([1, 2, 3])
                data = np.transpose(data, (1, 2, 0))
            else:
                data = src.read(1)
                if data.dtype != np.uint8:
                    data = np.nan_to_num(data)
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                    if data_max > data_min:
                        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        data = np.zeros_like(data, dtype=np.uint8)
                data = np.stack([data] * 3, axis=-1)

            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)

            img = Image.fromarray(data)
            img.save(png_path)

        return png_path

    return path


def handle_upload(image_path):
    if not image_path:
        return None, None

    resolved = resolve_existing_file(image_path) or image_path
    if resolved.lower().endswith((".tif", ".tiff")):
        preview_path = make_preview_if_tif(resolved)
        return preview_path, resolved

    return resolved, None


def normalize_uploaded_input(image_path, original_tif, sample_aliases=None):
    raw = original_tif or image_path
    if not raw:
        return None

    resolved = resolve_existing_file(raw) or raw

    if sample_aliases:
        basename = os.path.basename(resolved)
        if basename in sample_aliases:
            alias_target = resolve_existing_file(sample_aliases[basename])
            if alias_target:
                return alias_target

    return resolved


def detect_output_visual(tool_resp):
    if not isinstance(tool_resp, dict):
        return None

    candidate_keys = ["image", "output_path", "out_file", "path", "preview", "png"]
    for key in candidate_keys:
        value = tool_resp.get(key)
        if isinstance(value, str):
            found = resolve_existing_file(value)
            if found and found.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                return found

    for value in tool_resp.values():
        if isinstance(value, str):
            found = resolve_existing_file(value)
            if found and found.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                return found

    text = tool_resp.get("text", "")
    if isinstance(text, str):
        matches = re.findall(r"([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|tif|tiff))", text, re.I)
        for match in reversed(matches):
            found = resolve_existing_file(match)
            if found:
                return found

    return None


# ---------------------------------------------------------
# Registry Helpers
# ---------------------------------------------------------
def get_input_spec(key):
    for spec in INPUT_SPECS:
        if spec["key"] == key:
            return spec
    raise KeyError(f"Unknown input spec key: {key}")


def build_registry_from_args(flat_args):
    registry = {}
    arg_index = 0
    for spec in INPUT_SPECS:
        image_value = flat_args[arg_index]
        original_tif_value = flat_args[arg_index + 1]
        arg_index += 2
        registry[spec["key"]] = normalize_uploaded_input(
            image_value,
            original_tif_value,
            sample_aliases=spec.get("sample_aliases"),
        )
    return registry


def describe_registry_for_prompt(registry):
    lines = []
    for spec in INPUT_SPECS:
        key = spec["key"]
        if registry.get(key):
            lines.append(f"- {key}: {spec['description']}")
    return lines


def build_initial_user_message(user_question, registry):
    lines = [user_question.strip()]
    input_lines = describe_registry_for_prompt(registry)

    if input_lines:
        lines.extend([
            "",
            "Available uploaded inputs:",
            *input_lines,
            "Use the most appropriate tool and bind these uploaded inputs to the correct tool arguments.",
        ])

    return "\n".join(lines)


def choose_visual_image(registry):
    priority = [
        "primary_image",
        "time2_image",
        "time1_image",
        "lst_image",
        "ndvi_image",
    ]
    for key in priority:
        if registry.get(key):
            return registry[key]
    return None


def inject_runtime_arguments(api_name, api_args, registry, current_gpkg):
    merged_args = dict(api_args)

    # Declarative bindings from registry to tool arguments
    for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
        if registry.get(registry_key):
            merged_args[tool_arg] = registry[registry_key]

    # Shared gpkg propagation
    if api_name in GPKG_REQUIRED_TOOLS and current_gpkg:
        merged_args["gpkg"] = current_gpkg

    # Tool-specific defaults
    for key, value in TOOL_DEFAULT_ARGUMENTS.get(api_name, {}).items():
        if key not in merged_args or merged_args.get(key) in [None, ""]:
            merged_args[key] = value

    return merged_args


def update_registry_with_tool_response(registry, api_name, tool_resp):
    if api_name == "TVDIAnalysis":
        output_path = tool_resp.get("output_path") or detect_output_visual(tool_resp)
        if output_path:
            registry["tvdi_result"] = output_path

    detected_visual = detect_output_visual(tool_resp)
    if detected_visual:
        registry["latest_output"] = detected_visual

    return registry


# ---------------------------------------------------------
# Trace Rendering
# ---------------------------------------------------------
def format_final_answer(ans: str):
    if not ans or ans.strip() == "<image>":
        return "### Result\n已生成结果图像，请查看下方 **Output Image**。"
    return f"### Result\n{ans}"


def render_trace_html(trace_steps):
    if not trace_steps:
        return """
        <div class="trace-root">
            <div class="trace-empty">
                Execution Trace will appear here, grouped by round.
            </div>
        </div>
        """

    blocks = ['<div class="trace-root">']

    for i, step in enumerate(trace_steps):
        round_id = step.get("round", "-")
        thought = html.escape(step.get("thought", "") or "No thought")
        action = step.get("action")
        arguments = step.get("arguments")
        tool_response = html.escape(step.get("tool_response", "") or "")
        final_answer = html.escape(step.get("final_answer", "") or "")
        status = step.get("status", "info")

        if status == "success":
            status_text = "SUCCESS"
            badge_cls = "badge-success"
        elif status == "error":
            status_text = "ERROR"
            badge_cls = "badge-error"
        elif status == "final":
            status_text = "FINAL"
            badge_cls = "badge-final"
        elif status == "planning":
            status_text = "PLAN"
            badge_cls = "badge-plan"
        else:
            status_text = "INFO"
            badge_cls = "badge-info"

        title = f"Round {round_id} · {html.escape(action)}" if action else f"Round {round_id} · Planning"
        open_attr = "open" if i >= len(trace_steps) - 2 else ""

        blocks.append(f"""
        <details class="trace-round" {open_attr}>
            <summary>
                <span class="trace-title">{title}</span>
                <span class="trace-badge {badge_cls}">{status_text}</span>
            </summary>
            <div class="trace-body">
                <div class="trace-section">
                    <div class="trace-label">Thought</div>
                    <pre>{thought}</pre>
                </div>
        """)

        if action:
            blocks.append(f"""
                <div class="trace-section">
                    <div class="trace-label">Action</div>
                    <div class="trace-inline-tag">{html.escape(action)}</div>
                </div>
            """)

        if arguments is not None:
            arg_str = html.escape(json.dumps(arguments, ensure_ascii=False, indent=2))
            blocks.append(f"""
                <div class="trace-section">
                    <div class="trace-label">Arguments</div>
                    <pre>{arg_str}</pre>
                </div>
            """)

        if tool_response:
            blocks.append(f"""
                <div class="trace-section">
                    <div class="trace-label">Tool Response</div>
                    <pre>{tool_response}</pre>
                </div>
            """)

        if final_answer:
            blocks.append(f"""
                <div class="trace-section">
                    <div class="trace-label">Final Answer</div>
                    <pre>{final_answer}</pre>
                </div>
            """)

        blocks.append("""
            </div>
        </details>
        """)

    blocks.append("</div>")
    return "".join(blocks)


# ---------------------------------------------------------
# Load model + tools ONCE
# ---------------------------------------------------------
Model = LLM(PRETRAINED_PATH)
tool_manager = ToolManager()


# ---------------------------------------------------------
# Chat Function for Gradio
# ---------------------------------------------------------
def run_agent(user_question, *flat_args):
    registry = build_registry_from_args(flat_args)
    current_visual_image = choose_visual_image(registry)
    current_gpkg = None
    trace_steps = []

    initial_message = build_initial_user_message(user_question, registry)
    conversation = [{"role": "user", "content": initial_message}]

    def current_preview():
        nonlocal current_visual_image
        latest = registry.get("latest_output") or registry.get("tvdi_result") or current_visual_image
        if latest and os.path.isfile(latest):
            return make_preview_if_tif(latest)
        return None

    def emit(answer_md=""):
        return answer_md, render_trace_html(trace_steps), current_preview()

    for round_id in range(1, MAX_TOOL_ROUNDS + 1):
        model_output = Model.generate(conversation)
        thought, actions = extract_thought_and_actions(model_output)

        if not actions:
            trace_steps.append({
                "round": round_id,
                "thought": thought or model_output,
                "status": "planning"
            })
            yield emit("")

            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0]
        api_name = action["name"]
        api_args = dict(action["arguments"])

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "")
            trace_steps.append({
                "round": round_id,
                "thought": thought,
                "action": api_name,
                "arguments": api_args,
                "status": "final",
                "final_answer": final_answer
            })
            yield emit(format_final_answer(final_answer))
            return

        trace_steps.append({
            "round": round_id,
            "thought": thought,
            "action": api_name,
            "arguments": dict(api_args),
            "status": "info"
        })
        yield emit("")

        injected_args = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)
        trace_steps[-1]["arguments"] = dict(injected_args)
        yield emit("")

        tool_resp = tool_manager.call_tool(api_name, injected_args)

        trace_steps[-1]["status"] = "success" if tool_resp.get("error_code") == 0 else "error"
        trace_steps[-1]["tool_response"] = tool_resp.get("text", "")

        if "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")

        registry = update_registry_with_tool_response(registry, api_name, tool_resp)

        detected_visual = registry.get("latest_output") or registry.get("tvdi_result")
        if detected_visual:
            current_visual_image = detected_visual

        yield emit("")

        tool_text = tool_resp.get("text", "")
        obs = (
            f"OBSERVATION:\n{api_name} outputs: {tool_text}\n"
            "Please summarize and answer the question."
        )
        conversation.append({"role": "assistant", "content": model_output})
        conversation.append({"role": "user", "content": obs})

    trace_steps.append({
        "round": "MAX",
        "thought": "Reached maximum tool rounds.",
        "status": "error",
        "final_answer": "Max tool rounds reached."
    })
    yield emit("### Result\nMax tool rounds reached.")


# ---------------------------------------------------------
# UI / CSS
# ---------------------------------------------------------
css = """
:root {
    /* =========================
       Color tokens
       ========================= */
    --bg: #FFFFFF;
    --panel: #FFFFFF;
    --panel-soft: #EAF2F5;
    --line: #D6E7EE;
    --line-2: #B9D8E4;
    --primary: #95C5D4;
    --primary-soft: #DCECF2;
    --warm-1: #F8E4DB;
    --warm-2: #FCD6B3;
    --text: #2F3A45;
    --muted: #6E7B87;
    --success-bg: #DFF4EA;
    --success-fg: #1F7A55;
    --error-bg: #FBE5E5;
    --error-fg: #B54747;
    --final-bg: #F8E4DB;
    --final-fg: #8D5D48;
    --plan-bg: #E6ECF2;
    --plan-fg: #546271;
    --info-bg: #D3E6EC;
    --info-fg: #496C81;

    /* =========================
       Size tokens
       ========================= */
    --app-min-width: 1280px;
    --app-max-width: 1880px;

    --page-pad-x: clamp(24px, 2vw, 40px);
    --page-pad-y: clamp(18px, 1.6vh, 28px);
    --layout-gap: clamp(18px, 1vw, 24px);

    --hero-pad-y: clamp(20px, 2vh, 28px);
    --hero-pad-x: clamp(24px, 1.8vw, 34px);
    --panel-pad: clamp(20px, 1.35vw, 30px);

    --hero-height: clamp(108px, 11vh, 132px);

    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;

    --shadow: 0 10px 30px rgba(73, 108, 129, 0.08);

    --title-size: clamp(28px, 1.7vw, 36px);
    --subtitle-size: clamp(15px, 0.95vw, 17px);
    --section-title-size: clamp(20px, 1.1vw, 24px);
    --body-size: 15px;
    --mono-size: 13px;

    --question-min-h: 118px;
    --button-h: 54px;

    --input-image-lg-h: clamp(300px, 30vh, 390px);
    --input-image-md-h: clamp(240px, 23vh, 310px);
    --output-image-h: clamp(360px, 35vh, 500px);
    --answer-min-h: 120px;
    --trace-min-h: 420px;

    --scrollbar-w: 10px;
}

html, body {
    width: 100%;
    height: 100%;
    min-width: var(--app-min-width);
    overflow: hidden !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: "Microsoft YaHei", Arial, sans-serif;
}

.gradio-container {
    width: min(var(--app-max-width), calc(100vw - 2 * var(--page-pad-x))) !important;
    max-width: none !important;
    min-width: var(--app-min-width) !important;
    height: 100vh !important;
    margin: 0 auto !important;
    padding: var(--page-pad-y) var(--page-pad-x) !important;
    overflow: hidden !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#app-shell {
    height: calc(100vh - 2 * var(--page-pad-y));
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: var(--layout-gap);
    overflow: hidden;
}

#hero-row {
    flex: 0 0 auto;
    min-height: var(--hero-height);
    margin: 0 !important;
}

#main-shell {
    flex: 1 1 auto;
    min-height: 0;
    margin: 0 !important;
    gap: var(--layout-gap) !important;
    align-items: stretch !important;
    overflow: hidden !important;
}

.hero-wrap {
    border: 1px solid var(--line);
    border-radius: calc(var(--radius-xl) + 2px);
    padding: var(--hero-pad-y) var(--hero-pad-x);
    background: linear-gradient(135deg, #FFFFFF 0%, #E6ECF2 55%, #F8E4DB 100%);
    box-shadow: var(--shadow);
}

.hero-title h1,
.hero-title h2,
.hero-title h3,
.hero-title p {
    margin: 0 !important;
    color: var(--text) !important;
}

.hero-title h1 {
    font-size: var(--title-size) !important;
    line-height: 1.15 !important;
}

.hero-title h3 {
    font-size: calc(var(--title-size) * 0.48) !important;
    line-height: 1.25 !important;
}

.hero-sub {
    color: var(--muted) !important;
    margin-top: 8px !important;
    font-size: var(--subtitle-size) !important;
    line-height: 1.45 !important;
}

.panel-card {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-xl) !important;
    padding: var(--panel-pad) !important;
    box-shadow: var(--shadow) !important;

    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

#left-pane {
    height: 100%;
}

#right-pane {
    height: 100%;
}

#result-scroll {
    flex: 1 1 auto;
    min-height: 0;
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 8px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

#result-scroll::-webkit-scrollbar {
    width: var(--scrollbar-w);
}

#result-scroll::-webkit-scrollbar-thumb {
    background: #C8DDE6;
    border-radius: 999px;
}

#result-scroll::-webkit-scrollbar-track {
    background: transparent;
}

.section-head {
    margin-bottom: 10px;
}

.section-head h3 {
    margin: 0;
    color: var(--text);
    font-size: var(--section-title-size);
    font-weight: 700;
    line-height: 1.2;
}

.section-head p {
    margin: 6px 0 0;
    color: var(--muted);
    font-size: var(--body-size);
    line-height: 1.55;
}

.soft-note {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFD 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    color: var(--muted);
    font-size: var(--body-size);
    line-height: 1.5;
    margin-bottom: 12px;
}

.gradio-textbox,
.gradio-textbox textarea,
.gradio-textbox input {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--line) !important;
    background: #FFFFFF !important;
    color: var(--text) !important;
    box-shadow: none !important;
    font-size: 16px !important;
}

.question-box textarea {
    min-height: var(--question-min-h) !important;
    line-height: 1.6 !important;
}

.gradio-textbox textarea:focus,
.gradio-textbox input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(149, 197, 212, 0.18) !important;
}

.gradio-image {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px dashed var(--line-2) !important;
    background: #FAFCFD !important;
    box-shadow: none !important;
}

.input-image-lg,
.input-image-lg > div {
    height: var(--input-image-lg-h) !important;
    min-height: var(--input-image-lg-h) !important;
}

.input-image-md,
.input-image-md > div {
    height: var(--input-image-md-h) !important;
    min-height: var(--input-image-md-h) !important;
}

.output-image-lg,
.output-image-lg > div {
    height: var(--output-image-h) !important;
    min-height: var(--output-image-h) !important;
}

.gradio-image img,
.gradio-image canvas {
    object-fit: contain !important;
}

.gradio-button {
    border-radius: var(--radius-md) !important;
    min-height: var(--button-h) !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.action-row {
    margin-top: 6px !important;
}

.submit-btn {
    background: var(--primary) !important;
    color: #FFFFFF !important;
}

.submit-btn:hover {
    filter: brightness(0.97);
    transform: translateY(-1px);
}

.clear-btn {
    background: var(--warm-2) !important;
    color: var(--text) !important;
}

.clear-btn:hover {
    filter: brightness(0.98);
    transform: translateY(-1px);
}

.soft-accordion {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--line) !important;
    background: #FCFDFE !important;
    margin-top: 14px;
}

.gradio-accordion summary {
    font-weight: 700 !important;
    color: var(--text) !important;
    font-size: 15px !important;
}

.gradio-examples .gr-example {
    border-radius: var(--radius-md) !important;
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    padding: 14px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.gradio-examples .gr-example:hover {
    background: #F8FBFC !important;
    border-color: var(--primary) !important;
    transform: translateY(-1px);
}

.answer-box {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px 22px !important;
    background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFC 100%) !important;
    min-height: var(--answer-min-h);
}

.answer-box h1,
.answer-box h2,
.answer-box h3,
.answer-box p {
    color: var(--text) !important;
}

.trace-shell {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    background: #FBFDFE !important;
    padding: 12px !important;
    min-height: var(--trace-min-h);
    max-height: none !important;
    overflow: visible !important;
}

.trace-root {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.trace-empty {
    padding: 24px;
    border: 1px dashed var(--line-2);
    border-radius: var(--radius-md);
    color: var(--muted);
    background: #FFFFFF;
    font-size: 15px;
}

.trace-round {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: #FFFFFF;
    overflow: hidden;
}

.trace-round summary {
    list-style: none;
    cursor: pointer;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: linear-gradient(90deg, #FFFFFF 0%, #F7FAFB 100%);
}

.trace-round summary::-webkit-details-marker {
    display: none;
}

.trace-title {
    font-weight: 700;
    color: var(--text);
    font-size: 14px;
}

.trace-badge {
    font-size: 12px;
    font-weight: 700;
    padding: 6px 10px;
    border-radius: 999px;
}

.badge-success {
    background: var(--success-bg);
    color: var(--success-fg);
}

.badge-error {
    background: var(--error-bg);
    color: var(--error-fg);
}

.badge-final {
    background: var(--final-bg);
    color: var(--final-fg);
}

.badge-plan {
    background: var(--plan-bg);
    color: var(--plan-fg);
}

.badge-info {
    background: var(--info-bg);
    color: var(--info-fg);
}

.trace-body {
    padding: 16px 18px 18px;
    border-top: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.trace-section {
    background: #FAFCFD;
    border: 1px solid #EDF3F6;
    border-radius: 12px;
    padding: 14px;
}

.trace-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.trace-section pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: var(--mono-size);
    line-height: 1.6;
    color: var(--text);
    background: transparent;
}

.trace-inline-tag {
    display: inline-block;
    background: var(--primary-soft);
    color: #35556A;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.gr-markdown p,
.gr-markdown h1,
.gr-markdown h2,
.gr-markdown h3 {
    color: var(--text) !important;
}

@media (min-width: 1920px) {
    :root {
        --page-pad-x: 36px;
        --page-pad-y: 24px;
        --panel-pad: 28px;
        --input-image-lg-h: 400px;
        --input-image-md-h: 310px;
        --output-image-h: 500px;
        --trace-min-h: 480px;
    }
}

@media (max-width: 1440px) {
    :root {
        --page-pad-x: 20px;
        --layout-gap: 16px;
        --panel-pad: 18px;
        --input-image-lg-h: 300px;
        --input-image-md-h: 240px;
        --output-image-h: 360px;
    }
}

@media (max-width: 1280px) {
    :root {
        --app-min-width: 1280px;
        --page-pad-x: 16px;
        --page-pad-y: 16px;
    }
}
"""
# ---------------------------------------------------------
# Example Helpers
# ---------------------------------------------------------
def make_example(question, **paths):
    row = [question]
    for spec in INPUT_SPECS:
        row.append(paths.get(spec["key"]))
    return row


# ---------------------------------------------------------
# Build UI
# ---------------------------------------------------------
logo_path = "./assets/nwpu-logo.svg"

with gr.Blocks(title="WeavLight Demo", css=css, fill_height=True) as demo:
    state_components = {}
    image_components = {}

    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="hero-row", elem_classes=["hero-wrap"]):
            with gr.Column(scale=1, min_width=96):
                gr.Image(
                    logo_path,
                    show_label=False,
                    container=False,
                    height=72
                )
            with gr.Column(scale=8, elem_classes=["hero-title"]):
                gr.Markdown("""
                # WeavLight
                ### Config-Driven Multi-Image Remote Sensing Agent
                <div class="hero-sub">White background · Foldable examples · Round-based execution trace · Registry + binding-table architecture</div>
                """)

        with gr.Row(equal_height=True, elem_id="main-shell"):
            with gr.Column(
                scale=11,
                min_width=620,
                elem_id="left-pane",
                elem_classes=["panel-card"]
            ):
                gr.Markdown("""
                <div class="section-head">
                    <h3>Input</h3>
                    <p>This version uses a named input registry. To add more image roles later, extend <code>INPUT_SPECS</code> and <code>TOOL_ARG_BINDINGS</code>.</p>
                </div>
                """)

                user_input = gr.Textbox(
                    label="Question",
                    placeholder="Please enter your remote sensing image analysis question...",
                    lines=5,
                    elem_classes=["gradio-textbox", "question-box"]
                )

                with gr.Tabs():
                    for group in INPUT_GROUPS:
                        with gr.Tab(group["tab"]):
                            gr.Markdown(f'<div class="soft-note">{group["note"]}</div>')
                            group_keys = group["input_keys"]

                            if len(group_keys) == 1:
                                spec = get_input_spec(group_keys[0])
                                component = gr.Image(
                                    type="filepath",
                                    label=spec["label"],
                                    sources=["upload", "clipboard"],
                                    elem_classes=["gradio-image", "input-image", "input-image-lg"]
                                )
                                image_components[spec["key"]] = component
                                state_components[spec["key"]] = gr.State()
                            else:
                                with gr.Row():
                                    for key in group_keys:
                                        spec = get_input_spec(key)
                                        component = gr.Image(
                                            type="filepath",
                                            label=spec["label"],
                                            sources=["upload", "clipboard"],
                                            elem_classes=["gradio-image", "input-image", "input-image-md"]
                                        )
                                        image_components[spec["key"]] = component
                                        state_components[spec["key"]] = gr.State()

                for spec in INPUT_SPECS:
                    image_components[spec["key"]].change(
                        fn=handle_upload,
                        inputs=image_components[spec["key"]],
                        outputs=[image_components[spec["key"]], state_components[spec["key"]]],
                        trigger_mode="once"
                    )

                with gr.Row(elem_classes=["action-row"]):
                    submit_btn = gr.Button("Run", elem_classes=["submit-btn"], scale=3)
                    clear_btn = gr.Button("Clear", elem_classes=["clear-btn"], scale=2)

                with gr.Accordion("Sample Queries", open=False, elem_classes=["soft-accordion"]):
                    gr.Markdown("Click any example to auto-fill the corresponding inputs. Every row matches the registry order.")
                    example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]

                    with gr.Tabs():
                        with gr.Tab("General / Index"):
                            gr.Examples(
                                examples=[
                                    make_example(
                                        "Generate a preview from the NBR difference for Topanga State Park, Los Angeles, USA, covering December 2024 and February 2025."
                                    ),
                                    make_example(
                                        "Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.",
                                        primary_image="./assets/S_10_preview.png",
                                    ),
                                    make_example(
                                        "Locate and estimate the distance between aircrafts in the scene. Assuming GSD 0.6 px/meter",
                                        primary_image="./assets/TG_P0009.png",
                                    ),
                                ],
                                inputs=example_inputs,
                                preprocess=False,
                            )

                        with gr.Tab("Terrain / DEM"):
                            gr.Examples(
                                examples=[
                                    make_example(
                                        "For Manchester State Forest, South Carolina, United States, create a DEM layer from GEE, generate 20 m contours and 100 m elevation bands, and summarize the elevation range."
                                    ),
                                    make_example(
                                        "For the area within a 1000 m radius of Edinburgh Castle, generate a DEM layer, create contour lines, and visualize the resulting terrain bands on the map."
                                    ),
                                ],
                                inputs=example_inputs,
                                preprocess=False,
                            )

                        with gr.Tab("TVDI"):
                            gr.Examples(
                                examples=[
                                    make_example(
                                        "Compute the Temperature Vegetation Dryness Index (TVDI) from NDVI and LST rasters.",
                                        lst_image="./assets/Sichuan_2021-07-12_LST.tif",
                                        ndvi_image="./assets/Sichuan_2021-07-12_NDVI.tif",
                                    ),
                                ],
                                inputs=example_inputs,
                                preprocess=False,
                            )

                        with gr.Tab("Time Series"):
                            gr.Examples(
                                examples=[
                                    make_example(
                                        "Compare the two temporal images and identify the major changes.",
                                        time1_image=None,
                                        time2_image=None,
                                    ),
                                ],
                                inputs=example_inputs,
                                preprocess=False,
                            )

            with gr.Column(
                scale=13,
                min_width=700,
                elem_id="right-pane",
                elem_classes=["panel-card"]
            ):
                with gr.Column(elem_id="result-scroll"):
                    gr.Markdown("""
                    <div class="section-head">
                        <h3>Output</h3>
                        <p>Read the final answer first, then inspect the generated image and per-round execution trace.</p>
                    </div>
                    """)

                    final_answer = gr.Markdown(
                        value="### Result\nWaiting for execution...",
                        elem_classes=["answer-box"]
                    )

                    output_image = gr.Image(
                        type="filepath",
                        label="Output Image",
                        elem_classes=["gradio-image", "output-image-lg"]
                    )

                    execution_trace = gr.HTML(
                        value=render_trace_html([]),
                        elem_classes=["trace-shell"]
                    )

    submit_inputs = [user_input]
    for spec in INPUT_SPECS:
        submit_inputs.append(image_components[spec["key"]])
        submit_inputs.append(state_components[spec["key"]])

    def reset_ui():
        outputs = [""]
        for _ in INPUT_SPECS:
            outputs.extend([None, None])
        outputs.extend([
            render_trace_html([]),
            None,
            "### Result\nWaiting for execution...",
        ])
        return tuple(outputs)

    clear_outputs = [user_input]
    for spec in INPUT_SPECS:
        clear_outputs.append(image_components[spec["key"]])
        clear_outputs.append(state_components[spec["key"]])
    clear_outputs.extend([execution_trace, output_image, final_answer])

    submit_btn.click(
        run_agent,
        inputs=submit_inputs,
        outputs=[final_answer, execution_trace, output_image]
    )

    clear_btn.click(
        reset_ui,
        inputs=[],
        outputs=clear_outputs
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=4444,
        allowed_paths=["./"]
    )

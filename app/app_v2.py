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

PRETRAINED_PATH = "/home/ubuntu/00_CPK/OpenEarthAgent"
MAX_TOOL_ROUNDS = 16

INPUT_SPECS = [
    {
        "key": "primary_image",
        "label": "General Image / GeoTIFF",
        "description": "Single-scene image for detection, OCR, interpretation, or GeoTIFF overlay tasks.",
        "group": "General",
        "height": 260,
        "sample_aliases": {"S_10_preview.png": "./assets/S_10.tif"},
    },
    {
        "key": "lst_image",
        "label": "LST Raster",
        "description": "Land Surface Temperature raster used by TVDIAnalysis.",
        "group": "TVDI",
        "height": 220,
    },
    {
        "key": "ndvi_image",
        "label": "NDVI Raster",
        "description": "NDVI raster used by TVDIAnalysis.",
        "group": "TVDI",
        "height": 220,
    },
    {
        "key": "time1_image",
        "label": "Time-1 Image",
        "description": "Earlier timestamp image, often used as pre_image.",
        "group": "Time Series",
        "height": 220,
    },
    {
        "key": "time2_image",
        "label": "Time-2 Image",
        "description": "Later timestamp image, often used as post_image.",
        "group": "Time Series",
        "height": 220,
    },
]

INPUT_GROUPS = [
    {
        "tab": "General",
        "title": "Single-scene workspace",
        "note": "Upload one remote sensing image, photo, SAR preview, or GeoTIFF for detection, annotation, OCR, overlays, and general interpretation.",
        "input_keys": ["primary_image"],
    },
    {
        "tab": "TVDI",
        "title": "Dual-raster TVDI analysis",
        "note": "Upload both LST and NDVI rasters here. The runtime will bind them automatically to TVDIAnalysis(ndvi_path, lst_path, ...).",
        "input_keys": ["lst_image", "ndvi_image"],
    },
    {
        "tab": "Time Series",
        "title": "Bi-temporal comparison",
        "note": "Use paired inputs for pre/post change analysis. These files can be injected into tools such as ChangeDetection.",
        "input_keys": ["time1_image", "time2_image"],
    },
]

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
    "ChangeDetection": {"pre_image": "time1_image", "post_image": "time2_image"},
    "TVDIAnalysis": {"ndvi_path": "ndvi_image", "lst_path": "lst_image"},
    "SARToRGB": {"image": "primary_image"},
    "SARPreprocessing": {"image": "primary_image"},
}

TOOL_DEFAULT_ARGUMENTS = {
    "AddText": {"color": "green"},
    "TVDIAnalysis": {"output_path": "tvdi_result.tif"},
}

GPKG_REQUIRED_TOOLS = {
    "AddPoisLayer", "ComputeDistance", "DisplayOnMap", "AddIndexLayer",
    "AddDEMLayer", "ComputeIndexChange", "ShowIndexLayer", "DisplayOnGeotiff",
}


class LLM:
    def __init__(self, pretrained):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.system_prompt = RS_AGENT_PROMPT
        self.max_new_tokens = 256

    def prepend_system_prompt(self, conversation):
        if not conversation or conversation[0]["role"] != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def generate(self, conversation):
        conversation = self.prepend_system_prompt(conversation)
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def extract_actions(text: str):
    try:
        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None, "No action found."
        return json.loads(actions_match.group(1)), None
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


def resolve_existing_file(path):
    if not path:
        return None
    for candidate in [path, os.path.abspath(path), os.path.join(os.getcwd(), path)]:
        if os.path.isfile(candidate):
            return candidate
    return None


def make_preview_if_tif(path):
    if not path:
        return path
    if path.lower().endswith((".tif", ".tiff")):
        png_path = re.sub(r"\.(tif|tiff)$", "_preview.png", path, flags=re.IGNORECASE)
        with rasterio.open(path) as src:
            if src.count >= 3:
                data = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            else:
                data = src.read(1)
                if data.dtype != np.uint8:
                    data = np.nan_to_num(data)
                    data_min, data_max = np.nanmin(data), np.nanmax(data)
                    data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8) if data_max > data_min else np.zeros_like(data, dtype=np.uint8)
                data = np.stack([data] * 3, axis=-1)
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            Image.fromarray(data).save(png_path)
        return png_path
    return path


def handle_upload(image_path):
    if not image_path:
        return None, None
    resolved = resolve_existing_file(image_path) or image_path
    if resolved.lower().endswith((".tif", ".tiff")):
        return make_preview_if_tif(resolved), resolved
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
    for key in ["image", "output_path", "out_file", "path", "preview", "png"]:
        value = tool_resp.get(key)
        if isinstance(value, str):
            found = resolve_existing_file(value)
            if found and found.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                return found
    text = tool_resp.get("text", "")
    if isinstance(text, str):
        for match in reversed(re.findall(r"([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|tif|tiff))", text, re.I)):
            found = resolve_existing_file(match)
            if found:
                return found
    return None


def get_input_spec(key):
    for spec in INPUT_SPECS:
        if spec["key"] == key:
            return spec
    raise KeyError(f"Unknown input spec key: {key}")


def build_registry_from_args(flat_args):
    registry, idx = {}, 0
    for spec in INPUT_SPECS:
        registry[spec["key"]] = normalize_uploaded_input(flat_args[idx], flat_args[idx + 1], sample_aliases=spec.get("sample_aliases"))
        idx += 2
    return registry


def describe_registry_for_prompt(registry):
    return [f"- {spec['key']}: {spec['description']}" for spec in INPUT_SPECS if registry.get(spec["key"])]


def build_initial_user_message(user_question, registry):
    lines = [user_question.strip()]
    input_lines = describe_registry_for_prompt(registry)
    if input_lines:
        lines.extend(["", "Available uploaded inputs:", *input_lines, "Use the most appropriate tool and bind these uploaded inputs to the correct tool arguments."])
    return "\n".join(lines)


def choose_visual_image(registry):
    for key in ["primary_image", "time2_image", "time1_image", "lst_image", "ndvi_image"]:
        if registry.get(key):
            return registry[key]
    return None


def inject_runtime_arguments(api_name, api_args, registry, current_gpkg):
    merged = dict(api_args)
    for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
        if registry.get(registry_key):
            merged[tool_arg] = registry[registry_key]
    if api_name in GPKG_REQUIRED_TOOLS and current_gpkg:
        merged["gpkg"] = current_gpkg
    for k, v in TOOL_DEFAULT_ARGUMENTS.get(api_name, {}).items():
        if merged.get(k) in [None, ""] or k not in merged:
            merged[k] = v
    return merged


def update_registry_with_tool_response(registry, api_name, tool_resp):
    if api_name == "TVDIAnalysis":
        out = tool_resp.get("output_path") or detect_output_visual(tool_resp)
        if out:
            registry["tvdi_result"] = out
    visual = detect_output_visual(tool_resp)
    if visual:
        registry["latest_output"] = visual
    return registry


def make_example(question, **paths):
    row = [question]
    for spec in INPUT_SPECS:
        row.append(paths.get(spec["key"]))
    return row


def build_brand_panel_html():
    return f"""
    <div class="brand-panel">
        <div class="brand-shell">
            <div class="brand-mark">
                <img src="./assets/nwpu-logo.svg" alt="NWPU Logo" />
            </div>
            <div class="brand-copy">
                <div class="panel-kicker">Remote Sensing Workspace</div>
                <h1>WeavLight Agent</h1>
                <p>Structured uploads, explicit tool traces, and live preview for multi-image remote sensing tasks.</p>
            </div>
        </div>
        <div class="stat-row">
            <div class="stat-card">
                <span>Input Roles</span>
                <strong>{len(INPUT_SPECS)}</strong>
            </div>
            <div class="stat-card">
                <span>Max Rounds</span>
                <strong>{MAX_TOOL_ROUNDS}</strong>
            </div>
            <div class="stat-card">
                <span>Preview</span>
                <strong>GeoTIFF + PNG</strong>
            </div>
        </div>
    </div>
    """


def build_workflow_panel_html():
    return """
    <div class="workflow-panel">
        <div class="panel-kicker">Workflow</div>
        <div class="workflow-step">
            <span>1</span>
            <div>
                <strong>Select the right input mode</strong>
                <p>Choose General, TVDI, or Time Series based on the task type.</p>
            </div>
        </div>
        <div class="workflow-step">
            <span>2</span>
            <div>
                <strong>Describe the objective</strong>
                <p>Ask for analysis, visualization, distance estimation, or raster processing in plain language.</p>
            </div>
        </div>
        <div class="workflow-step">
            <span>3</span>
            <div>
                <strong>Inspect the execution trace</strong>
                <p>Each round logs the selected tool, injected arguments, observations, and final answer.</p>
            </div>
        </div>
    </div>
    """


def build_runtime_panel_html():
    return """
    <div class="runtime-panel">
        <div class="panel-kicker">Runtime Notes</div>
        <div class="runtime-item">
            <strong>Auto binding</strong>
            <p>Uploaded files are mapped into tool arguments from the registry, so users do not need to pass raw paths manually.</p>
        </div>
        <div class="runtime-item">
            <strong>GeoTIFF preview</strong>
            <p>Raster outputs are converted to lightweight preview PNGs for display while the original file remains downloadable.</p>
        </div>
        <div class="runtime-item">
            <strong>Round visibility</strong>
            <p>The center panel separates tool calls, observations, reasoning, and final answers for easier debugging.</p>
        </div>
    </div>
    """


def render_empty_chat_html():
    return """
    <div class="chat-area">
        <div class="empty-chat-card">
            <div class="panel-kicker">Mission Console</div>
            <h3>Ready for a new remote sensing task</h3>
            <p>Upload the required inputs, describe the objective, and this panel will stream the agent trace round by round.</p>
            <div class="placeholder-grid">
                <div class="placeholder-step">
                    <span>01</span>
                    <strong>Choose inputs</strong>
                    <p>General image, TVDI raster pair, or time-series comparison.</p>
                </div>
                <div class="placeholder-step">
                    <span>02</span>
                    <strong>Run the agent</strong>
                    <p>The model selects tools and injects uploaded assets into the correct arguments.</p>
                </div>
                <div class="placeholder-step">
                    <span>03</span>
                    <strong>Review outputs</strong>
                    <p>Check the live preview on the right and download the latest generated file.</p>
                </div>
            </div>
        </div>
    </div>
    """


def classify_chat_message(role, content):
    if role == "user":
        return "user-msg", "Request", "badge-user"
    if "✅ Final" in content or "\nFinal" in content:
        return "assistant-msg final-msg", "Final", "badge-final"
    if "🛠️ Action:" in content:
        return "assistant-msg action-msg", "Tool Call", "badge-action"
    if "Observation" in content:
        return "assistant-msg observation-msg", "Observation", "badge-observation"
    if "Thinking..." in content:
        return "assistant-msg thinking-msg", "Reasoning", "badge-thinking"
    if "达到最大轮数" in content:
        return "assistant-msg alert-msg", "Stopped", "badge-alert"
    return "assistant-msg", "Agent", "badge-agent"


def split_message_header(content):
    first_line, _, remainder = content.partition("\n")
    heading = first_line.strip() or "Agent update"
    round_label = ""
    round_match = re.match(r"\[(Round \d+)\]\s*(.*)", heading)
    if round_match:
        round_label = round_match.group(1)
        heading = round_match.group(2) or "Agent update"
    return round_label, heading, remainder.strip()


def render_chat_html(messages):
    if not messages:
        return render_empty_chat_html()
    blocks = ['<div class="chat-area">']
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        bubble_class, badge_label, badge_class = classify_chat_message(role, content)
        if role == "user":
            body = html.escape(content)
            blocks.append(
                f'<div class="{bubble_class}"><div class="msg-meta"><span class="msg-badge {badge_class}">{badge_label}</span>'
                f'<span class="msg-heading">Task</span></div><pre>{body}</pre></div>'
            )
            continue

        round_label, heading, body = split_message_header(content)
        meta_items = []
        if round_label:
            meta_items.append(f'<span class="round-pill">{html.escape(round_label)}</span>')
        meta_items.append(f'<span class="msg-badge {badge_class}">{badge_label}</span>')
        meta_items.append(f'<span class="msg-heading">{html.escape(heading)}</span>')
        body_html = f"<pre>{html.escape(body)}</pre>" if body else ""
        blocks.append(f'<div class="{bubble_class}"><div class="msg-meta">{"".join(meta_items)}</div>{body_html}</div>')
    blocks.append("</div>")
    return "".join(blocks)


Model = LLM(PRETRAINED_PATH)
tool_manager = ToolManager()


def run_agent(user_question, *flat_args):
    registry = build_registry_from_args(flat_args)
    current_visual_image = choose_visual_image(registry)
    current_gpkg = None
    chat_msgs = [{"role": "user", "content": user_question}]

    initial_message = build_initial_user_message(user_question, registry)
    conversation = [{"role": "user", "content": initial_message}]

    def current_preview():
        latest = registry.get("latest_output") or registry.get("tvdi_result") or current_visual_image
        return make_preview_if_tif(latest) if latest and os.path.isfile(latest) else None

    def emit(download_path=None):
        return render_chat_html(chat_msgs), current_preview(), download_path

    for round_id in range(1, MAX_TOOL_ROUNDS + 1):
        model_output = Model.generate(conversation)
        thought, actions = extract_thought_and_actions(model_output)

        if not actions:
            chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] Thinking...\n{thought or model_output}"})
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0]
        api_name = action["name"]
        api_args = dict(action["arguments"])

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "") or "已完成，结果请查看右侧图像或左侧下载文件。"
            chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] ✅ Final\n{final_answer}"})
            download = registry.get("latest_output") or registry.get("tvdi_result")
            yield emit(download)
            return

        injected_args = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)
        chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] 🛠️ Action: {api_name}\nArgs:\n{json.dumps(injected_args, ensure_ascii=False, indent=2)}"})
        yield emit(None)

        tool_resp = tool_manager.call_tool(api_name, injected_args)
        if "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")

        registry = update_registry_with_tool_response(registry, api_name, tool_resp)
        status = "✅" if tool_resp.get("error_code") == 0 else "❌"
        obs_text = tool_resp.get("text", "")
        chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] {status} Observation\n{obs_text}"})
        download = registry.get("latest_output") or registry.get("tvdi_result")
        yield emit(download)

        conversation.append({"role": "assistant", "content": model_output})
        conversation.append({"role": "user", "content": f"OBSERVATION:\n{api_name} outputs: {obs_text}\nPlease summarize and answer the question."})

    chat_msgs.append({"role": "assistant", "content": "达到最大轮数，已停止。"})
    yield emit(registry.get("latest_output") or registry.get("tvdi_result"))


css = """
:root {
    --bg-top: #f7f1e5;
    --bg-bottom: #edf4ef;
    --surface: rgba(255, 253, 249, 0.94);
    --surface-strong: #fffdfa;
    --surface-soft: #eef5f2;
    --line: #d7e1db;
    --line-strong: #b8c8c0;
    --text: #173042;
    --muted: #5f717d;
    --accent: #0f766e;
    --accent-deep: #0a5a54;
    --accent-soft: #dff1eb;
    --highlight: #c96d3d;
    --highlight-soft: #f5e4d9;
    --shadow: 0 18px 42px rgba(17, 42, 58, 0.10);
    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;
}

html, body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 26%),
        radial-gradient(circle at bottom right, rgba(201, 109, 61, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%) !important;
    color: var(--text) !important;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
}

.gradio-container {
    max-width: 1680px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

.workspace-grid {
    gap: 18px;
    align-items: stretch;
}

.left-rail, .center-rail, .right-rail {
    gap: 18px;
}

.panel-card {
    background: linear-gradient(180deg, rgba(255, 253, 250, 0.96) 0%, rgba(255, 255, 255, 0.92) 100%) !important;
    border: 1px solid rgba(184, 200, 192, 0.85) !important;
    border-radius: var(--radius-xl) !important;
    padding: 18px !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}

.brand-card {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.brand-panel {
    background: linear-gradient(135deg, rgba(255, 253, 250, 0.98) 0%, rgba(239, 247, 243, 0.98) 52%, rgba(245, 228, 217, 0.95) 100%);
    border: 1px solid rgba(184, 200, 192, 0.9);
    border-radius: 30px;
    padding: 22px;
    box-shadow: var(--shadow);
}

.brand-shell {
    display: flex;
    gap: 16px;
    align-items: center;
}

.brand-mark {
    width: 76px;
    height: 76px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(184, 200, 192, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.brand-mark img {
    width: 54px;
    height: 54px;
    object-fit: contain;
}

.brand-copy h1,
.empty-chat-card h3,
.section-head h3 {
    margin: 0;
    color: var(--text);
    font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
}

.brand-copy p,
.section-head p,
.mode-note p,
.runtime-item p,
.workflow-step p,
.result-note,
.empty-chat-card p,
.placeholder-step p {
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
}

.panel-kicker {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(23, 48, 66, 0.08);
    color: var(--text);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.stat-card {
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(184, 200, 192, 0.8);
}

.stat-card span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 6px;
}

.stat-card strong {
    color: var(--text);
    font-size: 16px;
    font-weight: 700;
}

.workflow-panel,
.runtime-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.workflow-step,
.runtime-item {
    display: flex;
    gap: 12px;
    padding: 14px 16px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(245, 249, 247, 0.96) 100%);
    border: 1px solid rgba(184, 200, 192, 0.75);
}

.workflow-step span {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--accent-soft);
    color: var(--accent-deep);
    font-size: 13px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.workflow-step strong,
.runtime-item strong,
.mode-note strong,
.placeholder-step strong {
    display: block;
    color: var(--text);
    margin-bottom: 4px;
}

.section-head {
    margin-bottom: 14px;
}

.section-head h3 {
    font-size: 24px;
    margin-bottom: 6px;
}

.mode-tabs button {
    border-radius: 999px !important;
    font-weight: 700 !important;
}

.mode-tabs button[aria-selected="true"] {
    background: var(--text) !important;
    color: #ffffff !important;
}

.mode-note {
    margin-bottom: 14px;
    padding: 14px 16px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(223, 241, 235, 0.7) 0%, rgba(255, 255, 255, 0.82) 100%);
    border: 1px solid rgba(184, 200, 192, 0.78);
}

.upload-slot,
.result-view,
.download-slot {
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 1px solid rgba(184, 200, 192, 0.9) !important;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(243, 248, 245, 0.95) 100%) !important;
}

.question-box textarea,
.question-box input {
    border-radius: 18px !important;
    border: 1px solid rgba(184, 200, 192, 0.9) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    font-size: 15px !important;
}

.question-box textarea:focus,
.question-box input:focus {
    border-color: rgba(15, 118, 110, 0.9) !important;
    box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12) !important;
}

.button-stack {
    gap: 10px;
}

.submit-btn button,
.clear-btn button {
    min-height: 48px !important;
    border-radius: 16px !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.submit-btn button {
    background: linear-gradient(135deg, var(--text) 0%, #23465c 100%) !important;
    color: #ffffff !important;
}

.clear-btn button {
    background: rgba(255, 255, 255, 0.9) !important;
    color: var(--text) !important;
    border: 1px solid rgba(184, 200, 192, 0.9) !important;
}

.chat-shell {
    min-height: 560px;
    max-height: 70vh;
    overflow: auto;
    border: 1px solid rgba(184, 200, 192, 0.85);
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(253, 251, 247, 0.94) 0%, rgba(244, 248, 246, 0.96) 100%);
}

.chat-area {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 16px;
}

.assistant-msg,
.user-msg {
    max-width: 92%;
    border-radius: 20px;
    padding: 14px 16px;
    border: 1px solid rgba(184, 200, 192, 0.9);
    box-shadow: 0 10px 24px rgba(17, 42, 58, 0.06);
}

.assistant-msg {
    background: rgba(255, 255, 255, 0.92);
}

.user-msg {
    margin-left: auto;
    background: linear-gradient(135deg, rgba(15, 118, 110, 0.12) 0%, rgba(255, 255, 255, 0.94) 100%);
    border-color: rgba(15, 118, 110, 0.25);
}

.action-msg {
    border-left: 4px solid var(--highlight);
    background: linear-gradient(180deg, rgba(245, 228, 217, 0.55) 0%, rgba(255, 255, 255, 0.95) 100%);
}

.observation-msg {
    border-left: 4px solid var(--accent);
    background: linear-gradient(180deg, rgba(223, 241, 235, 0.60) 0%, rgba(255, 255, 255, 0.95) 100%);
}

.final-msg {
    border-left: 4px solid var(--text);
    background: linear-gradient(180deg, rgba(227, 237, 244, 0.72) 0%, rgba(255, 255, 255, 0.96) 100%);
}

.thinking-msg,
.alert-msg {
    background: linear-gradient(180deg, rgba(248, 244, 237, 0.96) 0%, rgba(255, 255, 255, 0.96) 100%);
}

.msg-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.round-pill,
.msg-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.round-pill {
    background: rgba(23, 48, 66, 0.08);
    color: var(--text);
}

.msg-badge {
    color: var(--text);
    background: rgba(23, 48, 66, 0.08);
}

.badge-user {
    background: rgba(15, 118, 110, 0.14);
    color: var(--accent-deep);
}

.badge-action {
    background: rgba(201, 109, 61, 0.14);
    color: var(--highlight);
}

.badge-observation {
    background: rgba(15, 118, 110, 0.14);
    color: var(--accent-deep);
}

.badge-final {
    background: rgba(23, 48, 66, 0.12);
    color: var(--text);
}

.badge-thinking,
.badge-agent,
.badge-alert {
    background: rgba(95, 113, 125, 0.14);
    color: var(--muted);
}

.msg-heading {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
}

.assistant-msg pre,
.user-msg pre {
    margin: 0;
    white-space: pre-wrap;
    color: var(--text);
    line-height: 1.6;
    font-size: 12.5px;
    font-family: "IBM Plex Mono", "Consolas", monospace;
}

.empty-chat-card {
    padding: 24px;
    border-radius: 24px;
    border: 1px dashed rgba(184, 200, 192, 0.95);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(239, 247, 243, 0.92) 60%, rgba(245, 228, 217, 0.86) 100%);
}

.empty-chat-card h3 {
    margin-bottom: 8px;
}

.placeholder-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.placeholder-step {
    padding: 16px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(184, 200, 192, 0.8);
}

.placeholder-step span {
    display: inline-flex;
    margin-bottom: 10px;
    color: var(--highlight);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.08em;
}

.gallery-note {
    color: var(--muted);
    margin-bottom: 10px;
}

.example-accordion {
    border-radius: 18px !important;
    border: 1px solid rgba(184, 200, 192, 0.85) !important;
    background: rgba(255, 255, 255, 0.78) !important;
}

.result-note {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(223, 241, 235, 0.55);
    border: 1px solid rgba(184, 200, 192, 0.78);
    font-size: 13px;
}

@media (max-width: 1280px) {
    .stat-row,
    .placeholder-grid {
        grid-template-columns: 1fr;
    }

    .chat-shell {
        min-height: 460px;
        max-height: 60vh;
    }
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 12px !important;
    }

    .brand-shell {
        align-items: flex-start;
    }

    .chat-shell {
        min-height: 360px;
        max-height: none;
    }
}
"""


def reset_ui():
    outputs = ["", render_chat_html([]), None, None]
    for _ in INPUT_SPECS:
        outputs.extend([None, None])
    return tuple(outputs)


with gr.Blocks(title="WeavLight Workspace", css=css) as demo:
    state_components, image_components = {}, {}

    with gr.Row(elem_classes=["workspace-grid"]):
        with gr.Column(scale=4, min_width=360, elem_classes=["left-rail"]):
            gr.HTML(build_brand_panel_html(), elem_classes=["brand-card"])
            gr.HTML(build_workflow_panel_html(), elem_classes=["panel-card"])

            with gr.Column(elem_classes=["panel-card", "upload-panel"]):
                gr.HTML(
                    """
                    <div class="section-head">
                        <div class="panel-kicker">Inputs</div>
                        <h3>Task-specific upload slots</h3>
                        <p>Choose the right input group so the runtime can bind uploaded files to the correct tool arguments automatically.</p>
                    </div>
                    """
                )

                with gr.Tabs(elem_classes=["mode-tabs"]):
                    for group in INPUT_GROUPS:
                        with gr.Tab(group["tab"]):
                            gr.HTML(f"<div class='mode-note'><strong>{group['title']}</strong><p>{group['note']}</p></div>")
                            if len(group["input_keys"]) == 1:
                                spec = get_input_spec(group["input_keys"][0])
                                image_components[spec["key"]] = gr.Image(
                                    type="filepath",
                                    label=spec["label"],
                                    height=spec["height"],
                                    sources=["upload", "clipboard"],
                                    elem_classes=["upload-slot"],
                                )
                                state_components[spec["key"]] = gr.State()
                            else:
                                with gr.Row():
                                    for key in group["input_keys"]:
                                        spec = get_input_spec(key)
                                        image_components[spec["key"]] = gr.Image(
                                            type="filepath",
                                            label=spec["label"],
                                            height=spec["height"],
                                            sources=["upload", "clipboard"],
                                            elem_classes=["upload-slot"],
                                        )
                                        state_components[spec["key"]] = gr.State()

                gr.HTML(
                    "<div class='result-note'>GeoTIFF files are previewed automatically, but the original raster path is preserved for downstream tools.</div>"
                )

            for spec in INPUT_SPECS:
                image_components[spec["key"]].change(
                    fn=handle_upload,
                    inputs=image_components[spec["key"]],
                    outputs=[image_components[spec["key"]], state_components[spec["key"]]],
                    trigger_mode="once",
                )

        with gr.Column(scale=6, min_width=660, elem_classes=["center-rail"]):
            with gr.Column(elem_classes=["panel-card", "conversation-panel"]):
                gr.HTML(
                    """
                    <div class="section-head">
                        <div class="panel-kicker">Conversation</div>
                        <h3>Ask, inspect, iterate</h3>
                        <p>The chat surface below shows the selected tools, injected arguments, returned observations, and final answer in chronological order.</p>
                    </div>
                    """
                )
                chat_html = gr.HTML(value=render_chat_html([]), elem_classes=["chat-shell"])

                with gr.Row(elem_classes=["composer-row"]):
                    user_input = gr.Textbox(
                        label="Task Prompt",
                        placeholder="描述分析目标、目标区域、时间范围或对象，例如：比较两期影像的变化并标注主要区域。",
                        lines=4,
                        elem_classes=["question-box"],
                        scale=8,
                    )
                    with gr.Column(scale=2, min_width=160, elem_classes=["button-stack"]):
                        submit_btn = gr.Button("Run Agent", elem_classes=["submit-btn"])
                        clear_btn = gr.Button("Reset Workspace", elem_classes=["clear-btn"])

            with gr.Column(elem_classes=["panel-card", "gallery-panel"]):
                gr.HTML(
                    """
                    <div class="section-head">
                        <div class="panel-kicker">Templates</div>
                        <h3>Prompt gallery</h3>
                        <p>Load a starter prompt and the matching sample assets, then refine the request in the prompt box.</p>
                    </div>
                    """
                )
                with gr.Accordion("Browse starter prompts", open=False, elem_classes=["example-accordion"]):
                    gr.Markdown("Selecting an example will auto-fill the prompt and the corresponding upload components.")
                    example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]
                    with gr.Tabs():
                        with gr.Tab("General / Index"):
                            gr.Examples(
                                examples=[
                                    make_example("Generate a preview from the NBR difference for Topanga State Park, Los Angeles, USA, covering December 2024 and February 2025."),
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
                                    make_example("For Manchester State Forest, South Carolina, United States, create a DEM layer from GEE, generate 20 m contours and 100 m elevation bands, and summarize the elevation range."),
                                    make_example("For the area within a 1000 m radius of Edinburgh Castle, generate a DEM layer, create contour lines, and visualize the resulting terrain bands on the map."),
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
                                    make_example("Compare the two temporal images and identify the major changes."),
                                ],
                                inputs=example_inputs,
                                preprocess=False,
                            )

        with gr.Column(scale=4, min_width=340, elem_classes=["right-rail"]):
            with gr.Column(elem_classes=["panel-card", "preview-panel"]):
                gr.HTML(
                    """
                    <div class="section-head">
                        <div class="panel-kicker">Outputs</div>
                        <h3>Live preview and download</h3>
                        <p>The preview refreshes whenever a tool returns an image or raster. Download always points to the latest generated asset.</p>
                    </div>
                    """
                )
                output_image = gr.Image(
                    type="filepath",
                    label="Live Preview",
                    height=360,
                    elem_classes=["result-view"],
                )
                download_file = gr.File(label="Download Latest Output", elem_classes=["download-slot"])
                gr.HTML(
                    "<div class='result-note'>If the tool output is a GeoTIFF, the right panel shows a PNG preview while keeping the original file available for download.</div>"
                )

            gr.HTML(build_runtime_panel_html(), elem_classes=["panel-card", "tips-card"])

    submit_inputs = [user_input]
    for spec in INPUT_SPECS:
        submit_inputs.extend([image_components[spec["key"]], state_components[spec["key"]]])

    submit_btn.click(run_agent, inputs=submit_inputs, outputs=[chat_html, output_image, download_file])

    clear_outputs = [user_input, chat_html, output_image, download_file]
    for spec in INPUT_SPECS:
        clear_outputs.extend([image_components[spec["key"]], state_components[spec["key"]]])
    clear_btn.click(reset_ui, inputs=[], outputs=clear_outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=4444, allowed_paths=["./"])

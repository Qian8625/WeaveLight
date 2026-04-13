import gradio as gr
import json
import re
import torch
import os
import html
from PIL import Image
from datetime import datetime
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
        return render_chat_html(chat_msgs), current_preview(), download_path, download_path

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
    --c-deep: #355872;
    --c-mid: #7AAACE;
    --c-light: #9CD5FF;
    --c-bg: #F7F8F0;
    --c-border: #d6deea;
    --c-text: #132a3f;
    --c-muted: #58738a;
    --card: #ffffff;
    --shadow: 0 10px 28px rgba(53, 88, 114, 0.10);
}

html, body, .gradio-container {
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: "Inter", "IBM Plex Sans", "Segoe UI", sans-serif !important;
}

.gradio-container {
    max-width: 100vw !important;
    padding: 92px 16px 16px !important;
}

.topbar-card {
    position: fixed;
    top: 12px;
    left: 16px;
    right: 16px;
    z-index: 60;
    background: linear-gradient(120deg, var(--c-deep), #426a8c 50%, var(--c-mid));
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #fff;
    border-radius: 16px;
    box-shadow: var(--shadow);
    margin-bottom: 12px;
}

.topbar-inner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
}

.topbar-logo {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.35);
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.topbar-logo img { width: 26px; height: 26px; object-fit: contain; }
.topbar-title h1 { margin: 0; font-size: 20px; color: #fff; }
.topbar-title p { margin: 2px 0 0; font-size: 12px; color: rgba(255,255,255,.88); }

.workspace-grid {
    gap: 12px;
    align-items: stretch;
    height: calc(100vh - 108px);
}

.left-rail,
.right-rail {
    height: 100%;
    overflow: hidden;
}

.center-rail {
    height: 100%;
    min-width: 760px;
}

.fixed-column {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.scroll-y { overflow-y: auto; }

.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow) !important;
    padding: 12px !important;
    overflow: hidden;
}

.section-head h3 { margin: 0; color: var(--c-text); }
.section-head p { margin: 6px 0 0; color: var(--c-muted); font-size: 13px; line-height: 1.45; }
.section-head .panel-kicker { font-size: 11px; color: var(--c-deep); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }

.mode-tabs button[aria-selected="true"] { background: var(--c-deep) !important; color: #fff !important; }
.mode-note { background: #f8fbff; border: 1px solid #e4edf8; border-radius: 12px; padding: 10px; margin-bottom: 10px; }
.mode-note p { margin: 6px 0 0; color: var(--c-muted); font-size: 12px; }

.upload-slot,
.result-view,
.download-slot {
    border-radius: 12px !important;
    border: 1px solid #dde7f3 !important;
    background: #fbfdff !important;
}

.chat-shell {
    height: calc(100% - 132px);
    overflow: auto;
    border: 1px solid var(--c-border);
    border-radius: 14px;
    background: #fff;
}

.chat-area { display: flex; flex-direction: column; gap: 10px; padding: 14px; }

.user-msg, .assistant-msg { max-width: 88%; border-radius: 14px; padding: 12px; border: 1px solid #dbe5f1; }
.user-msg { margin-left: auto; background: #edf6ff; border-color: #cbe3fb; }
.assistant-msg { background: #f9fbff; }
.action-msg { background: #eef7ff; border-color: #cfe5fb; }
.observation-msg { background: #eefaf7; border-color: #cde9df; }
.final-msg { background: #f0f6fb; border-color: #d4e1ef; }

.msg-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.round-pill, .msg-badge { border-radius: 999px; padding: 3px 10px; font-size: 10px; font-weight: 700; background: #e9f1fa; color: var(--c-deep); }
.msg-heading { font-size: 13px; font-weight: 700; }
.assistant-msg pre, .user-msg pre { margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.55; }

.composer-row { height: 120px; margin-top: 10px; }
.question-box textarea { height: 104px !important; border-radius: 12px !important; border: 1px solid #cbd8e8 !important; }
.submit-btn button, .clear-btn button { border-radius: 12px !important; min-height: 44px !important; font-weight: 700 !important; }
.submit-btn button { background: var(--c-deep) !important; color: #fff !important; }
.clear-btn button { background: #fff !important; border: 1px solid #c8d7e7 !important; color: var(--c-text) !important; }

.example-panel { flex: 0 0 42%; }
.preview-panel { flex: 1; }
.input-panel { flex: 1; }
.bulk-download-panel { flex: 0 0 36%; }

@media (max-width: 1400px) {
    .workspace-grid { height: auto; }
    .left-rail, .center-rail, .right-rail { height: auto; }
}
"""


def reset_ui():
    outputs = ["", render_chat_html([]), None, None, None]
    for _ in INPUT_SPECS:
        outputs.extend([None, None])
    return tuple(outputs)


with gr.Blocks(title="WeavLight Workspace", css=css) as demo:
    state_components, image_components = {}, {}

    gr.HTML(
        """
        <div class="topbar-card">
            <div class="topbar-inner">
                <div class="topbar-logo"><img src="./assets/nwpu-logo.svg" alt="logo"/></div>
                <div class="topbar-title">
                    <h1>WeavLight · AI Agent Imagery Workspace</h1>
                    <p>Multi-modal remote-sensing analysis cockpit · desktop optimized</p>
                </div>
            </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["workspace-grid"]):
        with gr.Column(scale=3, min_width=330, elem_classes=["left-rail", "fixed-column"]):
            with gr.Column(elem_classes=["panel-card", "input-panel", "scroll-y"]):
                gr.HTML("<div class='section-head'><div class='panel-kicker'>Left</div><h3>Image upload inputs</h3><p>Upload task imagery here. This area is fixed-width for desktop workflow stability.</p></div>")
                with gr.Tabs(elem_classes=["mode-tabs"]):
                    for group in INPUT_GROUPS:
                        with gr.Tab(group["tab"]):
                            gr.HTML(f"<div class='mode-note'><strong>{group['title']}</strong><p>{group['note']}</p></div>")
                            if len(group["input_keys"]) == 1:
                                spec = get_input_spec(group["input_keys"][0])
                                image_components[spec["key"]] = gr.Image(type="filepath", label=spec["label"], height=spec["height"], sources=["upload", "clipboard"], elem_classes=["upload-slot"])
                                state_components[spec["key"]] = gr.State()
                            else:
                                with gr.Row():
                                    for key in group["input_keys"]:
                                        spec = get_input_spec(key)
                                        image_components[spec["key"]] = gr.Image(type="filepath", label=spec["label"], height=spec["height"], sources=["upload", "clipboard"], elem_classes=["upload-slot"])
                                        state_components[spec["key"]] = gr.State()

            with gr.Column(elem_classes=["panel-card", "bulk-download-panel"]):
                gr.HTML("<div class='section-head'><div class='panel-kicker'>Left</div><h3>Batch output download</h3><p>Download current generated assets in one place to close the operation loop.</p></div>")
                download_file = gr.File(label="Batch Download / Latest Output", elem_classes=["download-slot"])

            for spec in INPUT_SPECS:
                image_components[spec["key"]].change(
                    fn=handle_upload,
                    inputs=image_components[spec["key"]],
                    outputs=[image_components[spec["key"]], state_components[spec["key"]]],
                    trigger_mode="once",
                )

        with gr.Column(scale=6, min_width=760, elem_classes=["center-rail", "fixed-column"]):
            with gr.Column(elem_classes=["panel-card"], scale=1):
                gr.HTML("<div class='section-head'><div class='panel-kicker'>Center</div><h3>Agent Q&A process log</h3><p>The chat stream shows each tool action, observation, and final answer step-by-step.</p></div>")
                chat_html = gr.HTML(value=render_chat_html([]), elem_classes=["chat-shell"])
                with gr.Row(elem_classes=["composer-row"]):
                    user_input = gr.Textbox(label="", placeholder="Type your task for the AI Agent...", lines=4, elem_classes=["question-box"], scale=8)
                    with gr.Column(scale=2, min_width=160):
                        submit_btn = gr.Button("Send", elem_classes=["submit-btn"])
                        clear_btn = gr.Button("Clear", elem_classes=["clear-btn"])

        with gr.Column(scale=3, min_width=330, elem_classes=["right-rail", "fixed-column"]):
            with gr.Column(elem_classes=["panel-card", "example-panel", "scroll-y"]):
                gr.HTML("<div class='section-head'><div class='panel-kicker'>Right</div><h3>Example templates</h3><p>Open or collapse examples to quickly bootstrap prompts and inputs.</p></div>")
                with gr.Accordion("Open / Hide Examples", open=False, elem_classes=["example-accordion"]):
                    example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]
                    with gr.Tabs():
                        with gr.Tab("General / Index"):
                            gr.Examples(examples=[
                                make_example("Generate a preview from the NBR difference for Topanga State Park, Los Angeles, USA, covering December 2024 and February 2025."),
                                make_example("Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", primary_image="./assets/S_10_preview.png"),
                                make_example("Locate and estimate the distance between aircrafts in the scene. Assuming GSD 0.6 px/meter", primary_image="./assets/TG_P0009.png"),
                            ], inputs=example_inputs, preprocess=False)
                        with gr.Tab("Terrain / DEM"):
                            gr.Examples(examples=[
                                make_example("For Manchester State Forest, South Carolina, United States, create a DEM layer from GEE, generate 20 m contours and 100 m elevation bands, and summarize the elevation range."),
                                make_example("For the area within a 1000 m radius of Edinburgh Castle, generate a DEM layer, create contour lines, and visualize the resulting terrain bands on the map."),
                            ], inputs=example_inputs, preprocess=False)
                        with gr.Tab("TVDI"):
                            gr.Examples(examples=[
                                make_example("Compute the Temperature Vegetation Dryness Index (TVDI) from NDVI and LST rasters.", lst_image="./assets/Sichuan_2021-07-12_LST.tif", ndvi_image="./assets/Sichuan_2021-07-12_NDVI.tif"),
                            ], inputs=example_inputs, preprocess=False)
                        with gr.Tab("Time Series"):
                            gr.Examples(examples=[make_example("Compare the two temporal images and identify the major changes.")], inputs=example_inputs, preprocess=False)

            with gr.Column(elem_classes=["panel-card", "preview-panel"]):
                gr.HTML("<div class='section-head'><div class='panel-kicker'>Right</div><h3>HD preview + single download</h3><p>Inspect the latest visual result with high clarity before downloading.</p></div>")
                output_image = gr.Image(type="filepath", label="Output Preview", height=340, elem_classes=["result-view"])
                single_download_file = gr.File(label="Single Output Download", elem_classes=["download-slot"])


    submit_inputs = [user_input]
    for spec in INPUT_SPECS:
        submit_inputs.extend([image_components[spec["key"]], state_components[spec["key"]]])

    submit_btn.click(run_agent, inputs=submit_inputs, outputs=[chat_html, output_image, download_file, single_download_file])

    clear_outputs = [user_input, chat_html, output_image, download_file, single_download_file]
    for spec in INPUT_SPECS:
        clear_outputs.extend([image_components[spec["key"]], state_components[spec["key"]]])
    clear_btn.click(reset_ui, inputs=[], outputs=clear_outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=4444, allowed_paths=["./"])
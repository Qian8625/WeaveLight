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
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnMap",
    "AddIndexLayer",
    "AddDEMLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "DisplayOnGeotiff",
}


class LLM:
    def __init__(self, pretrained):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, use_fast=True, trust_remote_code=True
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
            conversation, tokenize=False, add_generation_prompt=True
        )
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
                    if data_max > data_min:
                        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        data = np.zeros_like(data, dtype=np.uint8)
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
        for match in reversed(
            re.findall(r"([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|tif|tiff))", text, re.I)
        ):
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
        registry[spec["key"]] = normalize_uploaded_input(
            flat_args[idx], flat_args[idx + 1], sample_aliases=spec.get("sample_aliases")
        )
        idx += 2
    return registry


def describe_registry_for_prompt(registry):
    return [
        f"- {spec['key']}: {spec['description']}"
        for spec in INPUT_SPECS
        if registry.get(spec["key"])
    ]


def build_initial_user_message(user_question, registry):
    lines = [user_question.strip()]
    input_lines = describe_registry_for_prompt(registry)
    if input_lines:
        lines.extend(
            [
                "",
                "Available uploaded inputs:",
                *input_lines,
                "Use the most appropriate tool and bind these uploaded inputs to the correct tool arguments.",
            ]
        )
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
                    <p>Check the live preview and download the latest generated file.</p>
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
                f'<div class="{bubble_class}"><div class="msg-meta">'
                f'<span class="msg-badge {badge_class}">{badge_label}</span>'
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
        blocks.append(
            f'<div class="{bubble_class}"><div class="msg-meta">{"".join(meta_items)}</div>{body_html}</div>'
        )
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
            chat_msgs.append(
                {"role": "assistant", "content": f"[Round {round_id}] Thinking...\n{thought or model_output}"}
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0]
        api_name = action["name"]
        api_args = dict(action["arguments"])

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "") or "已完成，结果请查看预览图像或下载文件。"
            chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] ✅ Final\n{final_answer}"})
            download = registry.get("latest_output") or registry.get("tvdi_result")
            yield emit(download)
            return

        injected_args = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)
        chat_msgs.append(
            {
                "role": "assistant",
                "content": f"[Round {round_id}] 🛠️ Action: {api_name}\nArgs:\n{json.dumps(injected_args, ensure_ascii=False, indent=2)}",
            }
        )
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
        conversation.append(
            {
                "role": "user",
                "content": f"OBSERVATION:\n{api_name} outputs: {obs_text}\nPlease summarize and answer the question.",
            }
        )

    chat_msgs.append({"role": "assistant", "content": "达到最大轮数，已停止。"})
    yield emit(registry.get("latest_output") or registry.get("tvdi_result"))


UPLOAD_HINT_HTML = """
<div class="hint-strip">
    <span>支持拖拽 / 点击 / 粘贴上传</span>
    <span>格式：PNG / JPG / JPEG / TIF / TIFF</span>
    <span>建议单文件 ≤ 200MB</span>
</div>
"""


def render_upload_registry_html(registry):
    registry = registry or {}
    blocks = ['<div class="asset-list">']
    populated = False

    for spec in INPUT_SPECS:
        item = registry.get(spec["key"])
        if not item:
            continue
        populated = True
        preview = html.escape(item.get("preview") or "")
        label = html.escape(item.get("label") or spec["label"])
        path = html.escape(item.get("path") or "")
        blocks.append(
            f"""
            <div class="asset-item">
                <div class="asset-thumb-wrap">
                    <img class="asset-thumb" src="/gradio_api/file={preview}" alt="{label}" />
                </div>
                <div class="asset-meta">
                    <div class="asset-title">{label}</div>
                    <div class="asset-sub">已绑定输入</div>
                    <div class="asset-path">{path}</div>
                </div>
            </div>
            """
        )

    if not populated:
        blocks.append(
            """
            <div class="empty-mini-card">
                <strong>暂无已绑定输入</strong>
                <p>上传图片后，这里会显示缩略图、绑定角色和文件路径。</p>
            </div>
            """
        )

    blocks.append("</div>")
    return "".join(blocks)


def render_output_history_html(history):
    history = history or []
    if not history:
        return """
        <div class="history-list empty-state">
            <strong>暂无输出历史</strong>
            <p>Agent 生成结果后，会按时间倒序显示在这里。</p>
        </div>
        """

    blocks = ['<div class="history-list">']
    for idx, item in enumerate(history, start=1):
        preview = html.escape(item.get("preview") or "")
        name = html.escape(item.get("name") or f"output_{idx}")
        created_at = html.escape(item.get("created_at") or "--")
        source = html.escape(item.get("source") or "Agent")
        status = html.escape(item.get("status") or "ready")
        path = html.escape(item.get("path") or "")
        blocks.append(
            f"""
            <div class="history-item">
                <div class="history-thumb-wrap">
                    <img class="history-thumb" src="/gradio_api/file={preview}" alt="{name}" />
                </div>
                <div class="history-meta">
                    <div class="history-title-row">
                        <strong>{name}</strong>
                        <span class="history-status status-{status}">{status}</span>
                    </div>
                    <div class="history-sub">{source} · {created_at}</div>
                    <div class="history-path">{path}</div>
                </div>
            </div>
            """
        )
    blocks.append("</div>")
    return "".join(blocks)


def append_output_history(history, path, source="Agent"):
    if not path:
        return history or []

    resolved = resolve_existing_file(path)
    if not resolved:
        return history or []

    history = list(history or [])
    if any(item.get("path") == resolved for item in history):
        return history

    preview = make_preview_if_tif(resolved)
    history.insert(
        0,
        {
            "name": os.path.basename(resolved),
            "path": resolved,
            "preview": preview if resolve_existing_file(preview) else resolved,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "status": "success",
        },
    )
    return history


def build_batch_files(history):
    files = []
    for item in history or []:
        path = resolve_existing_file(item.get("path"))
        if path:
            files.append(path)
    return files


def clear_output_history():
    return render_output_history_html([]), [], None, None, []


def make_upload_change_handler(spec_key):
    spec = get_input_spec(spec_key)

    def _handler(image_path, upload_registry):
        preview_path, original_tif = handle_upload(image_path)
        bound_path = normalize_uploaded_input(
            preview_path, original_tif, sample_aliases=spec.get("sample_aliases")
        )
        upload_registry = dict(upload_registry or {})
        if bound_path:
            upload_registry[spec_key] = {
                "label": spec["label"],
                "path": bound_path,
                "preview": preview_path or bound_path,
                "group": spec["group"],
                "status": "ready",
            }
        else:
            upload_registry.pop(spec_key, None)

        return (
            preview_path,
            original_tif,
            upload_registry,
            render_upload_registry_html(upload_registry),
        )

    return _handler


def run_agent_with_ui(user_question, *payload):
    flat_args = payload[:-1]
    output_history = payload[-1] or []

    for chat_rendered, preview_path, batch_candidate, single_candidate in run_agent(user_question, *flat_args):
        output_history = append_output_history(output_history, batch_candidate, source="Agent")
        latest_single = single_candidate or (output_history[0]["path"] if output_history else None)

        yield (
            chat_rendered,
            preview_path,
            build_batch_files(output_history),
            latest_single,
            render_output_history_html(output_history),
            output_history,
        )


def reset_ui_v2():
    outputs = [
        "",                              # user_input
        render_chat_html([]),            # chat_html
        None,                            # output_image
        [],                              # batch_download_files
        None,                            # preview_download_btn
        render_upload_registry_html({}), # upload_summary_html
        render_output_history_html([]),  # output_history_html
        {},                              # upload_registry_state
        [],                              # output_history_state
    ]
    for _ in INPUT_SPECS:
        outputs.extend([None, None])     # image component + original tif state
    return tuple(outputs)


css = """
:root {
    --topbar-h: 72px;

    --page-pad-top: 16px;
    --page-pad-right: 16px;
    --page-pad-bottom: 16px;
    --page-pad-left: 6px;

    --rail-gap: 10px;

    --left-cluster-w: 720px;
    --right-rail-w: 360px;
    --center-min-w: 760px;

    --left-col-gap: 10px;
    --example-col-fr: 0.9fr;
    --upload-col-fr: 1.1fr;

    --panel-radius: 18px;
    --panel-pad: 16px;

    --composer-h: 148px;
    --left-bottom-ratio: 38%;

    --c-deep: #355872;
    --c-mid: #7AAACE;
    --c-light: #9CD5FF;
    --c-bg: #F7F8F0;
    --c-border: #d6deea;
    --c-text: #132a3f;
    --c-muted: #58738a;
    --c-card: #ffffff;
    --c-soft: #f8fbff;
    --c-user: #edf6ff;
    --c-action: #eef7ff;
    --c-obs: #eefaf7;
    --c-final: #f0f6fb;
    --shadow: 0 10px 28px rgba(53, 88, 114, 0.10);
}

html, body, .gradio-container {
    height: 100%;
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: "Inter", "IBM Plex Sans", "Segoe UI", sans-serif !important;
}

body { overflow: hidden; }

.gradio-container {
    max-width: 100vw !important;
    padding:
        calc(var(--topbar-h) + 20px)
        var(--page-pad-right)
        var(--page-pad-bottom)
        var(--page-pad-left) !important;
    overflow: hidden !important;
}

.topbar-card {
    position: fixed;
    top: 12px;
    left: 16px;
    right: 16px;
    z-index: 60;
    background: linear-gradient(120deg, var(--c-deep), #426a8c 50%, var(--c-mid));
    border: 1px solid rgba(255, 255, 255, 0.24);
    color: #fff;
    border-radius: 16px;
    box-shadow: var(--shadow);
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

.workspace-shell {
    height: calc(100vh - var(--topbar-h) - 36px);
    gap: var(--rail-gap);
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}

#left-cluster,
#center-rail,
#right-rail {
    height: 100% !important;
    min-height: 100% !important;
}

#left-cluster {
    order: 1;
    flex: 0 0 var(--left-cluster-w) !important;
    max-width: var(--left-cluster-w) !important;
    min-width: var(--left-cluster-w) !important;
}

#center-rail {
    order: 2;
    flex: 1 1 auto !important;
    min-width: var(--center-min-w) !important;
}

#right-rail {
    order: 3;
    flex: 0 0 var(--right-rail-w) !important;
    max-width: var(--right-rail-w) !important;
    min-width: var(--right-rail-w) !important;
}

.rail-stack {
    display: flex;
    flex-direction: column;
    gap: var(--rail-gap);
    height: 100%;
    min-height: 100%;
}

.panel-card {
    background: var(--c-card) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--panel-radius) !important;
    box-shadow: var(--shadow) !important;
    padding: var(--panel-pad) !important;
    overflow: hidden;
}

.panel-fill {
    height: 100%;
    min-height: 100%;
    display: flex;
    flex-direction: column;
}

.left-top-card {
    flex: 1 1 auto;
    min-height: 0;
}

.left-bottom-card {
    flex: 0 0 var(--left-bottom-ratio);
    min-height: 280px;
}

.right-bottom-card {
    flex: 1 1 auto;
    min-height: 0;
}

.scroll-y { overflow-y: auto; min-height: 0; }

.section-head { margin-bottom: 12px; }
.section-head h3 { margin: 0; color: var(--c-text); font-size: 18px; }
.section-head p { margin: 6px 0 0; color: var(--c-muted); font-size: 13px; line-height: 1.45; }

.left-cluster-shell {
    display: grid;
    grid-template-columns: var(--example-col-fr) var(--upload-col-fr);
    gap: var(--left-col-gap);
    height: 100%;
    min-height: 0;
    align-items: stretch;
}

.left-sub-card {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
}

.left-sub-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding-right: 2px;
}

.hint-strip {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    border: 1px dashed #bfd3e8;
    background: #f7fbff;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
}

.hint-strip span {
    display: block;
    font-size: 12px;
    color: var(--c-muted);
}

.mode-tabs button[aria-selected="true"] {
    background: var(--c-deep) !important;
    color: #fff !important;
}

.mode-note {
    background: var(--c-soft);
    border: 1px solid #e4edf8;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

.mode-note p {
    margin: 6px 0 0;
    color: var(--c-muted);
    font-size: 12px;
}

.upload-slot,
.result-view,
.download-slot {
    border-radius: 14px !important;
    border: 1px solid #dde7f3 !important;
    background: #fbfdff !important;
}

.upload-slot {
    min-height: 200px;
}

.asset-list,
.history-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.asset-item,
.history-item {
    display: grid;
    grid-template-columns: 76px 1fr;
    gap: 10px;
    align-items: center;
    border: 1px solid #e1e9f3;
    background: #fbfdff;
    border-radius: 14px;
    padding: 10px;
}

.asset-thumb-wrap,
.history-thumb-wrap {
    width: 76px;
    height: 76px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #dbe6f2;
    background: #fff;
}

.asset-thumb,
.history-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.asset-meta,
.history-meta {
    min-width: 0;
}

.asset-title,
.history-title-row strong {
    display: block;
    color: var(--c-text);
    font-size: 13px;
    line-height: 1.35;
}

.asset-sub,
.history-sub {
    margin-top: 4px;
    color: var(--c-muted);
    font-size: 12px;
}

.asset-path,
.history-path {
    margin-top: 4px;
    color: #6f879b;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.history-status {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
}

.status-success {
    background: #e7f7ef;
    color: #1d7a49;
}

.empty-mini-card,
.empty-state {
    border: 1px dashed #c7d6e8;
    border-radius: 14px;
    background: #fbfdff;
    padding: 14px;
}

.empty-mini-card strong,
.empty-state strong {
    display: block;
    margin-bottom: 6px;
}

.empty-mini-card p,
.empty-state p {
    margin: 0;
    font-size: 12px;
    color: var(--c-muted);
}

#center-main-card {
    height: 100%;
    min-height: 100%;
    display: flex;
    flex-direction: column;
}

.chat-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    border: 1px solid var(--c-border);
    border-radius: 14px;
    background: #fff;
}

.chat-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px;
}

.user-msg, .assistant-msg {
    max-width: 88%;
    border-radius: 14px;
    padding: 12px;
    border: 1px solid #dbe5f1;
}

.user-msg {
    margin-left: auto;
    background: var(--c-user);
    border-color: #cbe3fb;
}

.assistant-msg { background: #f9fbff; }
.action-msg { background: var(--c-action); border-color: #cfe5fb; }
.observation-msg { background: var(--c-obs); border-color: #cde9df; }
.final-msg { background: var(--c-final); border-color: #d4e1ef; }
.alert-msg { background: #fff6f1; border-color: #f3d6c6; }

.msg-meta {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}

.round-pill,
.msg-badge {
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    background: #e9f1fa;
    color: var(--c-deep);
}

.msg-heading {
    font-size: 13px;
    font-weight: 700;
}

.assistant-msg pre,
.user-msg pre {
    margin: 0;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.58;
}

#composer-card {
    flex: 0 0 var(--composer-h);
    margin-top: 12px;
    border: 1px solid var(--c-border);
    border-radius: 16px;
    background: #fff;
    padding: 12px;
}

.composer-grid {
    height: 100%;
    align-items: stretch;
    gap: 10px;
}

.input-box textarea {
    min-height: calc(var(--composer-h) - 42px) !important;
    max-height: calc(var(--composer-h) - 42px) !important;
    border-radius: 12px !important;
    border: 1px solid #cbd8e8 !important;
}

.send-btn button,
.clear-btn button,
.ghost-btn button {
    border-radius: 12px !important;
    min-height: 46px !important;
    font-weight: 800 !important;
}

.send-btn button {
    background: var(--c-deep) !important;
    color: #fff !important;
}

.clear-btn button,
.ghost-btn button {
    background: #fff !important;
    border: 1px solid #c8d7e7 !important;
    color: var(--c-text) !important;
}

.toolbar-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}

.preview-toolbar {
    margin-bottom: 10px;
}

.preview-meta {
    margin-top: 10px;
    font-size: 12px;
    color: var(--c-muted);
}

@media (max-width: 1600px) {
    :root {
        --left-cluster-w: 680px;
        --right-rail-w: 340px;
    }
}

@media (max-width: 1360px) {
    body { overflow: auto; }
    .gradio-container { overflow: visible !important; }

    .workspace-shell {
        height: auto;
        flex-wrap: wrap !important;
    }

    #left-cluster,
    #center-rail,
    #right-rail {
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        height: auto !important;
    }

    .left-cluster-shell {
        grid-template-columns: 1fr;
    }

    #center-main-card {
        min-height: 820px;
    }
}
"""

with gr.Blocks(title="WeavLight Workspace", css=css) as demo:
    state_components, image_components = {}, {}
    upload_registry_state = gr.State({})
    output_history_state = gr.State([])

    gr.HTML(
        """
        <div class="topbar-card">
            <div class="topbar-inner">
                <div class="topbar-logo"><img src="./assets/nwpu-logo.svg" alt="logo"/></div>
                <div class="topbar-title">
                    <h1>WeavLight · AI Agent Imagery Workspace</h1>
                    <p>Left cluster optimized · example + upload side by side</p>
                </div>
            </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["workspace-shell"]):
        # Center first in code
        with gr.Column(elem_id="center-rail", elem_classes=["rail-stack"]):
            with gr.Column(elem_id="center-main-card", elem_classes=["panel-card"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>Agent 问答与流程日志</h3>"
                    "<p>中间主栏自适应宽度，聊天历史独立滚动，底部输入区固定悬浮。</p>"
                    "</div>"
                )

                chat_html = gr.HTML(
                    value=render_chat_html([]),
                    elem_classes=["chat-scroll"],
                )

                with gr.Column(elem_id="composer-card"):
                    with gr.Row(elem_classes=["composer-grid"]):
                        user_input = gr.Textbox(
                            label="",
                            placeholder="输入任务指令，可结合左侧已上传图片一起发送...",
                            lines=4,
                            max_lines=8,
                            elem_classes=["input-box"],
                            scale=9,
                        )
                        with gr.Column(scale=2, min_width=150):
                            submit_btn = gr.Button("发送", elem_classes=["send-btn"])
                            clear_btn = gr.Button("清空", elem_classes=["clear-btn"])

        # Left cluster
        with gr.Column(elem_id="left-cluster", elem_classes=["rail-stack"]):
            with gr.Column(elem_classes=["panel-card", "left-top-card", "panel-fill"]):
                with gr.Row(elem_classes=["left-cluster-shell"]):
                    # Upload first in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>输入图片上传</h3>"
                            "<p>与示例模块并排显示，统一高度，底部对齐。</p>"
                            "</div>"
                        )
                        gr.HTML(UPLOAD_HINT_HTML)

                        with gr.Column(elem_classes=["left-sub-scroll"]):
                            with gr.Tabs(elem_classes=["mode-tabs"]):
                                for group in INPUT_GROUPS:
                                    with gr.Tab(group["tab"]):
                                        gr.HTML(
                                            f"<div class='mode-note'><strong>{group['title']}</strong>"
                                            f"<p>{group['note']}</p></div>"
                                        )
                                        if len(group["input_keys"]) == 1:
                                            spec = get_input_spec(group["input_keys"][0])
                                            image_components[spec["key"]] = gr.Image(
                                                type="filepath",
                                                label=spec["label"],
                                                height=max(spec["height"], 230),
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
                                                        height=max(spec["height"], 190),
                                                        sources=["upload", "clipboard"],
                                                        elem_classes=["upload-slot"],
                                                    )
                                                    state_components[spec["key"]] = gr.State()

                        upload_summary_html = gr.HTML(
                            value=render_upload_registry_html({}),
                            elem_classes=["scroll-y"],
                        )

                    # Example second in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>Example 示例模板</h3>"
                            "<p>内容超出时在模块内部滚动，避免拉长整列高度。</p>"
                            "</div>"
                        )

                        example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]
                        with gr.Column(elem_classes=["left-sub-scroll"]):
                            with gr.Accordion("打开 / 收起示例", open=False):
                                with gr.Tabs():
                                    with gr.Tab("General / Index"):
                                        gr.Examples(
                                            examples=[
                                                make_example("Generate a preview from the NBR difference for Topanga State Park, Los Angeles, USA, covering December 2024 and February 2025."),
                                                make_example("Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", primary_image="./assets/S_10_preview.png"),
                                                make_example("Locate and estimate the distance between aircrafts in the scene. Assuming GSD 0.6 px/meter", primary_image="./assets/TG_P0009.png"),
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
                                            examples=[make_example("Compare the two temporal images and identify the major changes.")],
                                            inputs=example_inputs,
                                            preprocess=False,
                                        )

            with gr.Column(elem_classes=["panel-card", "left-bottom-card", "panel-fill"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>当前输出高清预览</h3>"
                    "<p>保留原卡片式设计，与上方左侧聚合区底部对齐。</p>"
                    "</div>"
                )

                with gr.Row(elem_classes=["toolbar-row", "preview-toolbar"]):
                    preview_download_btn = gr.File(
                        label="当前结果下载",
                        elem_classes=["download-slot"],
                    )

                output_image = gr.Image(
                    type="filepath",
                    label="Output Preview",
                    height=340,
                    elem_classes=["result-view"],
                )

                preview_meta_html = gr.HTML(
                    value="<div class='preview-meta'>预览区会始终显示最新输出结果。</div>"
                )

        # Right rail
        with gr.Column(elem_id="right-rail", elem_classes=["rail-stack"]):
            with gr.Column(elem_classes=["panel-card", "right-bottom-card", "panel-fill"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>输出历史下载</h3>"
                    "<p>按生成时间倒序显示所有输出，支持批量下载和一键清空。</p>"
                    "</div>"
                )

                with gr.Row(elem_classes=["toolbar-row"]):
                    clear_history_btn = gr.Button("清空历史", elem_classes=["ghost-btn"])

                output_history_html = gr.HTML(
                    value=render_output_history_html([]),
                    elem_classes=["scroll-y"],
                )

                batch_download_files = gr.File(
                    label="批量下载",
                    file_count="multiple",
                    elem_classes=["download-slot"],
                )

    for spec in INPUT_SPECS:
        image_components[spec["key"]].change(
            fn=make_upload_change_handler(spec["key"]),
            inputs=[image_components[spec["key"]], upload_registry_state],
            outputs=[
                image_components[spec["key"]],
                state_components[spec["key"]],
                upload_registry_state,
                upload_summary_html,
            ],
            trigger_mode="once",
        )

    submit_inputs = [user_input]
    for spec in INPUT_SPECS:
        submit_inputs.extend([image_components[spec["key"]], state_components[spec["key"]]])
    submit_inputs.append(output_history_state)

    submit_btn.click(
        fn=run_agent_with_ui,
        inputs=submit_inputs,
        outputs=[
            chat_html,
            output_image,
            batch_download_files,
            preview_download_btn,
            output_history_html,
            output_history_state,
        ],
    )

    clear_history_btn.click(
        fn=clear_output_history,
        inputs=[],
        outputs=[
            output_history_html,
            batch_download_files,
            preview_download_btn,
            output_image,
            output_history_state,
        ],
    )

    clear_outputs = [
        user_input,
        chat_html,
        output_image,
        batch_download_files,
        preview_download_btn,
        upload_summary_html,
        output_history_html,
        upload_registry_state,
        output_history_state,
    ]
    for spec in INPUT_SPECS:
        clear_outputs.extend([image_components[spec["key"]], state_components[spec["key"]]])

    clear_btn.click(
        fn=reset_ui_v2,
        inputs=[],
        outputs=clear_outputs,
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=4444, allowed_paths=["./"])

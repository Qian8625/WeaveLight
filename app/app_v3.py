import gradio as gr
import json
import re
import torch
import os
import html
import inspect
from PIL import Image
from datetime import datetime
import rasterio
from pathlib import Path
import numpy as np
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from tool_server.tool_workers.skills.router import route_skills
from tool_server.tool_workers.skills.catalog import build_selected_skill_catalog
from tool_server.tool_workers.skills.registry import SKILL_REGISTRY
# 修改输入文件的路径
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
PROJECT_ROOT = APP_DIR.parent


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
        "input_keys": ["primary_image"],
    },
    {
        "tab": "TVDI",
        "input_keys": ["lst_image", "ndvi_image"],
    },
    {
        "tab": "Time Series",
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
    "SmallObjectDetection": {"image": "primary_image"},
    "CloudRemoval": {
        "image": "primary_image",
        "nir_image": "time1_image",
    },
    "GetBboxFromGeotiff": {"geotiff": "primary_image"},
    "DisplayOnGeotiff": {"geotiff": "primary_image"},
    "ChangeDetection": {"pre_image": "time1_image", "post_image": "time2_image"},
    "TVDIAnalysis": {"ndvi_path": "ndvi_image", "lst_path": "lst_image"},
    "SARToRGB": {"image": "primary_image"},
    "SARPreprocessing": {"image": "primary_image"},
}

SKILL_INPUT_BINDINGS = {
    "image": "primary_image",
    "geotiff": "primary_image",
    "pre_image": "time1_image",
    "post_image": "time2_image",
    "rgb_image": "primary_image",
    "sar_image": "time1_image",
}

TOOL_DEFAULT_ARGUMENTS = {
    "AddText": {"color": "green"},
    "TVDIAnalysis": {"output_path": "tvdi_result.tif"},
    "CloudRemoval": {"output_path": "cloud_removed_result.tif"},
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

    def _build_system_prompt(self, extra_system_prompt: str = "") -> str:
        if extra_system_prompt and extra_system_prompt.strip():
            return self.system_prompt + "\n\n" + extra_system_prompt.strip()
        return self.system_prompt

    def prepend_system_prompt(self, conversation, extra_system_prompt: str = ""):
        final_system_prompt = self._build_system_prompt(extra_system_prompt)

        conversation = list(conversation)

        if not conversation or conversation[0]["role"] != "system":
            conversation = [{"role": "system", "content": final_system_prompt}] + conversation
        else:
            conversation[0] = {"role": "system", "content": final_system_prompt}

        return conversation

    def generate(self, conversation, extra_system_prompt: str = ""):
        conversation = self.prepend_system_prompt(
            conversation,
            extra_system_prompt=extra_system_prompt
        )

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )

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
    if not isinstance(actions, list):
        actions = []
    return thought, actions

def resolve_existing_file(path):
    if not path:
        return None

    p = Path(str(path)).expanduser()

    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [
            Path.cwd() / p,
            PROJECT_ROOT / p,
            APP_DIR / p,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return None

def inject_skill_runtime_arguments(injected_args, registry):
    merged = dict(injected_args or {})
    skill_name = merged.get("skill_name")

    if not skill_name or skill_name not in SKILL_REGISTRY:
        return merged

    spec = SKILL_REGISTRY[skill_name]
    input_names = list(spec.get("required_inputs", [])) + list(spec.get("optional_inputs", []))

    for arg_name in input_names:
        if merged.get(arg_name) not in [None, ""]:
            continue

        registry_key = SKILL_INPUT_BINDINGS.get(arg_name)
        if registry_key and registry.get(registry_key):
            merged[arg_name] = registry[registry_key]

    return merged

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
    expected = len(INPUT_SPECS) * 2
    flat_args = list(flat_args or [])
    if len(flat_args) < expected:
        flat_args.extend([None] * (expected - len(flat_args)))

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

    lines.extend(
        [
            "",
            "Reference rules for tool arguments:",
            "- Uploaded inputs can be referenced explicitly with $primary_image, $lst_image, $ndvi_image, $time1_image, $time2_image if available.",
            "- Intermediate outputs from previous tool calls will be exposed as reusable artifact ids such as $latest_image or $round_2_sartorgb_image.",
            "- If a later tool should use an intermediate result, reference it explicitly in the action arguments.",
        ]
    )
    return "\n".join(lines)

def find_unresolved_references(api_args):
    unresolved = []

    def _walk(value):
        if isinstance(value, str) and value.startswith("$"):
            unresolved.append(value)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)

    _walk(api_args)
    return unresolved

def build_artifact_summary(registry):
    lines = []

    uploaded = []
    for spec in INPUT_SPECS:
        key = spec["key"]
        if registry.get(key):
            uploaded.append(f"- ${key} | uploaded input | path={registry[key]}")
    if uploaded:
        lines.append("Uploaded inputs:")
        lines.extend(uploaded)

    artifacts = registry.get("artifacts", [])
    if artifacts:
        lines.append("")
        lines.append("Reusable artifacts:")
        for item in artifacts[-10:]:
            lines.append(
                f"- ${item['id']} | type={item['type']} | tool={item['tool']} | path={item['path']}"
            )

    alias_lines = []
    if registry.get("latest_image"):
        alias_lines.append(f"- $latest_image | path={registry['latest_image']}")
    if registry.get("latest_vector"):
        alias_lines.append(f"- $latest_vector | path={registry['latest_vector']}")
    if registry.get("latest_output"):
        alias_lines.append(f"- $latest_output | path={registry['latest_output']}")
    if alias_lines:
        lines.append("")
        lines.append("Convenience aliases:")
        lines.extend(alias_lines)

    if not lines:
        return "No uploaded inputs or reusable artifacts are currently available."

    return "\n".join(lines)

def _resolve_ref_value(value, registry):
    if isinstance(value, str) and value.startswith("$"):
        ref_key = value[1:]
        return registry.get(ref_key, value)

    if isinstance(value, list):
        return [_resolve_ref_value(v, registry) for v in value]

    if isinstance(value, dict):
        return {k: _resolve_ref_value(v, registry) for k, v in value.items()}

    return value


def resolve_argument_references(api_args, registry):
    return _resolve_ref_value(dict(api_args or {}), registry)

def inject_runtime_arguments(api_name, api_args, registry, current_gpkg):
    merged = dict(api_args or {})

    for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
        if merged.get(tool_arg) in [None, ""] and registry.get(registry_key):
            merged[tool_arg] = registry[registry_key]

    if api_name in GPKG_REQUIRED_TOOLS and current_gpkg and merged.get("gpkg") in [None, ""]:
        merged["gpkg"] = current_gpkg

    for k, v in TOOL_DEFAULT_ARGUMENTS.get(api_name, {}).items():
        if merged.get(k) in [None, ""] or k not in merged:
            merged[k] = v

    return merged

def register_artifact(registry, artifact_id, path, artifact_type, tool_name):
    if not path:
        return registry

    resolved = resolve_existing_file(path) or path
    preview = resolved
    if artifact_type == "image":
        preview = make_preview_if_tif(resolved)
        preview = resolve_existing_file(preview) or resolved

    registry[artifact_id] = resolved

    artifacts = list(registry.get("artifacts", []))
    if not any(item.get("id") == artifact_id for item in artifacts):
        artifacts.append(
            {
                "id": artifact_id,
                "path": resolved,
                "preview": preview,
                "type": artifact_type,
                "tool": tool_name,
            }
        )
    registry["artifacts"] = artifacts

    if artifact_type == "image":
        registry["latest_image"] = resolved
        registry["latest_output"] = resolved
    elif artifact_type == "vector":
        registry["latest_vector"] = resolved
        registry["latest_output"] = resolved
    else:
        registry["latest_output"] = resolved

    return registry

def build_media_item(path, name=None):
    resolved = resolve_existing_file(path)
    if not resolved:
        return None

    preview = make_preview_if_tif(resolved)
    preview_resolved = resolve_existing_file(preview) or resolved

    return {
        "path": resolved,
        "preview": preview_resolved,
        "name": name or os.path.basename(resolved),
    }


def collect_round_image_media(registry, round_id, api_name=None):
    media = []
    seen = set()

    for item in registry.get("artifacts", []):
        artifact_id = item.get("id", "")
        artifact_type = item.get("type")
        artifact_tool = item.get("tool")
        artifact_path = item.get("path")

        if artifact_type != "image":
            continue
        if not artifact_id.startswith(f"round_{round_id}_"):
            continue
        if api_name is not None and artifact_tool != api_name:
            continue
        if not artifact_path or artifact_path in seen:
            continue

        media_item = {
            "path": artifact_path,
            "preview": item.get("preview") or artifact_path,
            "name": f"{artifact_tool} · {os.path.basename(artifact_path)}",
        }
        media.append(media_item)
        seen.add(artifact_path)

    return media

def collect_action_input_media(api_args):
    media = []
    seen = set()

    candidate_keys = [
        "image", "geotiff", "pre_image", "post_image",
        "nir_image", "ndvi_path", "lst_path"
    ]

    for key in candidate_keys:
        value = api_args.get(key)
        if not isinstance(value, str):
            continue

        resolved = resolve_existing_file(value)
        if not resolved or resolved in seen:
            continue

        media_item = build_media_item(resolved, name=f"input:{key}")
        if media_item:
            media.append(media_item)
            seen.add(resolved)

    return media

def update_registry_with_tool_response(registry, api_name, tool_resp, round_id=None):
    if not isinstance(tool_resp, dict):
        return registry

    if tool_resp.get("error_code") != 0:
        return registry

    registry = dict(registry)

    # 1) Prefer normalized artifacts returned by SkillExecutor
    artifacts = tool_resp.get("artifacts")
    if isinstance(artifacts, list):
        for idx, item in enumerate(artifacts, start=1):
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            artifact_type = item.get("type") or "other"
            artifact_id = item.get("id") or f"round_{round_id}_{api_name.lower()}_{idx}"

            if not path:
                continue

            # 保持当前 registry 命名习惯，避免不同 round 的 id 冲突
            scoped_artifact_id = f"round_{round_id}_{api_name.lower()}_{artifact_id}"

            registry = register_artifact(
                registry,
                artifact_id=scoped_artifact_id,
                path=path,
                artifact_type=artifact_type,
                tool_name=api_name,
            )

    # 2) Backward-compatible fallback for old tool outputs
    visual = detect_output_visual(tool_resp)
    if visual:
        artifact_id = f"round_{round_id}_{api_name.lower()}_image"
        registry = register_artifact(
            registry,
            artifact_id=artifact_id,
            path=visual,
            artifact_type="image",
            tool_name=api_name,
        )

    gpkg_path = tool_resp.get("gpkg")
    if isinstance(gpkg_path, str):
        resolved_gpkg = resolve_existing_file(gpkg_path)
        if resolved_gpkg:
            artifact_id = f"round_{round_id}_{api_name.lower()}_vector"
            registry = register_artifact(
                registry,
                artifact_id=artifact_id,
                path=resolved_gpkg,
                artifact_type="vector",
                tool_name=api_name,
            )

    if api_name == "TVDIAnalysis":
        out = tool_resp.get("output_path") or visual
        if out:
            registry["tvdi_result"] = out
            registry = register_artifact(
                registry,
                artifact_id=f"round_{round_id}_{api_name.lower()}_tvdi",
                path=out,
                artifact_type="image",
                tool_name=api_name,
            )

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

        media_html = ""
        media_items = msg.get("media") or []
        if media_items:
            cards = []
            for media in media_items:
                preview = html.escape(media.get("preview", ""))
                path = html.escape(media.get("path", ""))
                name = html.escape(media.get("name", "image"))

                cards.append(
                    f"""
                    <a class="msg-media-card" href="/gradio_api/file={path}" target="_blank">
                        <img class="msg-media-img" src="/gradio_api/file={preview}" alt="{name}" />
                        <div class="msg-media-name">{name}</div>
                    </a>
                    """
                )
            media_html = f'<div class="msg-media-grid">{"".join(cards)}</div>'

        blocks.append(
            f'<div class="{bubble_class}"><div class="msg-meta">{"".join(meta_items)}</div>{body_html}{media_html}</div>'
        )
    blocks.append("</div>")
    return "".join(blocks)


Model = None
MODEL_INIT_ERROR = None
tool_manager = ToolManager()


def normalize_tool_response(tool_resp, api_name):
    if isinstance(tool_resp, dict):
        return tool_resp

    text = (
        f"{api_name} returned empty response."
        if tool_resp is None
        else f"{api_name} returned non-dict response: {tool_resp}"
    )
    return {"error_code": -1, "text": text}


def get_model():
    global Model, MODEL_INIT_ERROR

    if Model is None and MODEL_INIT_ERROR is None:
        try:
            Model = LLM(PRETRAINED_PATH)
        except Exception as exc:
            MODEL_INIT_ERROR = exc

    if MODEL_INIT_ERROR is not None:
        raise RuntimeError(f"模型初始化失败: {MODEL_INIT_ERROR}")

    return Model


def run_agent(user_question, *flat_args):
    registry = build_registry_from_args(flat_args)
    current_gpkg = None
    chat_msgs = [{"role": "user", "content": user_question}]


    candidate_skills = route_skills(
        user_query=user_question,
        input_registry=registry,
        top_k=3,
        min_score=2.5,
        score_margin=0.5,
    )
    selected_skill_prompt = build_selected_skill_catalog(candidate_skills)

    initial_message = build_initial_user_message(user_question, registry)
    conversation = [{"role": "user", "content": initial_message}]

    if candidate_skills:
            router_text = "\n".join(
                [
                    f"- {item['skill_name']} (score={item['score']:.2f}) | reasons: {'; '.join(item.get('reasons', []))}"
                    for item in candidate_skills
                ]
            )
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round 0] 🧭 SkillRouter\nCandidate skills:\n{router_text}",
                }
            )
    else:
        chat_msgs.append(
            {
                "role": "assistant",
                "content": "[Round 0] 🧭 SkillRouter\nNo suitable skill matched. Fall back to raw tools if needed.",
            }
        )

    def emit(download_path=None):
        return render_chat_html(chat_msgs), download_path

    try:
        model = get_model()
    except Exception as exc:
        chat_msgs.append(
            {"role": "assistant", "content": f"[Round 0] ❌ Observation\n{exc}"}
        )
        yield emit(None)
        return

    for round_id in range(1, MAX_TOOL_ROUNDS + 1):
        try:
            model_output = model.generate(conversation, extra_system_prompt=selected_skill_prompt)
        except Exception as exc:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n模型推理失败：{exc}",
                }
            )
            yield emit(None)
            return

        thought, actions = extract_thought_and_actions(model_output)

        if not actions:
            chat_msgs.append(
                {"role": "assistant", "content": f"[Round {round_id}] Thinking...\n{thought or model_output}"}
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0] if isinstance(actions[0], dict) else {}
        api_name = action.get("name")
        raw_args = action.get("arguments", {})
        api_args = raw_args if isinstance(raw_args, dict) else {}
        api_args = resolve_argument_references(api_args, registry)  

        if api_name in SKILL_REGISTRY:
            api_args = dict(api_args or {})
            api_args["skill_name"] = api_name
            api_name = "SkillExecutor"

        unresolved_refs = find_unresolved_references(api_args)
        if unresolved_refs:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n未解析的中间结果引用：{', '.join(unresolved_refs)}",
                }
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"The action used unresolved references: {', '.join(unresolved_refs)}.\n"
                        f"Please use one of the available uploaded inputs or reusable artifacts."
                    ),
                }
            )
            continue

        if not api_name:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n无效动作格式：缺少工具名。",
                }
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append(
                {
                    "role": "user",
                    "content": "The action format is invalid. Please output a valid JSON action list.",
                }
            )
            continue

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "") or "已完成，结果请查看输出历史或下载文件。"
            chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] ✅ Final\n{final_answer}"})
            download = registry.get("latest_output") or registry.get("tvdi_result")
            yield emit(download)
            return

        injected_args = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)

        if api_name == "SkillExecutor":
            # 情况1：当前没有任何可用候选 skill，禁止走 SkillExecutor，强制回退 raw tools
            if not candidate_skills:
                chat_msgs.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"[Round {round_id}] ❌ Observation\n"
                            "There are currently no available skills matched. "
                            "Please do not use SkillExecutor, but directly select the appropriate raw tools."
                        ),
                    }
                )
                yield emit(None)
                conversation.append({"role": "assistant", "content": model_output})
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "No suitable skill is available for this task. "
                            "Do not use SkillExecutor. "
                            "Please choose raw tools directly from the tool list and continue."
                        ),
                    }
                )
                continue

            # 情况2：有候选 skill，但模型没填 skill_name，自动补第一个候选
            if not injected_args.get("skill_name"):
                injected_args["skill_name"] = candidate_skills[0]["skill_name"]

            # 关键新增：根据 skill registry 自动补上传输入
            injected_args = inject_skill_runtime_arguments(injected_args, registry)

        action_media = collect_action_input_media(injected_args)
        chat_msgs.append(
            {
                "role": "assistant",
                "content": f"[Round {round_id}] 🛠️ Action: {api_name}\nArgs:\n{json.dumps(injected_args, ensure_ascii=False, indent=2)}",
                "media": action_media,
            }
        )
        yield emit(None)

        try:
            raw_tool_resp = tool_manager.call_tool(api_name, injected_args)
        except Exception as exc:
            raw_tool_resp = {"error_code": -1, "text": f"Tool call failed: {exc}"}

        tool_resp = normalize_tool_response(raw_tool_resp, api_name)
        obs_text = tool_resp.get("text", "")
        if tool_resp.get("error_code") == 0 and "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")

        registry = update_registry_with_tool_response(
            registry,
            api_name,
            tool_resp,
            round_id=round_id,
        )

        round_media = collect_round_image_media(
            registry,
            round_id=round_id,
            api_name=api_name,
        )

        artifact_summary = build_artifact_summary(registry)
        status = "✅" if tool_resp.get("error_code") == 0 else "❌"
        chat_msgs.append(
            {
                "role": "assistant",
                "content": f"[Round {round_id}] {status} Observation\n{obs_text}",
                "media": round_media,
            }
        )
        download = registry.get("latest_output") or registry.get("tvdi_result")
        yield emit(download)

        conversation.append({"role": "assistant", "content": model_output})
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"OBSERVATION:\n"
                    f"{api_name} outputs: {obs_text}\n\n"
                    f"{artifact_summary}\n\n"
                    f"When a later tool should use an uploaded input or an intermediate result, "
                    f"reference it explicitly in the action arguments using $key."
                ),
            }
        )

    chat_msgs.append({"role": "assistant", "content": "达到最大轮数，已停止。"})
    yield emit(registry.get("latest_output") or registry.get("tvdi_result"))


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
    return render_output_history_html([]), [], []


def make_upload_change_handler(spec_key):
    def _handler(image_path):
        preview_path, original_tif = handle_upload(image_path)
        return preview_path, original_tif
    return _handler
    
def run_agent_with_ui(user_question, *payload):
    flat_args = payload[:-1]
    output_history = payload[-1] or []

    for chat_rendered, batch_candidate in run_agent(user_question, *flat_args):
        output_history = append_output_history(output_history, batch_candidate, source="Agent")

        yield (
            chat_rendered,
            build_batch_files(output_history),
            render_output_history_html(output_history),
            output_history,
        )


def reset_ui_v2():
    outputs = [
        "",                             # user_input
        render_chat_html([]),           # chat_html
        [],                             # batch_download_files
        render_output_history_html([]), # output_history_html
        [],                             # output_history_state
    ]
    for _ in INPUT_SPECS:
        outputs.extend([None, None])     # image component + original tif state
    return tuple(outputs)


def supports_blocks_css():
    try:
        return "css" in inspect.signature(gr.Blocks.__init__).parameters
    except Exception:
        return True


def supports_launch_css():
    try:
        return "css" in inspect.signature(gr.Blocks.launch).parameters
    except Exception:
        return False


css = """
/* ===== layout backbone: single source of truth ===== */
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

    /* 统一由这一处控制桌面三栏高度 */
    --workspace-h: 900px;

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

/* 允许整页滚动，彻底解除外层裁切 */
body {
    overflow: auto !important;
}

.gradio-container {
    max-width: 100vw !important;
    padding:
        var(--page-pad-top)
        var(--page-pad-right)
        var(--page-pad-bottom)
        var(--page-pad-left) !important;
    overflow: visible !important;
}

/* title */
.page-header {
    width: 100%;
    margin: 0 0 14px 0;
    padding: 10px 0 6px 0;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.page-header-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;

    width: 100%;
    text-align: left;
    margin-bottom: 12px;
}

.page-header-logo {
    width: 88px;
    height: 88px;
    flex: 0 0 88px;

    display: flex;
    align-items: center;
    justify-content: center;
}

.page-header-logo img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
}

.page-header-title h1 {
    margin: 0;
    font-size: 28px;
    line-height: 1.2;
    color: var(--c-text);
    font-weight: 800;
}

.page-header-title p {
    margin: 6px 0 0 0;
    font-size: 13px;
    line-height: 1.5;
    color: var(--c-muted);
}

.page-header-links {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    margin-top: 0px;
}

.header-link-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    padding: 8px 24px;
    border-radius: 999px;
    border: 1px solid #c8d7e7;
    background: #ffffff;
    color: var(--c-text);
    text-decoration: none;
    font-size: 13px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.header-link-btn:hover {
    background: var(--c-deep);
    color: #ffffff;
    border-color: var(--c-deep);
}

/* ===== workspace ===== */
.workspace-shell {
    min-height: var(--workspace-h);
    height: auto !important;
    max-height: none !important;
    gap: var(--rail-gap);
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}

/* 三栏统一固定高度，只在这一处控制 */
#left-cluster,
#center-rail,
#right-rail {
    height: var(--workspace-h) !important;
    min-height: var(--workspace-h) !important;
    max-height: var(--workspace-h) !important;
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

/* 这些层只负责继承父高度，不再抢高度控制权 */
.rail-stack,
.panel-fill,
.left-top-card,
.right-bottom-card,
#center-main-card,
.left-cluster-shell,
.left-sub-card {
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
}

.rail-stack {
    display: flex;
    flex-direction: column;
    gap: var(--rail-gap);
}

.panel-card {
    background: var(--c-card) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--panel-radius) !important;
    box-shadow: var(--shadow) !important;
    padding: var(--panel-pad) !important;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
}

.panel-fill,
#center-main-card,
.left-sub-card {
    display: flex;
    flex-direction: column;
}

.section-head { margin-bottom: 12px; }
.section-head h3 { margin: 0; color: var(--c-text); font-size: 18px; }
.section-head p { margin: 6px 0 0; color: var(--c-muted); font-size: 13px; line-height: 1.45; }

/* 左侧上下两块，等高分布 */
.left-cluster-shell {
    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--left-col-gap);
    align-items: stretch;
    height: 100% !important;
    min-height: 0 !important;
}

.left-cluster-shell > .left-sub-card {
    min-height: 0 !important;
    height: auto !important;
    max-height: none !important;
}

/* 统一滚动职责：只让真正内容区滚动 */
.scroll-y {
    min-height: 0 !important;
    max-height: none !important;
    overflow-y: auto;
}

.left-sub-scroll,
.chat-scroll,
.history-scroll {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow-y: auto !important;
}

.upload-panel-body {
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
    height: 100% !important;
}

.upload-tabs {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

.upload-slot {
    min-height: 0 !important;
    height: 100% !important;
}

/* 中间聊天区 */
.chat-scroll {
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

#center-main-card {
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
}

/* Mission Console / chat 区 */
#mission-console {
    flex: 4 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;

    border: 1px solid var(--c-border);
    border-radius: 14px;
    background: #fff;
}

/* 输入区固定高度，由 composer 独占 */
#composer-card {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    margin-top: 12px;

    border: 1px solid var(--c-border);
    border-radius: 16px;
    background: #fff;
    padding: 12px;
    align-items: stretch;
    gap: 10px;
}

#composer-card .input-box textarea {
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    border-radius: 12px !important;
    border: 1px solid #cbd8e8 !important;
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

/* 右侧：用 order 直接把下载区放到历史区上面，无需改 Python */
.right-bottom-card {
    display: flex;
    flex-direction: column;
}

.right-bottom-card > .toolbar-row {
    order: 1;
    flex: 0 0 auto;
}

.right-bottom-card > .history-scroll {
    order: 2;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

.right-bottom-card > .download-slot {
    order: 3;
    flex: 0 0 220px !important;
    min-height: 220px;
    max-height: 220px;
    overflow-y: auto;
    margin-top: 12px;
    margin-bottom: 0;
}

/* 视觉样式 */
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
.download-slot {
    border-radius: 14px !important;
    border: 1px solid #dde7f3 !important;
    background: #fbfdff !important;
}

.upload-slot {
    min-height: 200px;
}

.history-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

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

.history-thumb-wrap {
    width: 76px;
    height: 76px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #dbe6f2;
    background: #fff;
}

.history-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.history-meta {
    min-width: 0;
}

.history-title-row strong {
    display: block;
    color: var(--c-text);
    font-size: 13px;
    line-height: 1.35;
}

.history-sub {
    margin-top: 4px;
    color: var(--c-muted);
    font-size: 12px;
}

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

.msg-media-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.msg-media-card {
    display: block;
    text-decoration: none;
    background: #ffffff;
    border: 1px solid #dbe5f1;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.2s ease;
}

.msg-media-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(53, 88, 114, 0.10);
}

.msg-media-img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    display: block;
    background: #f5f8fc;
}

.msg-media-name {
    padding: 8px 10px;
    font-size: 11px;
    color: var(--c-muted);
    border-top: 1px solid #e8eef5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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

/* 中等屏幕稍微收窄左右栏 */
@media (max-width: 1600px) {
    :root {
        --left-cluster-w: 680px;
        --right-rail-w: 340px;
    }
}

/* 小屏切换为自然流式布局，不再强行固定三栏高度 */
@media (max-width: 1360px) {
    :root {
        --workspace-h: auto;
    }

    .workspace-shell {
        min-height: 0;
        height: auto !important;
        max-height: none !important;
        flex-wrap: wrap !important;
    }

    #left-cluster,
    #center-rail,
    #right-rail {
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
    }

    .left-cluster-shell {
        grid-template-columns: 1fr;
        height: auto !important;
        max-height: none !important;
    }

    .rail-stack,
    .panel-fill,
    .left-top-card,
    .right-bottom-card,
    #center-main-card,
    .left-sub-card {
        height: auto !important;
        max-height: none !important;
    }

    .left-sub-scroll,
    .chat-scroll,
    .history-scroll {
        flex: 0 0 auto !important;
        max-height: 520px !important;
    }
}
"""

blocks_kwargs = {"title": "Remote Sensing Data Intelligent Interpretation System"}
BLOCKS_HAS_CSS = supports_blocks_css()
LAUNCH_HAS_CSS = supports_launch_css()
if BLOCKS_HAS_CSS:
    blocks_kwargs["css"] = css

LOGO_PATH = os.path.join("/home/ubuntu/01_Code/OpenEarthAgent/assets/nwpu-logo.png")

with gr.Blocks(**blocks_kwargs) as demo:
    state_components, image_components = {}, {}
    output_history_state = gr.State([])

    gr.HTML(
        f"""
        <div class="page-header">
            <div class="page-header-main">
                <div class="page-header-logo">
                    <img src="/gradio_api/file={LOGO_PATH}" alt="logo"/>
                </div>
                <div class="page-header-title">
                    <h1>NPU Remote Sensing Data Intelligent Interpretation System</h1>
                    <p>NPU Intelligent Vision and Information Processing Laboratory, Manager : Li Huihui</p>
                </div>
            </div>
            
            <div class="page-header-links">
                <a class="header-link-btn" href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank" rel="noopener noreferrer">arXiv</a>
                <a class="header-link-btn" href="https://github.com/NWPU-LHH/" target="_blank" rel="noopener noreferrer">GitHub</a>
                <a class="header-link-btn" href="https://your-project-page.example.com" target="_blank" rel="noopener noreferrer">Project Page</a>
                <a class="header-link-btn" href="https://huggingface.co/" target="_blank" rel="noopener noreferrer">Hugging Face</a>
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
                    "</div>"
                )

                chat_html = gr.HTML(
                    value=render_chat_html([]),
                    elem_id="mission-console",
                    elem_classes=["chat-scroll"],
                )

                with gr.Row(elem_id="composer-card", elem_classes=["composer-grid"]):
                    user_input = gr.Textbox(
                        label="",
                        placeholder="请输入任务指令，相关例子可以直接点击Example...",
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
                with gr.Column(elem_classes=["left-cluster-shell"]):
                    # Upload first in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>输入图片上传</h3>"
                            "</div>"
                        )

                        with gr.Column(elem_classes=["left-sub-scroll", "upload-panel-body"]):
                            with gr.Tabs(elem_classes=["mode-tabs", "upload-tabs"]):
                                for group in INPUT_GROUPS:
                                    with gr.Tab(group["tab"]):
                                        if len(group["input_keys"]) == 1:
                                            spec = get_input_spec(group["input_keys"][0])
                                            image_components[spec["key"]] = gr.Image(
                                                type="filepath",
                                                label=spec["label"],
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
                                                        sources=["upload", "clipboard"],
                                                        elem_classes=["upload-slot"],
                                                    )
                                                    state_components[spec["key"]] = gr.State()

                    # Example second in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>Example</h3>"
                            "</div>"
                        )

                        example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]
                        with gr.Column(elem_classes=["left-sub-scroll"]):
                            with gr.Tabs():
                                with gr.Tab("Det/Seg"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Detect the aircraft in the picture, draw it and estimate the distance between aircrafts in the scene. assuming GSD 0.6px/meter.", 
                                                primary_image="./assets/TG_P0009.png"
                                                ),
                                            make_example(
                                                "Use a small target detection model to detect ships in the map and label them in the image.",
                                                primary_image="./assets/b2_48_6.tif"
                                            ),
                                            make_example(
                                                "Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", 
                                                primary_image="./assets/S_10_preview.png"
                                                ),
                                            make_example(
                                                "Segment the road roundabout and measure its pixel area. Convert that pixel count to square meters using a ground sample distance of 0.132599419033 m/pixel. ",
                                                primary_image="./assets/TG_P0104.png"
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("Attribute"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Detect all red house targets and calculate the area of the largest red house (in square meters), and plot it in a chart.",
                                                primary_image="./assets/TG_70028.jpg"
                                            ),
                                            make_example(
                                                "How many cars are there and are all of them traveling in the southwest direction?",
                                                primary_image="./assets/TG_P0214.png"
                                            ),
                                            make_example(
                                                "How many aircraft are parked at the terminal apron, and what are their color attributes? Are they have same color?",
                                                primary_image="./assets/TG_P0010.png"
                                            ),
                                            make_example(
                                                "Can you check the image if the ship is heading towards the shore?",
                                                primary_image="./assets/000642.jpg"
                                            ),
                                            make_example("For the area within a 1000 m radius of Shimen Reservoir,China, generate a DEM layer, create contour lines, and visualize the create contour lines on the map.",),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("SAR Trans/Det"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Please convert this SAR image to an RGB image",
                                                primary_image="./assets/sar_test_3.png"
                                            ),
                                            make_example(
                                                "First, the SAR image is converted from SAR to RGB. Next, target detection is performed, and the target is then marked on the RGB image.",
                                                primary_image="./assets/sar_test_3.png"
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("TimeSeries Analyze"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", 
                                                primary_image="./assets/S_10_preview.png"
                                                ),
                                            make_example(
                                                "List all the damaged buildings in the post-disaster image..",
                                                time1_image="./assets/TG_santa-rosa-wildfire_00000102_pre_disaster.png",
                                                time2_image="./assets/TG_santa-rosa-wildfire_00000102_post_disaster.png"
                                            ),
                                            make_example(
                                                "Find remote sensing images from June 2024, output the number of images, and describe the last image",
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("Skill Test"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Examine buildings damaged after the incident and express the target proportion of damaged in percentage.",
                                                time1_image="./assets/TG_santa-rosa-wildfire_00000181_pre_disaster.png",
                                                time2_image="./assets/TG_santa-rosa-wildfire_00000181_post_disaster.png"
                                            ),
                                            make_example(
                                                "List all the damaged buildings in the post-disaster image..",
                                                time1_image="./assets/TG_santa-rosa-wildfire_00000102_pre_disaster.png",
                                                time2_image="./assets/TG_santa-rosa-wildfire_00000102_post_disaster.png"
                                            ),
                                            make_example(
                                                "Find remote sensing images from June 2024, output the number of images, and describe the last image",
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )

        # Right rail
        with gr.Column(elem_id="right-rail", elem_classes=["rail-stack"]):
            with gr.Column(elem_classes=["panel-card", "right-bottom-card", "panel-fill"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>输出历史下载</h3>"
                    "</div>"
                )

                batch_download_files = gr.File(
                    label="批量下载",
                    file_count="multiple",
                    elem_classes=["download-slot"],
                )

                output_history_html = gr.HTML(
                    value=render_output_history_html([]),
                    elem_classes=["scroll-y", "history-scroll"],
                )

                with gr.Row(elem_classes=["toolbar-row"]):
                    clear_history_btn = gr.Button("清空历史", elem_classes=["ghost-btn"])

    for spec in INPUT_SPECS:
        image_components[spec["key"]].change(
            fn=make_upload_change_handler(spec["key"]),
            inputs=[image_components[spec["key"]]],
            outputs=[
                image_components[spec["key"]],
                state_components[spec["key"]],
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
            batch_download_files,
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
            output_history_state,
        ],
    )

    clear_outputs = [
        user_input,
        chat_html,
        batch_download_files,
        output_history_html,
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
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 4444,
        "allowed_paths": ["./"],
    }
    if LAUNCH_HAS_CSS and not BLOCKS_HAS_CSS:
        launch_kwargs["css"] = css
    demo.queue().launch(**launch_kwargs)

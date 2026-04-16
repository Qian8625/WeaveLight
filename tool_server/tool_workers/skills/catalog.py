import os
import re
from typing import Dict, List, Any

from tool_server.tool_workers.skills.registry import SKILL_REGISTRY


def _read_text(path: str) -> str:
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getcwd(), path),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def _extract_section(md_text: str, heading: str) -> List[str]:
    """
    Extract bullet items under a section like:
    # When to use
    - xxx
    - yyy
    """
    pattern = rf"(?ims)^#\s*{re.escape(heading)}\s*$([\s\S]*?)(?=^#\s|\Z)"
    m = re.search(pattern, md_text)
    if not m:
        return []

    block = m.group(1).strip()
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
    return lines


def _extract_frontmatter_field(md_text: str, field_name: str, default=None):
    """
    Lightweight parser for simple YAML-like frontmatter.
    Handles:
      key: value
      key:
        - a
        - b
    """
    if default is None:
        default = []

    m = re.match(r"(?s)^---\n(.*?)\n---", md_text)
    if not m:
        return default

    frontmatter = m.group(1)
    lines = frontmatter.splitlines()

    for idx, line in enumerate(lines):
        if re.match(rf"^{re.escape(field_name)}:\s*", line):
            value = line.split(":", 1)[1].strip()
            if value:
                # scalar
                return value.strip().strip('"').strip("'")
            # list block
            items = []
            j = idx + 1
            while j < len(lines):
                l = lines[j]
                if re.match(r"^[A-Za-z0-9_]+:\s*", l):
                    break
                l = l.strip()
                if l.startswith("- "):
                    items.append(l[2:].strip())
                j += 1
            return items if items else default

    return default


def load_skill_catalog() -> Dict[str, Dict[str, Any]]:
    catalog = {}
    for skill_name, spec in SKILL_REGISTRY.items():
        md_path = spec["md_path"]
        md_text = _read_text(md_path)

        description = _extract_frontmatter_field(md_text, "description", "")
        required_inputs = spec.get("required_inputs", [])
        when_to_use = _extract_section(md_text, "When to use")
        do_not_use = _extract_section(md_text, "Do not use when")

        catalog[skill_name] = {
            "skill_name": skill_name,
            "description": description,
            "required_inputs": required_inputs,
            "when_to_use": when_to_use,
            "do_not_use": do_not_use,
            "md_path": md_path,
            "executor_model": spec.get("executor_model", ""),
        }
    return catalog


def build_selected_skill_catalog(selected_skills: List[Dict[str, Any]]) -> str:
    """
    selected_skills: list of router outputs, each item includes at least:
      - skill_name
      - score
      - reason
    """
    if not selected_skills:
        return (
            "Selected Skills:\n"
            "No suitable skill was matched for the current query and inputs.\n"
            "You should fall back to raw tools directly when needed."
        )

    catalog = load_skill_catalog()
    lines = [
        "Selected Skills:",
        "Use SkillExecutor with one of the following skill names when appropriate.",
        "",
    ]

    for idx, item in enumerate(selected_skills, start=1):
        skill_name = item["skill_name"]
        score = item.get("score", 0.0)
        reasons = item.get("reasons", [])
        meta = catalog.get(skill_name, {})

        lines.append(f"{idx}. {skill_name}")
        if meta.get("description"):
            lines.append(f"   Description: {meta['description']}")
        if meta.get("required_inputs"):
            lines.append(f"   Required inputs: {', '.join(meta['required_inputs'])}")
        if meta.get("when_to_use"):
            lines.append("   Use when:")
            for cond in meta["when_to_use"][:3]:
                lines.append(f"   - {cond}")
        if reasons:
            lines.append(f"   Router reasons: {'; '.join(reasons)}")
        lines.append(f"   Router score: {score:.2f}")
        lines.append("")

    lines.append(
        "If none of the selected skills clearly fit the task, do not force SkillExecutor. "
        "Use the raw tools directly."
    )
    return "\n".join(lines)
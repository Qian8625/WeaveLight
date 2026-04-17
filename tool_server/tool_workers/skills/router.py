import os
import re
from typing import Dict, List, Any, Tuple

from tool_server.tool_workers.skills.registry import SKILL_REGISTRY
from tool_server.tool_workers.skills.router_rules import ROUTER_RULES


class SkillRouter:
    """
    Lightweight rule-based router for hybrid skills.

    Goals:
      - Recommend top-k candidate skills
      - Be conservative when confidence is low
      - Return [] when query and current inputs do not clearly match any skill
    """

    def __init__(
        self,
        min_score: float = 2.5,
        top_k: int = 3,
        score_margin: float = 0.5,
    ):
        self.min_score = min_score
        self.top_k = top_k
        self.score_margin = score_margin

    # -----------------------------
    # Public API
    # -----------------------------
    def route(
        self,
        user_query: str,
        input_registry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Returns:
            [
              {
                "skill_name": "...",
                "score": 5.5,
                "reasons": ["matched keyword: sar", ...]
              },
              ...
            ]

        Returns [] if no skill is confidently matched.
        """
        query = self._normalize_text(user_query)
        inferred_modalities = self._infer_modalities(query, input_registry)

        scored = []
        for skill_name in SKILL_REGISTRY.keys():
            score, reasons = self._score_skill(
                skill_name=skill_name,
                query=query,
                input_registry=input_registry,
                inferred_modalities=inferred_modalities,
            )
            scored.append(
                {
                    "skill_name": skill_name,
                    "score": score,
                    "reasons": reasons,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Conservative filtering
        if not scored:
            return []

        best_score = scored[0]["score"]
        if best_score < self.min_score:
            return []

        selected = []
        for item in scored:
            if item["score"] < self.min_score:
                continue
            if best_score - item["score"] > self.score_margin and len(selected) >= 1:
                break
            selected.append(item)
            if len(selected) >= self.top_k:
                break

        return selected

    # -----------------------------
    # Scoring
    # -----------------------------
    def _score_skill(
        self,
        skill_name: str,
        query: str,
        input_registry: Dict[str, Any],
        inferred_modalities: List[str],
    ) -> Tuple[float, List[str]]:
        rules = ROUTER_RULES.get(skill_name, {})
        score = 0.0
        reasons = []

        keywords = rules.get("keywords", [])
        negative_keywords = rules.get("negative_keywords", [])
        preferred_inputs = rules.get("preferred_inputs", [])
        required_modalities = rules.get("required_modalities", [])

        # 1) keyword matching
        for kw in keywords:
            if self._contains(query, kw):
                score += 1.2
                reasons.append(f"matched keyword: {kw}")

        # 2) negative keyword penalty
        for nkw in negative_keywords:
            if self._contains(query, nkw):
                score -= 1.5
                reasons.append(f"negative keyword: {nkw}")

        # 3) input preference bonus
        for inp in preferred_inputs:
            if input_registry.get(inp):
                score += 0.5
                reasons.append(f"preferred input available: {inp}")

        # 4) modality constraints / bonuses
        if required_modalities:
            matched_modalities = 0
            for mod in required_modalities:
                if mod in inferred_modalities:
                    matched_modalities += 1
            if matched_modalities == len(required_modalities):
                score += 1.5
                reasons.append(
                    f"required modalities matched: {', '.join(required_modalities)}"
                )
            elif matched_modalities > 0:
                score += 0.5
                reasons.append(
                    f"partial modality match: {', '.join(required_modalities)}"
                )
            else:
                score -= 1.0
                reasons.append(
                    f"missing expected modalities: {', '.join(required_modalities)}"
                )

        # 5) skill-specific structure bonuses
        score, bonus_reasons = self._apply_skill_specific_heuristics(
            skill_name, query, input_registry
        )
        reasons.extend(bonus_reasons)

        return round(score, 3), reasons

    def _apply_skill_specific_heuristics(
        self,
        skill_name: str,
        query: str,
        input_registry: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        bonus = 0.0
        reasons = []

        has_primary = bool(input_registry.get("primary_image"))
        has_t1 = bool(input_registry.get("time1_image"))
        has_t2 = bool(input_registry.get("time2_image"))

        # Single-image locate / measure
        if skill_name == "TargetLocateMeasureSkill":
            if any(self._contains(query, x) for x in ["distance", "area", "segment", "locate", "detect", "框选", "面积", "距离", "分割"]):
                bonus += 1.0
                reasons.append("single-image locate/measure intent")

        # SAR-specific
        elif skill_name == "SARTargetLocateMeasureSkill":
            if self._contains(query, "sar") or self._contains(query, "雷达"):
                bonus += 2.0
                reasons.append("SAR explicitly mentioned")

        # Attribute
        elif skill_name == "TargetAttributeSkill":
            if any(self._contains(query, x) for x in ["color", "attribute", "same", "different", "颜色", "属性", "是否相同", "比较"]):
                bonus += 1.0
                reasons.append("attribute comparison/filter intent")

        # Conditional count
        elif skill_name == "ConditionalCountSkill":
            if any(self._contains(query, x) for x in ["how many", "count", "docked", "parked", "多少", "统计", "靠泊", "停放"]):
                bonus += 1.2
                reasons.append("conditional counting intent")

        # Cross-modal
        elif skill_name == "MultConfirmSkill":
            if any(self._contains(query, x) for x in ["rgb", "sar", "cross-modal", "fusion", "确认", "融合", "两模态"]):
                bonus += 1.8
                reasons.append("cross-modal intent")
            if has_primary and has_t1:
                bonus += 0.5
                reasons.append("two likely modality inputs available")

        # Change
        elif skill_name == "ChangeSummarySkill":
            if has_t1 and has_t2:
                bonus += 1.5
                reasons.append("bi-temporal inputs available")
            if any(self._contains(query, x) for x in ["change", "before and after", "新增", "消失", "变化", "两期", "前后"]):
                bonus += 1.8
                reasons.append("change-analysis intent")

        # GeoTIFF POI explore
        elif skill_name == "GeoTIFFPoiExploreSkill":
            if has_primary and self._looks_like_geotiff(input_registry.get("primary_image")):
                bonus += 1.5
                reasons.append("GeoTIFF input detected")
            if any(self._contains(query, x) for x in ["poi", "hospital", "museum", "mall", "周边", "是否存在", "医院", "博物馆", "商场"]):
                bonus += 1.8
                reasons.append("POI exploration intent")

        # GeoTIFF POI distance
        elif skill_name == "GeoTIFFPoiDistanceSkill":
            if has_primary and self._looks_like_geotiff(input_registry.get("primary_image")):
                bonus += 1.5
                reasons.append("GeoTIFF input detected")
            if any(self._contains(query, x) for x in ["nearest", "closest", "poi distance", "最近距离", "poi距离", "两类poi"]):
                bonus += 2.0
                reasons.append("POI distance intent")

        return bonus, reasons

    # -----------------------------
    # Modality inference
    # -----------------------------
    def _infer_modalities(
        self,
        query: str,
        input_registry: Dict[str, Any],
    ) -> List[str]:
        modalities = set()

        if self._contains(query, "sar") or self._contains(query, "雷达"):
            modalities.add("sar")
        if self._contains(query, "rgb") or self._contains(query, "optical") or self._contains(query, "光学"):
            modalities.add("rgb")
        if self._contains(query, "geotiff") or self._contains(query, "poi") or self._contains(query, "兴趣点"):
            modalities.add("geotiff")

        primary = input_registry.get("primary_image")
        if primary and self._looks_like_geotiff(primary):
            modalities.add("geotiff")

        if input_registry.get("time1_image") and input_registry.get("time2_image"):
            modalities.add("time_series")

        return sorted(list(modalities))

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _contains(text: str, keyword: str) -> bool:
        return keyword.lower() in text

    @staticmethod
    def _looks_like_geotiff(path: Any) -> bool:
        if not isinstance(path, str):
            return False
        return path.lower().endswith((".tif", ".tiff"))


def route_skills(
    user_query: str,
    input_registry: Dict[str, Any],
    top_k: int = 3,
    min_score: float = 2.5,
    score_margin: float = 0.5,
) -> List[Dict[str, Any]]:
    router = SkillRouter(
        min_score=min_score,
        top_k=top_k,
        score_margin=score_margin,
    )
    return router.route(user_query=user_query, input_registry=input_registry)
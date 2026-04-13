import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"TimeSeriesAssetRetriever_worker_{worker_id}.log")


class TimeSeriesAssetRetrieverWorker(BaseToolWorker):
    REQUIRED_COLUMNS = ["Time", "Image_Path", "Metadata_Path"]

    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="TimeSeriesAssetRetriever",
        limit_model_concurrency=3,
        host="0.0.0.0",
        port=None,
        excel_catalog_path="data_catalog.xlsx",
        count_tool_name="CountGivenObject",
        describe_tool_name="RegionAttributeDescription",
        detect_tool_name="TextToBbox",
        model_semaphore=None,
        image_base_dir=None,
        metadata_base_dir=None, 
        wait_timeout=120.0,
        task_timeout=120.0,
    ):
        self.excel_catalog_path = Path(excel_catalog_path).expanduser().resolve()
        self.catalog_df: Optional[pd.DataFrame] = None
        self.catalog_mtime: Optional[float] = None

        self.count_tool_name = count_tool_name
        self.describe_tool_name = describe_tool_name
        self.detect_tool_name = detect_tool_name
        self.tool_addr_cache: Dict[str, str] = {}

        self.image_base_dir = Path(image_base_dir).expanduser().resolve() if image_base_dir else None
        self.metadata_base_dir = Path(metadata_base_dir).expanduser().resolve() if metadata_base_dir else None

        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    # -----------------------------
    # Init / catalog loading
    # -----------------------------
    def init_model(self):
        self._load_catalog()
        logger.info(f"{self.model_name} initialized successfully.")

    def _normalize_path(self, x: Any, base_dir: Optional[Path]) -> Optional[str]:
        if pd.isna(x):
            return None

        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None

        p = Path(s).expanduser()

        if p.is_absolute():
            return str(p.resolve())

        if base_dir is not None:
            return str((base_dir / p).resolve())

        return str((self.excel_catalog_path.parent / p).resolve())

    def _parse_time_value(self, value: Any) -> Tuple[pd.Timestamp, str]:
        """
        Parse a catalog time value robustly.

        Returns:
            (timestamp, precision)
        precision in {"month", "day", "datetime", "unknown"}
        """
        if pd.isna(value):
            return pd.NaT, "unknown"

        # already datetime-like
        if isinstance(value, pd.Timestamp):
            return value, "datetime"

        # Excel / numeric-like handling
        s = str(value).strip()

        # handle strings like "20230624.0"
        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]

        # YYYYMMDD
        if re.fullmatch(r"\d{8}", s):
            ts = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            return ts, "day"

        # YYYYMM
        if re.fullmatch(r"\d{6}", s):
            ts = pd.to_datetime(s, format="%Y%m", errors="coerce")
            return ts, "month"

        # YYYY-MM-DD
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            ts = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
            return ts, "day"

        # YYYY-MM
        if re.fullmatch(r"\d{4}-\d{2}", s):
            ts = pd.to_datetime(s, format="%Y-%m", errors="coerce")
            return ts, "month"

        # fallback to pandas general parser
        ts = pd.to_datetime(s, errors="coerce")
        return ts, "datetime" if pd.notna(ts) else "unknown"

    def _load_catalog(self):
        if not self.excel_catalog_path.exists():
            raise FileNotFoundError(f"Excel catalog not found: {self.excel_catalog_path}")

        df = pd.read_excel(self.excel_catalog_path)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in Excel: {missing}")

        df["Time_raw"] = df["Time"].astype(str).str.strip()

        parsed = df["Time"].map(self._parse_time_value)
        df["Time_dt"] = parsed.map(lambda x: x[0])
        df["Time_precision"] = parsed.map(lambda x: x[1])

        invalid_count = int(df["Time_dt"].isna().sum())
        if invalid_count > 0:
            logger.warning(f"{invalid_count} rows have invalid Time and will be ignored.")

        df = df.dropna(subset=["Time_dt"]).copy()

        df["Image_Path"] = df["Image_Path"].map(lambda x: self._normalize_path(x, self.image_base_dir))
        df["Metadata_Path"] = df["Metadata_Path"].map(lambda x: self._normalize_path(x, self.metadata_base_dir))

        df["image_exists"] = df["Image_Path"].map(lambda x: Path(x).exists() if x else False)
        df["metadata_exists"] = df["Metadata_Path"].map(lambda x: Path(x).exists() if x else False)

        df = df.sort_values("Time_dt").reset_index(drop=True)

        self.catalog_df = df
        self.catalog_mtime = self.excel_catalog_path.stat().st_mtime
        logger.info(f"Loaded {len(df)} valid records from {self.excel_catalog_path}")

    def _maybe_reload_catalog(self):
        try:
            mtime = self.excel_catalog_path.stat().st_mtime
            if self.catalog_mtime is None or mtime != self.catalog_mtime:
                logger.info("Catalog file changed. Reloading...")
                self._load_catalog()
        except Exception as e:
            logger.warning(f"Failed to reload catalog: {e}")

    # -----------------------------
    # Time query
    # -----------------------------
    def _infer_match_mode(self, time_query: str, end_time: Optional[str], match_mode: str) -> str:
        if match_mode != "auto":
            return match_mode

        if end_time:
            return "range"

        # compact formats
        if re.fullmatch(r"\d{8}", time_query):
            return "day"
        if re.fullmatch(r"\d{6}", time_query):
            return "month"

        # standard formats
        if re.fullmatch(r"\d{4}-\d{2}$", time_query):
            return "month"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}$", time_query):
            return "day"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$", time_query):
            return "exact"

        return "nearest"

    def _parse_query_time(self, value: str) -> pd.Timestamp:
        ts, _ = self._parse_time_value(value)
        if pd.isna(ts):
            raise ValueError(f"Invalid time query: {value}")
        return ts

    def _query_by_time(
        self,
        time_query: str,
        match_mode: str,
        end_time: Optional[str],
        top_k: int,
    ) -> pd.DataFrame:
        if self.catalog_df is None:
            raise ValueError("Catalog is not loaded")

        if top_k <= 0:
            raise ValueError("top_k must be >= 1")

        df = self.catalog_df
        q = self._parse_query_time(time_query)

        if match_mode == "exact":
            matched = df[df["Time_dt"] == q]

        elif match_mode == "day":
            start = pd.Timestamp(q).normalize()
            end = start + pd.Timedelta(days=1)
            matched = df[(df["Time_dt"] >= start) & (df["Time_dt"] < end)]

        elif match_mode == "month":
            start = pd.Timestamp(q).to_period("M").start_time
            end = (pd.Timestamp(q).to_period("M") + 1).start_time
            matched = df[(df["Time_dt"] >= start) & (df["Time_dt"] < end)]

        elif match_mode == "range":
            if not end_time:
                raise ValueError("end_time is required when match_mode='range'")
            q_end = self._parse_query_time(end_time)
            if q_end < q:
                raise ValueError("end_time must be >= time_query")
            matched = df[(df["Time_dt"] >= q) & (df["Time_dt"] <= q_end)]

        elif match_mode == "nearest":
            temp = df.copy()
            temp["time_delta_abs"] = (temp["Time_dt"] - q).abs()
            matched = temp.sort_values("time_delta_abs").head(top_k)

        else:
            raise ValueError(f"Unsupported match_mode: {match_mode}")

        matched = matched.sort_values("Time_dt").head(top_k).copy()
        return matched

    # -----------------------------
    # Tool routing
    # -----------------------------
    def _resolve_tool_addr(self, tool_name: str) -> str:
        cached = self.tool_addr_cache.get(tool_name)
        if cached:
            return cached

        try:
            ret = requests.post(
                self.controller_addr + "/get_worker_address",
                json={"model": tool_name},
                timeout=5,
            )
            addr = (ret.json().get("address") or "").strip()
            if not addr:
                raise RuntimeError(f"No worker address returned for tool '{tool_name}'")
            self.tool_addr_cache[tool_name] = addr
            logger.info(f"Resolved tool {tool_name} to {addr}")
            return addr
        except Exception as e:
            raise RuntimeError(f"Failed to resolve tool {tool_name}: {e}")

    def _call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_addr = self._resolve_tool_addr(tool_name)
        resp = requests.post(
            tool_addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=60,
        )
        return resp.json()

    # -----------------------------
    # Optional post analysis
    # -----------------------------
    def _run_optional_analysis(
        self,
        image_path: Optional[str],
        post_action: str,
        target: Optional[str],
        attribute: Optional[str],
        top1: bool,
    ) -> Dict[str, Any]:
        if post_action == "none":
            return {
                "enabled": False,
                "action": "none",
            }

        if not image_path or not Path(image_path).exists():
            return {
                "enabled": True,
                "action": post_action,
                "status": "failed",
                "reason": "image_not_found",
            }

        try:
            if post_action == "count":
                if not target:
                    return {
                        "enabled": True,
                        "action": "count",
                        "status": "failed",
                        "reason": "target_required_for_count",
                    }
                result = self._call_tool(
                    self.count_tool_name,
                    {"image": image_path, "text": target},
                )
                return {
                    "enabled": True,
                    "action": "count",
                    "status": "success" if result.get("error_code") == 0 else "failed",
                    "tool_name": self.count_tool_name,
                    "raw_result": result.get("text", ""),
                    "error_code": result.get("error_code", 1),
                }

            elif post_action == "describe":
                desc_attr = attribute or target
                if not desc_attr:
                    return {
                        "enabled": True,
                        "action": "describe",
                        "status": "failed",
                        "reason": "attribute_or_target_required_for_describe",
                    }
                result = self._call_tool(
                    self.describe_tool_name,
                    {"image": image_path, "attribute": desc_attr},
                )
                return {
                    "enabled": True,
                    "action": "describe",
                    "status": "success" if result.get("error_code") == 0 else "failed",
                    "tool_name": self.describe_tool_name,
                    "raw_result": result.get("text", ""),
                    "error_code": result.get("error_code", 1),
                }

            elif post_action == "detect":
                if not target:
                    return {
                        "enabled": True,
                        "action": "detect",
                        "status": "failed",
                        "reason": "target_required_for_detect",
                    }
                result = self._call_tool(
                    self.detect_tool_name,
                    {"image": image_path, "text": target, "top1": top1},
                )
                return {
                    "enabled": True,
                    "action": "detect",
                    "status": "success" if result.get("error_code") == 0 else "failed",
                    "tool_name": self.detect_tool_name,
                    "raw_result": result.get("text", ""),
                    "error_code": result.get("error_code", 1),
                }

            else:
                return {
                    "enabled": True,
                    "action": post_action,
                    "status": "failed",
                    "reason": f"unsupported_post_action: {post_action}",
                }

        except Exception as e:
            return {
                "enabled": True,
                "action": post_action,
                "status": "failed",
                "reason": str(e),
            }

    # -----------------------------
    # Main
    # -----------------------------
    def generate(self, params):
        if "time_query" not in params:
            return {"text": "Missing required parameter: 'time_query'.", "error_code": 2}

        self._maybe_reload_catalog()

        if self.catalog_df is None:
            return {"text": "Catalog is not loaded.", "error_code": 1}

        time_query = str(params["time_query"]).strip()
        match_mode = str(params.get("match_mode", "auto")).strip().lower()
        end_time = params.get("end_time")
        top_k = int(params.get("top_k", 3))
        only_existing = bool(params.get("only_existing", False))

        post_action = str(params.get("post_action", "none")).strip().lower()
        target = params.get("target")
        target = str(target).strip() if target is not None else None
        attribute = params.get("attribute")
        attribute = str(attribute).strip() if attribute is not None else None
        top1 = bool(params.get("top1", False))

        try:
            match_mode = self._infer_match_mode(time_query, end_time, match_mode)
            matched_df = self._query_by_time(time_query, match_mode, end_time, top_k)

            if only_existing:
                matched_df = matched_df[matched_df["image_exists"] == True].copy()

            if matched_df.empty:
                result = {
                    "message": f"No records found for time_query='{time_query}'.",
                    "time_query": time_query,
                    "match_mode": match_mode,
                    "only_existing": only_existing,
                    "total_images": 0,
                    "post_action": post_action,
                    "assets": [],
                }
                return {
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                    "error_code": 0,
                }

            assets: List[Dict[str, Any]] = []
            analysis_success = 0

            for idx, (_, row) in enumerate(matched_df.iterrows(), start=1):
                image_id = chr(ord("a") + idx - 1) if idx <= 26 else f"img_{idx}"

                image_path = row["Image_Path"] if row["Image_Path"] else None
                metadata_path = row["Metadata_Path"] if row["Metadata_Path"] else None

                asset = {
                    "image_id": image_id,
                    "time": row["Time_raw"],
                    "time_normalized": row["Time_dt"].isoformat(),
                    "time_precision": row.get("Time_precision", "unknown"),
                    "image_path": image_path,
                    "metadata_path": metadata_path,
                    "image_exists": bool(row["image_exists"]),
                    "metadata_exists": bool(row["metadata_exists"]),
                }

                analysis = self._run_optional_analysis(
                    image_path=image_path,
                    post_action=post_action,
                    target=target,
                    attribute=attribute,
                    top1=top1,
                )
                asset["analysis"] = analysis

                if analysis.get("status") == "success":
                    analysis_success += 1

                assets.append(asset)

            result = {
                "message": f"Retrieved {len(assets)} asset(s).",
                "time_query": time_query,
                "match_mode": match_mode,
                "only_existing": only_existing,
                "total_images": len(assets),
                "post_action": post_action,
                "analysis_summary": {
                    "enabled": post_action != "none",
                    "successful_images": analysis_success if post_action != "none" else 0,
                    "failed_images": (len(assets) - analysis_success) if post_action != "none" else 0,
                    "target": target,
                    "attribute": attribute,
                },
                "assets": assets,
            }

            ret = {
                "text": json.dumps(result, ensure_ascii=False, indent=2),
                "error_code": 0,
            }

            # 单图命中时，兼容现有 agent 的 current_image / metadata_path 传递方式
            if len(assets) == 1 and assets[0]["image_exists"] and assets[0]["image_path"]:
                ret["image"] = assets[0]["image_path"]
            if len(assets) == 1 and assets[0]["metadata_exists"] and assets[0]["metadata_path"]:
                ret["metadata_path"] = assets[0]["metadata_path"]

            return ret

        except Exception as e:
            msg = f"{self.model_name} failed: {e}"
            logger.error(msg)
            return {"text": msg, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Retrieve one or more local image assets by time from an Excel catalog. "
                    "Optionally attach per-image analysis such as count, describe, or detect. "
                    "By default it only retrieves assets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_query": {
                            "type": "string",
                            "description": "Time query such as '20230624', '202306', '2021-05', '2021-05-09', or '2021-05-09 10:30:00'."
                        },
                        "match_mode": {
                            "type": "string",
                            "description": "One of: auto, exact, day, month, range, nearest. Default is auto."
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Required when match_mode='range'."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of matched assets to return. Default is 3."
                        },
                        "only_existing": {
                            "type": "boolean",
                            "description": "If true, only return assets whose image files actually exist. Default is false."
                        },
                        "post_action": {
                            "type": "string",
                            "description": "Optional follow-up analysis. One of: none, count, describe, detect. Default is none."
                        },
                        "target": {
                            "type": "string",
                            "description": "Target object or concept. Required for count and detect. Optional for describe."
                        },
                        "attribute": {
                            "type": "string",
                            "description": "Attribute name used in describe mode, e.g. 'roof color', 'water condition'."
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "Used in detect mode. If true, only keep the highest-confidence detection."
                        }
                    },
                    "required": ["time_query"]
                }
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20100)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="TimeSeriesAssetRetriever")
    parser.add_argument("--limit-model-concurrency", type=int, default=3)
    parser.add_argument("--excel-path", type=str, default="data_catalog.xlsx")
    parser.add_argument("--count-tool-name", type=str, default="CountGivenObject")
    parser.add_argument("--describe-tool-name", type=str, default="RegionAttributeDescription")
    parser.add_argument("--detect-tool-name", type=str, default="TextToBbox")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--image-base-dir", type=str, default=None)
    parser.add_argument("--metadata-base-dir", type=str, default=None)
    args = parser.parse_args()

    worker = TimeSeriesAssetRetrieverWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        excel_catalog_path=args.excel_path,
        count_tool_name=args.count_tool_name,
        describe_tool_name=args.describe_tool_name,
        detect_tool_name=args.detect_tool_name,
        image_base_dir=args.image_base_dir,
        metadata_base_dir=args.metadata_base_dir,   
    )
    worker.run()
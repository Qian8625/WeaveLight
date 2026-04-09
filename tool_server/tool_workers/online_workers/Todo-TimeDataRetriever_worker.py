import uuid
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"TimeDataRetriever_worker_{worker_id}.log")


class TimeDataRetrieverWorker(BaseToolWorker):
    REQUIRED_COLUMNS = ["Time", "Image_Path", "Metadata_Path"]

    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="TimeDataRetriever",
        limit_model_concurrency=5,
        host="0.0.0.0",
        port=None,
        excel_catalog_path="data_catalog.xlsx",
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=30.0,
    ):
        self.excel_catalog_path = Path(excel_catalog_path).expanduser().resolve()
        self.catalog_df = None
        self.catalog_mtime = None

        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,   # model_path
            None,   # model_base
            model_name,
            False,  # load_8bit
            False,  # load_4bit
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        self._load_catalog()
        logger.info(f"{self.model_name} is ready.")

    def _load_catalog(self):
        if not self.excel_catalog_path.exists():
            raise FileNotFoundError(f"Excel catalog not found: {self.excel_catalog_path}")

        df = pd.read_excel(self.excel_catalog_path)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 保留原始字符串展示列
        df["Time_raw"] = df["Time"].astype(str)

        # 标准化时间列
        df["Time_dt"] = pd.to_datetime(df["Time"], errors="coerce")
        invalid_count = df["Time_dt"].isna().sum()
        if invalid_count > 0:
            logger.warning(f"{invalid_count} rows have invalid Time and will be ignored.")

        df = df.dropna(subset=["Time_dt"]).copy()

        # 路径标准化
        df["Image_Path"] = df["Image_Path"].astype(str).map(lambda x: str(Path(x).expanduser()))
        df["Metadata_Path"] = df["Metadata_Path"].astype(str).map(lambda x: str(Path(x).expanduser()))

        df["image_exists"] = df["Image_Path"].map(lambda x: Path(x).exists())
        df["metadata_exists"] = df["Metadata_Path"].map(lambda x: Path(x).exists())

        df = df.sort_values("Time_dt").reset_index(drop=True)

        self.catalog_df = df
        self.catalog_mtime = self.excel_catalog_path.stat().st_mtime

        logger.info(
            f"Loaded catalog from {self.excel_catalog_path} with {len(self.catalog_df)} valid rows."
        )

    def _maybe_reload_catalog(self):
        try:
            mtime = self.excel_catalog_path.stat().st_mtime
            if self.catalog_mtime is None or mtime != self.catalog_mtime:
                logger.info("Catalog file changed. Reloading...")
                self._load_catalog()
        except Exception as e:
            logger.warning(f"Failed to reload catalog: {e}")

    def _infer_match_mode(self, time_query: str, end_time: str | None, match_mode: str) -> str:
        if match_mode != "auto":
            return match_mode

        if end_time:
            return "range"
        if re.fullmatch(r"\d{4}-\d{2}$", time_query):
            return "month"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}$", time_query):
            return "day"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$", time_query):
            return "exact"
        return "nearest"

    def _query(self, time_query: str, match_mode: str, end_time: str | None, top_k: int):
        df = self.catalog_df
        q = pd.to_datetime(time_query, errors="raise")

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
            q_end = pd.to_datetime(end_time, errors="raise")
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

    def generate(self, params):
        if "time_query" not in params:
            return {"text": "Missing required parameter: 'time_query'.", "error_code": 2}

        self._maybe_reload_catalog()

        if self.catalog_df is None:
            return {"text": "Catalog is not loaded.", "error_code": 1}

        time_query = str(params["time_query"]).strip()
        match_mode = str(params.get("match_mode", "auto")).strip().lower()
        end_time = params.get("end_time")
        top_k = int(params.get("top_k", 5))

        try:
            match_mode = self._infer_match_mode(time_query, end_time, match_mode)
            matched_df = self._query(time_query, match_mode, end_time, top_k)

            if matched_df.empty:
                result = {
                    "message": f"No records found for time_query='{time_query}'",
                    "match_mode": match_mode,
                    "count": 0,
                    "records": []
                }
                return {
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                    "error_code": 0
                }

            records = []
            for _, row in matched_df.iterrows():
                records.append({
                    "time": row["Time_raw"],
                    "time_normalized": row["Time_dt"].isoformat(),
                    "image_path": row["Image_Path"],
                    "metadata_path": row["Metadata_Path"],
                    "image_exists": bool(row["image_exists"]),
                    "metadata_exists": bool(row["metadata_exists"]),
                })

            result = {
                "message": f"Found {len(records)} record(s).",
                "match_mode": match_mode,
                "count": len(records),
                "records": records
            }

            ret = {
                "text": json.dumps(result, ensure_ascii=False, indent=2),
                "error_code": 0
            }

            # 只有唯一命中时，自动把 image 注入到 agent 上下文
            if len(records) == 1 and records[0]["image_exists"]:
                ret["image"] = records[0]["image_path"]

            # metadata 先返回顶层，后续如果你扩展 agent state，可以自动传递
            if len(records) == 1 and records[0]["metadata_exists"]:
                ret["metadata_path"] = records[0]["metadata_path"]

            return ret

        except Exception as e:
            error_msg = f"TimeDataRetriever failed: {e}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Retrieve local image files and metadata files from a temporal catalog using a time query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_query": {
                            "type": "string",
                            "description": "Time query such as '2021-05', '2021-05-09', or '2021-05-09 10:30:00'."
                        },
                        "match_mode": {
                            "type": "string",
                            "description": "One of: auto, exact, day, month, range, nearest."
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Required when match_mode='range'."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of records to return. Default is 5."
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
    parser.add_argument("--model-name", type=str, default="TimeDataRetriever")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--excel-path", type=str, default="data_catalog.xlsx")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = TimeDataRetrieverWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        excel_catalog_path=args.excel_path,
    )
    worker.run()
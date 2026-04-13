import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import rasterio
import torch

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"CloudRemoval_worker_{worker_id}.log")


def load_image(path: str) -> Tuple[np.ndarray, Optional[dict], str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = file_path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        return arr, None, ext

    if ext in [".tif", ".tiff"]:
        with rasterio.open(file_path) as src:
            profile = src.profile.copy()
            arr = src.read().astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            return arr, profile, ext

    raise ValueError("Only PNG/JPG/JPEG/TIF/TIFF inputs are supported.")


def load_single_band_image(path: str) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = file_path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file_path).convert("L")
        return np.asarray(img).astype(np.float32)[None, ...]

    if ext in [".tif", ".tiff"]:
        with rasterio.open(file_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            return arr[None, ...]

    raise ValueError("Only PNG/JPG/JPEG/TIF/TIFF inputs are supported for the auxiliary NIR image.")


def resolve_output_path(base_dir: Path, requested: str, input_path: str, suffix: str) -> Path:
    if requested:
        output = Path(requested)
        if not output.is_absolute():
            output = base_dir / output
    else:
        output = base_dir / f"{Path(input_path).stem}_cloud_removed"
    return output.with_suffix(suffix)


def build_preview_path(primary_output_path: Path) -> Path:
    return primary_output_path.with_name(f"{primary_output_path.stem}_preview.png")


def robust_rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for ch in range(rgb.shape[0]):
        band = rgb[ch]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        low = np.percentile(valid, 1.0)
        high = np.percentile(valid, 99.0)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            scaled = np.clip(band, 0.0, 1.0)
        else:
            scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        out[ch] = (scaled * 255.0).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))


def save_preview_png(output_path: Path, rgb_uint8: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(output_path)


def save_float_tiff(output_path: Path, arr: np.ndarray, profile: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_out = profile.copy()
    profile_out.update(
        dtype="float32",
        count=int(arr.shape[0]),
        height=int(arr.shape[1]),
        width=int(arr.shape[2]),
    )
    with rasterio.open(output_path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32))


def pad_to_multiple(arr: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    if multiple <= 1:
        return arr, (arr.shape[1], arr.shape[2])

    h, w = arr.shape[1], arr.shape[2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, (h, w)

    pad_mode = "reflect" if h > 1 and w > 1 else "edge"
    padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)
    return padded, (h, w)


def crop_to_shape(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return arr[:, :h, :w]


def process_optical(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] != expected_channels:
        raise ValueError(
            f"Optical image channel mismatch: expected {expected_channels}, got {arr.shape[0]}."
        )

    if expected_channels == 13:
        arr = np.clip(arr, 0.0, 10000.0)
        return arr / 10000.0

    if expected_channels == 3:
        if np.nanmax(arr) > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    if expected_channels == 4:
        if np.nanmax(arr) > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    raise ValueError(
        f"Unsupported optical channel count for current worker implementation: {expected_channels}."
    )


def synthesize_pseudo_nir(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if rgb.shape[0] != 3:
        raise ValueError(f"Pseudo-NIR synthesis expects 3 RGB channels, got {rgb.shape[0]}.")
    # Testing-only approximation when no real NIR is available.
    nir = 0.5 * rgb[0] + 0.3 * rgb[1] + 0.2 * rgb[2]
    return nir[None, ...]


def build_cuhk_input(optical_raw: np.ndarray, nir_raw: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    if optical_raw.shape[0] == 4:
        return optical_raw, False

    if optical_raw.shape[0] != 3:
        raise ValueError(
            f"CUHK backend expects a 4-band RGB+NIR input, or a 3-band RGB image plus optional nir_image. "
            f"Got {optical_raw.shape[0]} channels."
        )

    if nir_raw is None:
        return np.concatenate([optical_raw, synthesize_pseudo_nir(optical_raw)], axis=0), True

    if nir_raw.shape[0] != 1:
        raise ValueError(f"nir_image must provide exactly 1 band, got {nir_raw.shape[0]}.")

    return np.concatenate([optical_raw, nir_raw], axis=0), False


def process_sar(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] != expected_channels:
        raise ValueError(f"SAR image channel mismatch: expected {expected_channels}, got {arr.shape[0]}.")

    if expected_channels == 2:
        arr = np.clip(arr, -25.0, 0.0)
        return (arr + 25.0) / 25.0

    raise ValueError(
        f"Unsupported SAR channel count for current worker implementation: {expected_channels}."
    )


class CloudRemovalWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="CloudRemoval",
        emrdm_root=None,
        config_path=None,
        model_path=None,
        device="auto",
        save_path=None,
        pad_multiple=16,
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=300.0,
    ):
        self.emrdm_root = emrdm_root
        self.config_path = config_path
        self.fixed_model_path = model_path
        self.save_path = save_path
        self.pad_multiple = int(pad_multiple)
        self.init_error = None
        self.backend_name = "unknown"
        self.expected_optical_channels = None
        self.expected_sar_channels = 0
        self.conditioner_input_keys = []
        self.model = None
        self.uses_pseudo_nir = False

        if device == "auto":
            self.runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.runtime_device = torch.device(device)

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
            str(self.runtime_device),
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def _get_save_dir(self) -> Path:
        if self.save_path and os.path.isdir(self.save_path):
            return Path(self.save_path)

        if self.save_path and not os.path.isdir(self.save_path):
            logger.warning(f"Invalid save_path '{self.save_path}', fallback to ./tools_output")

        save_dir = Path(os.getcwd()) / "tools_output"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _append_emrdm_path(self):
        emrdm_root = Path(self.emrdm_root).resolve()
        if not emrdm_root.exists():
            raise FileNotFoundError(f"EMRDM root not found: {emrdm_root}")
        emrdm_root_str = str(emrdm_root)
        if emrdm_root_str not in sys.path:
            sys.path.insert(0, emrdm_root_str)

    def init_model(self):
        try:
            if not self.emrdm_root:
                raise ValueError("emrdm_root must be provided when starting CloudRemoval worker.")
            if not self.config_path:
                raise ValueError("config_path must be provided when starting CloudRemoval worker.")
            if not self.fixed_model_path:
                raise ValueError("model_path must be provided when starting CloudRemoval worker.")

            self._append_emrdm_path()
            from omegaconf import OmegaConf
            from sgm.util import load_model_from_config

            config_path = Path(self.config_path)
            model_path = Path(self.fixed_model_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {model_path}")

            logger.info(f"Loading EMRDM config from {config_path}")
            logger.info(f"Loading EMRDM checkpoint from {model_path}")
            logger.info(f"Using device: {self.runtime_device}")

            config = OmegaConf.load(config_path)
            model = load_model_from_config(config, str(model_path), verbose=False, freeze=False)
            model = model.to(self.runtime_device)
            model.eval()
            if hasattr(model, "sampler") and hasattr(model.sampler, "device"):
                model.sampler.device = str(self.runtime_device)
            if hasattr(model, "ideal_sampler") and model.ideal_sampler is not None and hasattr(model.ideal_sampler, "device"):
                model.ideal_sampler.device = str(self.runtime_device)

            network_params = config.model.params.network_config.params
            optical_channels = int(network_params.out_channels)
            model_input_channels = int(network_params.in_channels)
            concat_condition_channels = max(model_input_channels - optical_channels, 0)

            conditioner_keys = [embedder.input_key for embedder in model.conditioner.embedders]
            sar_channels = 0
            if "S1S2" in conditioner_keys:
                sar_channels = max(concat_condition_channels - optical_channels, 0)

            self.model = model
            self.expected_optical_channels = optical_channels
            self.expected_sar_channels = sar_channels
            self.conditioner_input_keys = conditioner_keys
            self.backend_name = config_path.stem

            logger.info(
                "CloudRemoval backend initialized: "
                f"backend={self.backend_name}, optical_channels={self.expected_optical_channels}, "
                f"sar_channels={self.expected_sar_channels}, conditioner_keys={self.conditioner_input_keys}"
            )
        except Exception as e:
            self.init_error = str(e)
            logger.exception(f"Failed to initialize CloudRemoval worker: {e}")

    def _build_batch(self, optical: np.ndarray, sar: Optional[np.ndarray]) -> Dict[str, torch.Tensor]:
        optical_tensor = torch.from_numpy(optical).unsqueeze(0).float().to(self.runtime_device)
        optical_tensor = optical_tensor * 2.0 - 1.0

        batch: Dict[str, torch.Tensor] = {
            self.model.mean_key: optical_tensor,
            "target": optical_tensor,
        }

        sar_tensor = None
        if sar is not None:
            sar_tensor = torch.from_numpy(sar).unsqueeze(0).float().to(self.runtime_device)
            sar_tensor = sar_tensor * 2.0 - 1.0
            batch["S1"] = sar_tensor

        if "S1S2" in self.conditioner_input_keys:
            if sar_tensor is None and self.expected_sar_channels > 0:
                raise ValueError(
                    "The configured EMRDM backend requires a SAR conditioning image. "
                    "Provide `sar_image`, or pass a combined GeoTIFF whose first bands are SAR "
                    f"({self.expected_sar_channels}) and remaining bands are optical ({self.expected_optical_channels})."
                )
            batch["S1S2"] = torch.cat([sar_tensor, optical_tensor], dim=1) if sar_tensor is not None else optical_tensor

        for key in self.conditioner_input_keys:
            if key not in batch:
                if key == self.model.mean_key:
                    batch[key] = optical_tensor
                else:
                    raise ValueError(f"Unsupported conditioning key for current worker implementation: {key}")

        return batch

    def _make_preview_rgb(self, output_tensor: torch.Tensor) -> np.ndarray:
        if hasattr(self.model, "to_rgb_func") and self.model.to_rgb_func is not None:
            preview_tensor = self.model.to_rgb_func(output_tensor.unsqueeze(0)).detach().cpu().float()[0]
        else:
            preview_tensor = output_tensor[:3].detach().cpu().float()

        if preview_tensor.ndim == 2:
            preview_tensor = preview_tensor.unsqueeze(0).repeat(3, 1, 1)
        elif preview_tensor.shape[0] == 1:
            preview_tensor = preview_tensor.repeat(3, 1, 1)

        if preview_tensor.shape[0] < 3:
            pad = 3 - preview_tensor.shape[0]
            preview_tensor = torch.cat([preview_tensor, preview_tensor[-1:].repeat(pad, 1, 1)], dim=0)

        return robust_rgb_to_uint8(preview_tensor[:3].numpy())

    def _prepare_input_tensor(
        self,
        optical_raw: np.ndarray,
        aux_nir_raw: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        if self.expected_optical_channels == 4:
            prepared, used_pseudo_nir = build_cuhk_input(optical_raw, aux_nir_raw)
            prepared = process_optical(prepared, self.expected_optical_channels)
            return prepared, used_pseudo_nir

        prepared = process_optical(optical_raw, self.expected_optical_channels)
        return prepared, False

    def generate(self, params):
        if self.init_error:
            return {
                "text": f"CloudRemoval backend is not ready: {self.init_error}",
                "error_code": 1,
            }

        if self.model is None:
            return {"text": "CloudRemoval model is not initialized.", "error_code": 1}

        if "image" not in params:
            msg = "Missing required parameter: 'image'."
            logger.error(msg)
            return {"text": msg, "error_code": 2}

        image_path = str(params["image"]).strip()
        nir_image_path = str(params.get("nir_image", "")).strip() or None
        legacy_aux_path = str(params.get("sar_image", "")).strip() or None
        output_path = str(params.get("output_path", "")).strip()

        try:
            optical_raw, optical_profile, optical_ext = load_image(image_path)

            aux_nir_raw = None
            aux_image_path = nir_image_path or legacy_aux_path
            if aux_image_path:
                aux_nir_raw = load_single_band_image(aux_image_path)

            sar_raw = None
            if aux_nir_raw is None and self.expected_sar_channels > 0:
                combined_channels = self.expected_sar_channels + self.expected_optical_channels
                if optical_raw.shape[0] == combined_channels:
                    logger.info(
                        "Detected combined SAR+optical GeoTIFF in `image`; "
                        f"using first {self.expected_sar_channels} bands as SAR "
                        f"and remaining {self.expected_optical_channels} bands as optical."
                    )
                    sar_raw = optical_raw[: self.expected_sar_channels]
                    optical_raw = optical_raw[self.expected_sar_channels :]

            optical, used_pseudo_nir = self._prepare_input_tensor(optical_raw, aux_nir_raw)
            sar = process_sar(sar_raw, self.expected_sar_channels) if sar_raw is not None else None

            optical_padded, original_hw = pad_to_multiple(optical, self.pad_multiple)
            sar_padded = None
            if sar is not None:
                sar_padded, _ = pad_to_multiple(sar, self.pad_multiple)

            batch = self._build_batch(optical_padded, sar_padded)
            mu = batch[self.model.mean_key]

            with torch.no_grad():
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    force_uc_zero_embeddings=[],
                )
                z_mu = self.model.encode_first_stage(mu)
                with self.model.ema_scope("CloudRemoval"):
                    samples, _ = self.model.sample(
                        c,
                        z_mu,
                        shape=z_mu.shape[1:],
                        uc=uc,
                        batch_size=z_mu.shape[0],
                    )
                    samples = self.model.decode_first_stage(samples)

            output_tensor = self.model.scale_01(samples).detach().cpu()[0]
            output_arr = crop_to_shape(output_tensor.numpy(), original_hw).astype(np.float32)

            save_dir = self._get_save_dir()
            prefer_tiff = optical_profile is not None or output_arr.shape[0] != 3
            primary_output_path = resolve_output_path(
                save_dir,
                output_path,
                image_path,
                ".tif" if prefer_tiff else ".png",
            )
            preview_output_path = build_preview_path(primary_output_path)

            if prefer_tiff:
                if optical_profile is None:
                    optical_profile = {
                        "driver": "GTiff",
                        "height": output_arr.shape[1],
                        "width": output_arr.shape[2],
                        "count": output_arr.shape[0],
                        "dtype": "float32",
                    }
                save_float_tiff(primary_output_path, output_arr, optical_profile)
            else:
                save_preview_png(primary_output_path, np.transpose(np.clip(output_arr, 0.0, 1.0) * 255.0, (1, 2, 0)).astype(np.uint8))

            preview_rgb = self._make_preview_rgb(torch.from_numpy(output_arr))
            save_preview_png(preview_output_path, preview_rgb)

            result_text = (
                f"Cloud removal completed with EMRDM backend '{self.backend_name}'. "
                f"Primary output saved to {primary_output_path}. Preview saved to {preview_output_path}."
            )
            if used_pseudo_nir:
                result_text += " RGB-only compatibility mode was used, so a pseudo-NIR channel was synthesized for testing."
            logger.info(result_text)

            return {
                "text": result_text,
                "error_code": 0,
                "image": str(preview_output_path),
                "preview": str(preview_output_path),
                "output_path": str(primary_output_path),
                "backend": self.backend_name,
                "optical_channels": self.expected_optical_channels,
                "sar_channels": self.expected_sar_channels,
                "used_pseudo_nir": used_pseudo_nir,
            }
        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.exception(error_msg)
            return {"text": error_msg, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Remove clouds from a remote-sensing image using an EMRDM backend. "
                    "Current configured backend uses the CUHK RGB+NIR checkpoint and expects either a 4-band RGB+NIR image, "
                    "or a normal RGB image with an optional single-band `nir_image`. If `nir_image` is omitted, "
                    "the worker can synthesize a testing-only pseudo-NIR channel so you can verify the environment and preview the effect. "
                    "The tool returns a primary output file and a PNG preview."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": (
                                "Path to the cloudy input image. For the current backend, this can be a normal RGB image, "
                                "a 3-band TIFF, or a 4-band RGB+NIR TIFF."
                            ),
                        },
                        "nir_image": {
                            "type": "string",
                            "description": (
                                "Optional single-band NIR image path used to build a true RGB+NIR input. "
                                "If omitted and the main image is RGB-only, a pseudo-NIR channel is synthesized for testing."
                            ),
                        },
                        "output_path": {
                            "type": "string",
                            "description": (
                                "Optional output file path. GeoTIFF is used for multi-band outputs; a preview PNG "
                                "is saved alongside it automatically."
                            ),
                        },
                    },
                    "required": ["image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20031)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CloudRemoval")
    parser.add_argument("--emrdm-root", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = CloudRemovalWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        emrdm_root=args.emrdm_root,
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
        save_path=args.save_path,
        pad_multiple=args.pad_multiple,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()

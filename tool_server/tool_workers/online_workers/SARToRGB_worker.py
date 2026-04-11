import argparse
import functools
import os
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from torch import nn

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARToRGB_worker_{worker_id}.log")


# =========================
# Network definition
# =========================
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        return lambda _x: Identity()
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def define_generator():
    return ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=get_norm_layer("instance"),
        use_dropout=False,
        n_blocks=9,
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    for key in ["state_dict", "netG", "generator", "model", "model_state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# =========================
# Image preprocessing
# =========================
def read_rgb_or_3ch_tiff(image_path: str) -> np.ndarray:
    """
    Return HxWx3 float32 image.
    Supports:
      - RGB image formats (png/jpg/jpeg)
      - 3-channel TIFF
    """
    path = Path(image_path)
    ext = path.suffix.lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(path).convert("RGB")
        return np.asarray(img).astype(np.float32)

    if ext in [".tif", ".tiff"]:
        with rasterio.open(path) as src:
            if src.count != 3:
                raise ValueError(f"TIFF must have exactly 3 bands, got {src.count}")
            arr = src.read([1, 2, 3]).astype(np.float32)  # C,H,W
            arr = np.transpose(arr, (1, 2, 0))            # H,W,C
            return arr

    raise ValueError("Only RGB PNG/JPG/JPEG and 3-channel TIFF are supported.")


def robust_normalize_rgb(arr: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    """
    Per-channel robust normalization to [0,1].
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        band = arr[..., c]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        low = np.percentile(valid, p_low)
        high = np.percentile(valid, p_high)
        if high <= low:
            continue
        band = np.clip(band, low, high)
        band = (band - low) / (high - low + 1e-8)
        band[~np.isfinite(band)] = 0.0
        out[..., c] = band
    return out


def resize_with_letterbox(img: np.ndarray, size: int, pad_value: float = 0.0):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), pad_value, dtype=np.float32)

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    meta = {
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "top": top,
        "left": left,
    }
    return canvas, meta


def recover_from_letterbox(img: np.ndarray, meta: dict) -> np.ndarray:
    top, left = meta["top"], meta["left"]
    new_h, new_w = meta["new_h"], meta["new_w"]
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]

    cropped = img[top:top + new_h, left:left + new_w]
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return restored


def preprocess_image(image_path: str, img_size: int):
    arr = read_rgb_or_3ch_tiff(image_path)
    arr = robust_normalize_rgb(arr)
    arr, meta = resize_with_letterbox(arr, img_size, pad_value=0.0)

    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
    tensor = tensor * 2.0 - 1.0   # [0,1] -> [-1,1]
    tensor = tensor.unsqueeze(0)
    return tensor, meta


def postprocess_image(tensor: torch.Tensor, meta: dict) -> Image.Image:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"postprocess_image expects torch.Tensor, got {type(tensor)}")

    image = tensor.squeeze(0).detach().cpu()
    image = (image * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    image = np.transpose(image.numpy(), (1, 2, 0))
    image = recover_from_letterbox(image, meta)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


# =========================
# Worker
# =========================
class SARToRGBWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="SARToRGB",
        model_path="/home/ubuntu/01_Code/OpenEarthAgent/models/Seg-CycleGAN/175_net_G_A.pth",
        img_size=256,
        device="auto",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
        save_path=None,
        model_semaphore=None,
        wait_timeout=120.0,
        task_timeout=120.0,
    ):
        self.fixed_model_path = model_path
        self.img_size = int(img_size)
        self.save_path = save_path

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

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
            str(self.device),
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        if not self.fixed_model_path:
            raise ValueError("A fixed model_path must be provided when starting the worker.")

        if not os.path.exists(self.fixed_model_path):
            raise FileNotFoundError(f"Model file not found: {self.fixed_model_path}")

        logger.info(f"Loading fixed SAR-to-RGB model from {self.fixed_model_path}")
        logger.info(f"Using device: {self.device}")

        model = define_generator().to(self.device)
        checkpoint = torch.load(self.fixed_model_path, map_location=self.device)
        state_dict = strip_module_prefix(extract_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        logger.info("Model loaded successfully.")

    def _get_save_dir(self) -> Path:
        if self.save_path and os.path.isdir(self.save_path):
            return Path(self.save_path)

        if self.save_path and not os.path.isdir(self.save_path):
            logger.warning(f"Invalid save_path '{self.save_path}', fallback to ./tools_output")

        save_dir = Path(os.getcwd()) / "tools_output"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _parse_request_params(self, params):
        """
        统一参数入口：
        - 推荐新接口：image
        - 兼容旧接口：input_path
        - output_path 可选
        """
        image_value = params.get("image", params.get("input_path"))
        if image_value is None or str(image_value).strip() == "":
            raise ValueError("Missing required parameter: image")

        input_path = str(image_value).strip()

        output_path = params.get("output_path", None)
        if output_path is not None:
            output_path = str(output_path).strip()
            if output_path == "":
                output_path = None

        return input_path, output_path

    def _build_output_path(self, input_path: str, output_path: str = None) -> Path:
        save_dir = self._get_save_dir()

        if output_path:
            final_output_path = save_dir / output_path
        else:
            input_stem = Path(input_path).stem
            if not input_stem:
                input_stem = f"sar_to_rgb_{uuid.uuid4().hex[:8]}"
            final_output_path = save_dir / f"{input_stem}_rgb.png"

        if final_output_path.suffix.lower() != ".png":
            final_output_path = final_output_path.with_suffix(".png")

        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        return final_output_path

    @torch.inference_mode()
    def generate(self, params):
        try:
            input_path, output_path = self._parse_request_params(params)
        except ValueError as e:
            error_msg = str(e)
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 2}

        if not os.path.exists(input_path):
            error_msg = f"Image not found: {input_path}"
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 3}

        if self.model is None:
            error_msg = "Model is not initialized."
            logger.error(error_msg)
            return {"text": error_msg, "error_code": 1}

        try:
            input_tensor, meta = preprocess_image(input_path, img_size=self.img_size)
            input_tensor = input_tensor.to(self.device, non_blocking=True)

            output_tensor = self.model(input_tensor)
            output_image = postprocess_image(output_tensor, meta)

            final_output_path = self._build_output_path(input_path, output_path)
            output_image.save(final_output_path)

            result_msg = f"SAR-to-RGB translation completed. Result saved at {final_output_path}"
            logger.info(result_msg)

            return {
                "text": result_msg,
                "error_code": 0,
                "image": str(final_output_path),
                "output_path": str(final_output_path),
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
                    "Translate an input SAR image into an RGB image using a fixed pretrained model. "
                    "Supports RGB PNG/JPG/JPEG input and 3-channel TIFF input. "
                    "The output is saved as a PNG image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": (
                                "Path to the input image. Supports RGB PNG/JPG/JPEG and 3-channel TIFF. "
                                "This is the recommended parameter name."
                            ),
                        },
                        "output_path": {
                            "type": "string",
                            "description": (
                                "Optional relative output path. If omitted, the worker auto-generates "
                                "a PNG filename under the save directory."
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
    parser.add_argument("--port", type=int, default=20102)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SARToRGB")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARToRGBWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        model_path=args.model_path,
        img_size=args.img_size,
        device=args.device,
        save_path=args.save_path,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
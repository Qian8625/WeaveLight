import os
import uuid
import argparse
import functools
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SARToRGBTool_Worker_{worker_id}.log")

TEMP_DIR = Path("./temp_results")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_path, img_size=256):
    """
    图像预处理：
    1. 读取 RGB 图像
    2. 缩放到固定尺寸
    3. 归一化到 [-1, 1]
    4. 增加 batch 维度
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)



def postprocess_image(tensor):
    """
    将模型输出张量转换为 PIL 图像。
    默认假设模型输出范围为 [-1, 1]。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"postprocess_image 输入必须是 torch.Tensor，当前为: {type(tensor)}")

    image = tensor.squeeze(0).detach().cpu()
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(image)



def extract_state_dict(checkpoint):
    """
    从 checkpoint 中提取 state_dict。
    兼容以下常见保存格式：
    - 直接保存的 state_dict
    - {"state_dict": ...}
    - {"netG": ...}
    - {"generator": ...}
    - {"model": ...}
    - {"model_state_dict": ...}
    """
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"不支持的 checkpoint 类型: {type(checkpoint)}，期望为 nn.Module 或 dict"
        )

    for key in ["state_dict", "netG", "generator", "model", "model_state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise KeyError("无法从 checkpoint 中提取 state_dict，请检查模型保存格式。")



def strip_module_prefix(state_dict):
    """
    去掉多卡训练保存时参数名前的 'module.' 前缀。
    """
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict 必须是 dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict



def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True,
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )
    elif norm_type == "none":
        def norm_layer(_):
            return Identity()
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")

    return norm_layer


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

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
        n_blocks=6,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
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



def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=None,
):
    """
    创建生成器。
    当前仅支持与你原始代码一致的 resnet_9blocks。
    """
    if gpu_ids is None:
        gpu_ids = []

    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
        )
    else:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized")

    return net



def load_model(model_path, device, network_type="Seg-CycleGAN"):
    """
    加载预训练生成器模型。
    """
    if network_type == "Seg-CycleGAN":
        netG = define_G(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG="resnet_9blocks",
            norm="instance",
            use_dropout=False,
            init_type="normal",
            init_gain=0.02,
        )
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")

    netG = netG.to(device)
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)
    netG.load_state_dict(state_dict, strict=True)

    netG.eval()
    return netG


class SARToRGBWorker(BaseToolWorker):
    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_name="sar_to_rgb",
        limit_model_concurrency=1,
        host="0.0.0.0",
        port=None,
    ):
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_name=model_name,
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port,
        )

    def init_model(self):
        """
        工具初始化。
        该工具无需在启动时预加载固定模型，真正的模型路径由调用参数提供。
        """
        logger.info(f"{self.model_name} 初始化成功！准备进行 SAR->RGB 图像翻译。")

    def generate(self, params):
        """
        核心执行逻辑。
        接收模型路径、输入图像路径和输出路径，执行 Seg-CycleGAN 推理并保存结果。
        """
        required_params = ["model_path", "image_path", "output_path"]
        for req in required_params:
            if req not in params:
                error_msg = f"Missing required parameter: '{req}'."
                logger.error(error_msg)
                return {"text": error_msg, "error_code": 2}

        model_path = params.get("model_path")
        image_path = params.get("image_path")
        output_path = params.get("output_path")

        device = params.get("device")
        img_size = int(params.get("img_size", 256))
        network_type = params.get("network_type", "Seg-CycleGAN")

        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                device = torch.device(device)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"输入图像不存在: {image_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            logger.info(f"Using device: {device}")
            logger.info(f"Start inference: model={model_path}, image={image_path}, img_size={img_size}")

            model = load_model(model_path, device, network_type)
            input_tensor = preprocess_image(image_path, img_size=img_size).to(device)

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_image = postprocess_image(output_tensor)

            final_output_path = TEMP_DIR / output_path
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(final_output_path)

            result_msg = f"Result saved at {final_output_path}"
            logger.info(result_msg)
            return {"text": result_msg, "error_code": 0}

        except Exception as e:
            error_msg = f"Error in {self.model_name}: {e}"
            logger.exception(error_msg)
            return {"text": error_msg, "error_code": 1}

    def get_tool_instruction(self):
        """
        返回工具说明文档，供上层系统组装到 LLM Prompt 中。
        """
        return {
            "name": self.model_name,
            "description": "SAR to RGB image translation tool based on Seg-CycleGAN. Load a trained generator checkpoint, run inference on an input SAR image, and save the translated RGB image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the trained Seg-CycleGAN generator checkpoint (.pth)."
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to the input SAR image file."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Relative output image path, e.g. 'question17/sar_to_rgb_result.png'."
                    },
                    "device": {
                        "type": "string",
                        "description": "Inference device, e.g. 'cuda', 'cuda:0', or 'cpu'. Default: auto detect."
                    },
                    "img_size": {
                        "type": "integer",
                        "description": "Resize the input image to (img_size, img_size) before inference. Default is 256."
                    },
                    "network_type": {
                        "type": "string",
                        "description": "Network type. Currently only supports 'Seg-CycleGAN'."
                    }
                },
                "required": ["model_path", "image_path", "output_path"]
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20101)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = SARToRGBWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
    )
    worker.run()

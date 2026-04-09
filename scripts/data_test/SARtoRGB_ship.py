import os
import functools
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

def seg_cyclegan_inference(model_path, image_path, device=None, **kwargs):
    """
    Seg-CycleGAN 本地推理

    参数:
        model_path (str): 模型路径
        image_path (str): 输入图像路径
        device (torch.device | str | None): 推理设备
        **kwargs: 预留参数

    返回:
        output_image: 后处理后的输出图像（通常为 PIL.Image）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"使用设备: {device}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图像不存在: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        print("加载单阶段模型: Seg-CycleGAN")

        # 1. 加载模型
        model = load_model(model_path, device, "Seg-CycleGAN")

        # 2. 预处理图像
        input_tensor = preprocess_image(image_path).to(device)

        # 3. 推理
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 4. 后处理
        output_image = postprocess_image(output_tensor)

        return output_image

    except Exception as e:
        print(f"Seg-CycleGAN 推理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def local_inference(model_path, image_path, device=None, **kwargs):
    """
    兼容旧调用方式的统一入口
    """
    return seg_cyclegan_inference(
        model_path=model_path,
        image_path=image_path,
        device=device,
        **kwargs
    )


def preprocess_image(image_path, img_size=256):
    """
    图像预处理
    将输入图像固定缩放到 (img_size, img_size)，并归一化到 [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, C, H, W]


def postprocess_image(tensor):
    """
    将模型输出张量转换为 PIL 图像
    假设模型输出范围为 [-1, 1]
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"postprocess_image 输入必须是 torch.Tensor，当前为: {type(tensor)}")

    image = tensor.squeeze(0).detach().cpu()   # [C, H, W]
    image = (image * 0.5 + 0.5).clamp(0, 1)    # 映射到 [0, 1]
    return transforms.ToPILImage()(image)


def load_model(model_path, device, network_type):
    """加载预训练生成器模型"""

    if network_type == "Seg-CycleGAN":
        netG = define_G(input_nc=3, output_nc=3, ngf=64, netG="resnet_9blocks", norm='instance', use_dropout=False, init_type='normal', init_gain=0.02)
    else:
        raise ValueError(f"加载与训练生成器模型出错,不支持得网络类型: {network_type}")
    

    netG = netG.to(device)

    print('loading the model from %s' % model_path)
                    
    state_dict = torch.load(model_path, map_location=str(device))
    netG.load_state_dict(state_dict)
            
    netG.eval()  # 设置为评估模式
    return netG


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
    创建生成器
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




def extract_state_dict(checkpoint):
    """
    从 checkpoint 中提取 state_dict
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"不支持的 checkpoint 类型: {type(checkpoint)}，"
            "期望为 nn.Module 或 dict"
        )

    # 常见字段名
    for key in ["state_dict", "netG", "generator", "model", "model_state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # 如果本身就是 state_dict（键值大概率是参数名 -> tensor）
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise KeyError("无法从 checkpoint 中提取 state_dict，请检查保存格式。")


def strip_module_prefix(state_dict):
    """
    去掉多卡训练保存时参数名前的 'module.' 前缀
    """
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict 必须是 dict")

    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def get_norm_layer(norm_type="instance"):
    """
    返回归一化层
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False
        )
    elif norm_type == "none":
        def norm_layer(_):
            return Identity()
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")

    return norm_layer


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator
    """

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

        # 下采样
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

        # ResNet blocks
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

        # 上采样
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


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResnetBlock(nn.Module):
    """
    ResNet Block
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        # 第一层卷积
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

        # 第二层卷积
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




if __name__ == "__main__":

    output = seg_cyclegan_inference("/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/CPK/Seg-CycleGAN/175_net_G_A.pth", "/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_test_1.png")
    output.save("/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_to_rgb_result_1.png")
    pass

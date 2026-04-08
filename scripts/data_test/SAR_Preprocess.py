import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy.ndimage import uniform_filter
import os
# ===================== 配置 =====================
INPUT_TIFF_PATH = "/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_test_1.png"

# 输出文件
OUTPUT_FILTERED_TIFF = "sar_filtered_gray.tif"
OUTPUT_PREVIEW_PNG = "sar_preview.png"

# Lee滤波参数
FILTER_WINDOW = 5
NOISE_SCALE = 1.2
CLAHE_CLIP_LIMIT = 1.2
CLAHE_TILE_GRID_SIZE = (4, 4)

# 是否忽略“无地理参考”警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ===================== RGB图像读取函数 =====================
def read_png_as_gray(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 PNG: {file_path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)

# ===================== 读取RGB TIFF并转灰度 =====================
def read_rgb_tiff_as_gray(file_path):
    """
    读取 3 波段 RGB TIFF，并转换为灰度图
    返回：
        gray_img: float32, shape(H, W)
        profile: rasterio profile
    """
    with rasterio.open(file_path) as src:
        profile = src.profile.copy()

        if src.count != 3:
            raise ValueError(f"当前文件有 {src.count} 个波段，不是 3 波段 RGB TIFF。")

        # 读取三个波段，rasterio读出形状为 (C, H, W)
        rgb = src.read([1, 2, 3]).astype(np.float32)

        # 判断是否为 RGB
        colorinterp = src.colorinterp
        print("=== 检测到 3 波段 RGB TIFF，转换为灰度图 ===")

        # 转灰度：使用标准亮度公式
        # 注意：rasterio返回顺序已经是 [R, G, B]
        gray_img = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

        # 处理 nodata
        nodata = src.nodata
        if nodata is not None:
            gray_img = np.where(gray_img == nodata, np.nan, gray_img)

    return gray_img.astype(np.float32), profile

# ===================== 保存TIFF =====================
def save_single_band_tiff(output_path, img, src_profile):
    """
    保存为单波段 float32 TIFF
    """
    out_profile = src_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw"
    )

    # 没有地理参考也没关系，照样可以保存 TIFF
    nodata_value = -9999.0
    img_save = np.where(np.isfinite(img), img, nodata_value).astype(np.float32)
    out_profile.update(nodata=nodata_value)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(img_save, 1)

def save_png(output_path, img_array):
    """
    保存 uint8 PNG
    """
    img_save = np.clip(img_array, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(output_path, img_save)
    if not ok:
        raise IOError(f"保存 PNG 失败: {output_path}")

# ===================== 图像工具函数 =====================
def normalize_to_uint8(img, lower_percent=1, upper_percent=99):
    """
    百分位拉伸到 0~255
    """
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    low = np.percentile(valid, lower_percent)
    high = np.percentile(valid, upper_percent)

    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)

    img_clip = np.clip(img, low, high)
    img_norm = (img_clip - low) / (high - low + 1e-8) * 255.0
    img_norm[~np.isfinite(img_norm)] = 0

    return img_norm.astype(np.uint8)

# ===================== Lee滤波 =====================
def lee_filter(img, win_size=5, noise_scale=1.2):
    """
    简化版 Lee 滤波
    适用于灰度图像去噪
    """
    img = img.astype(np.float32)

    # 有效像素掩膜
    valid_mask = np.isfinite(img).astype(np.float32)
    img_filled = np.where(np.isfinite(img), img, 0.0)

    # 局部统计
    local_count = uniform_filter(valid_mask, size=win_size) * (win_size ** 2)
    local_sum = uniform_filter(img_filled, size=win_size) * (win_size ** 2)
    local_sum_sq = uniform_filter(img_filled ** 2, size=win_size) * (win_size ** 2)

    local_count = np.maximum(local_count, 1.0)
    local_mean = local_sum / local_count
    local_var = np.maximum(local_sum_sq / local_count - local_mean ** 2, 0.0)

    # 简化噪声方差估计
    valid_var = local_var[np.isfinite(local_var)]
    noise_var = np.mean(valid_var) * noise_scale if valid_var.size > 0 else 0.0

    weights = local_var / (local_var + noise_var + 1e-8)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (img_filled - local_mean)
    filtered[valid_mask == 0] = np.nan

    return filtered.astype(np.float32)

# ===================== 图像增强 =====================
def image_enhancement(img, clip_limit=1.5, tile_grid_size=(6, 6)):
    """
    显示增强：百分位拉伸 + CLAHE
    """
    img_8u = normalize_to_uint8(img, lower_percent=1, upper_percent=99)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(img_8u)
    return enhanced.astype(np.float32)

# ===================== 主流程 =====================
def sar_preprocessing_rgb_tiff(
    input_path,
    output_filtered_tiff=OUTPUT_FILTERED_TIFF,
    output_preview_png=OUTPUT_PREVIEW_PNG,
    filter_window=FILTER_WINDOW,
    noise_scale=NOISE_SCALE,
    clahe_clip=CLAHE_CLIP_LIMIT,
    clahe_tile=CLAHE_TILE_GRID_SIZE
):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".png":
        gray_img = read_png_as_gray(input_path)
        profile = None
    elif ext in [".tif", ".tiff"]:
        gray_img, profile = read_rgb_tiff_as_gray(input_path)
    else:
        raise ValueError("仅支持 PNG / TIFF")

    print("=== 执行 Lee 滤波 ===")
    denoised_img = lee_filter(gray_img, win_size=filter_window, noise_scale=noise_scale)

    print("=== 执行图像增强（CLAHE）===")
    enhanced_img = image_enhancement(
        denoised_img,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tile
    )

    print("=== 保存滤波结果 ===")
    if profile is not None:
        save_single_band_tiff(output_filtered_tiff, denoised_img, profile)
    else:
        save_png("sar_filtered.png", normalize_to_uint8(denoised_img))

    return gray_img, denoised_img, enhanced_img

# ===================== 执行入口 =====================
if __name__ == "__main__":

    original_img, denoised_img, preview_img = sar_preprocessing_rgb_tiff(
        INPUT_TIFF_PATH,
        output_filtered_tiff=OUTPUT_FILTERED_TIFF,
        output_preview_png=OUTPUT_PREVIEW_PNG,
        filter_window=FILTER_WINDOW,
        noise_scale=NOISE_SCALE,
        clahe_clip=CLAHE_CLIP_LIMIT,
        clahe_tile=CLAHE_TILE_GRID_SIZE
    )
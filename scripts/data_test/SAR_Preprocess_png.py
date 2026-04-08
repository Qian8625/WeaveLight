import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# ===================== 配置 =====================
FILTER_WINDOW = 5
OUTPUT_PATH = "sar_preprocessed_png.png"

# ===================== 工具函数 =====================
def read_sar_png(file_path):
    """
    读取 PNG 灰度图
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {file_path}")

    # 若是彩色图，转灰度
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)

def save_png(output_path, img_array):
    """
    保存 uint8 PNG
    """
    img_save = np.clip(img_array, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(output_path, img_save)
    if not ok:
        raise IOError(f"保存失败: {output_path}")

def normalize_to_uint8(img, lower_percent=1, upper_percent=99):
    """
    百分位拉伸到 0~255，适合显示
    """
    low = np.percentile(img, lower_percent)
    high = np.percentile(img, upper_percent)

    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)

    img_clip = np.clip(img, low, high)
    img_norm = (img_clip - low) / (high - low) * 255.0
    return img_norm.astype(np.uint8)

# ===================== 核心算法 =====================
def lee_filter(img, win_size=5):
    """
    简化版 Lee 滤波
    适合做经验性 speckle 抑制；对 PNG 仅用于图像增强场景
    """
    img = img.astype(np.float32)

    img_mean = uniform_filter(img, size=win_size)
    img_sqr_mean = uniform_filter(img ** 2, size=win_size)

    local_var = np.maximum(img_sqr_mean - img_mean ** 2, 0.0)

    # 简化噪声方差估计
    noise_var = np.mean(local_var)

    weights = local_var / (local_var + noise_var + 1e-8)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = img_mean + weights * (img - img_mean)
    return filtered

def image_enhancement_for_display(img):
    """
    仅用于显示增强，不保留物理辐射意义
    """
    img_8u = normalize_to_uint8(img, lower_percent=1, upper_percent=99)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_8u)
    return enhanced.astype(np.float32)

# ===================== 主流程 =====================
def sar_preprocessing_png(input_path, output_path=OUTPUT_PATH, filter_window=FILTER_WINDOW):
    print("=== 读取 PNG SAR 图像 ===")
    sar_img = read_sar_png(input_path)

    print("=== 执行 Lee 滤波 ===")
    denoised_img = lee_filter(sar_img, win_size=filter_window)

    print("=== 执行显示增强（CLAHE）===")
    enhanced_img = image_enhancement_for_display(denoised_img)

    print("=== 保存结果 ===")
    save_png(output_path, enhanced_img)
    print(f"=== 处理完成，结果保存至: {output_path} ===")

    return sar_img, denoised_img, enhanced_img

# ===================== 可视化 =====================
def visualize_result(original, denoised, enhanced):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title("Original PNG")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(132)
    plt.title("Lee Filtered")
    plt.imshow(denoised, cmap="gray")
    plt.axis("off")

    plt.subplot(133)
    plt.title("Enhanced PNG")
    plt.imshow(enhanced, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ===================== 执行入口 =====================
if __name__ == "__main__":
    INPUT_SAR_PATH = "/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_test_1.png"

    original_img, denoised_img, enhanced_img = sar_preprocessing_png(INPUT_SAR_PATH)
    visualize_result(original_img, denoised_img, enhanced_img)
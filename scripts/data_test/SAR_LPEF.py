import cv2
import numpy as np
# from lpef_algorithm import LPEFAlgorithm, IFEFAlgorithm
from scipy.ndimage import uniform_filter, minimum_filter

def main():
    # 1. 读取SAR图像（灰度图）
    input_path = "/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_test_1.png"
    output_dir = "/home/ubuntu/01_Code/OpenEarthAgent/scripts/data_test/SAR/sar_test_1_process.png"

    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("错误：无法读取图像，请检查路径")
        return

    print(f"成功读取图像，尺寸: {img.shape}")

    # 2. 初始化LPEF算法
    # 参数说明：
    # - lee_window: LEE滤波窗口大小（建议7或9）
    # - k_sigma: IPFE阈值系数（建议1.2-2.0）
    # - ktn_k_values: KTN算法k值，默认[3, 6]对应R、B通道
    lpef = LPEFAlgorithm(
        lee_window=7,
        k_sigma=1.5,
        pfe_window=3,
        ktn_k_values=[3, 6]
    )

    # 3. 执行处理
    print("正在执行LPEF算法...")
    results = lpef.process(img)

    # 4. 保存结果
    cv2.imwrite(f"{output_dir}01_original.png", results['gray'])
    cv2.imwrite(f"{output_dir}02_lee_filter.png", results['lee'])
    cv2.imwrite(f"{output_dir}03_ipfe.png", results['ipfe'])
    cv2.imwrite(f"{output_dir}04_lpef_rgb.png", results['result'])

    print("LPEF处理完成！")
    print(f"结果保存在: {output_dir}")

    # 5. （可选）使用IFEF算法（含二次增强）
    # print("\n正在执行IFEF算法（含二次增强）...")
    # ifef = IFEFAlgorithm(
    #     lee_window=7,
    #     k_sigma=1.5,
    #     second_enhance=True
    # )
    # results_ifef = ifef.process(img)
    # cv2.imwrite(f"{output_dir}05_ifef_rgb.png", results_ifef['result'])
    # print("IFEF处理完成！")




class LPEFAlgorithm:
    """
    LPEF算法复现：基于LEE滤波去噪的改进峰值特征提取算法
    用于SAR图像增强，提升下游视觉检测任务性能
    """

    def __init__(self, lee_window=7, k_sigma=1.5, pfe_window=3, ktn_k_values=[3, 6]):
        self.lee_window = lee_window
        self.k_sigma = k_sigma
        self.pfe_window = pfe_window
        self.ktn_k_values = ktn_k_values

    def lee_filter(self, img):
        """LEE自适应滤波器（全局去噪）"""
        img = img.astype(np.float64)
        local_mean = uniform_filter(img, self.lee_window)
        local_mean_sq = uniform_filter(img**2, self.lee_window)
        local_var = np.maximum(local_mean_sq - local_mean**2, 1e-10)
        noise_var = np.var(img) * 0.1
        weight = np.maximum(0, local_var - noise_var) / local_var
        weight = np.clip(weight, 0, 1)
        filtered = local_mean + weight * (img - local_mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def improved_pfe(self, img):
        """
        改进的峰值特征提取算法(IPFE)

        根据PPT逻辑：
        if x > μ + kσ and min(aij - aN(i,j)) > σ: → 255
        elif x > μ + kσ or min(aij - aN(i,j)) > σ: → 127
        else: → 0
        """
        img = img.astype(np.float64)
        mu, sigma = np.mean(img), np.std(img)
        threshold = mu + self.k_sigma * sigma

        # 使用minimum_filter加速邻域最小值计算
        local_min = minimum_filter(img, size=self.pfe_window, mode='reflect')
        min_diff = img - local_min

        result = np.zeros_like(img, dtype=np.uint8)
        cond1 = img > threshold
        cond2 = min_diff > sigma

        result[cond1 & cond2] = 255      # P1: 强散射点
        result[(cond1 | cond2) & ~(cond1 & cond2)] = 127  # P2: 中等散射点

        return result

    def ktn_algorithm(self, img, k):
        """
        KTN算法：k倍非零均值截断归一化
        公式: I_norm = min(I, k * mean_nonzero(I)) / max(I) * 255
        """
        img = img.astype(np.float64)
        non_zero = img[img > 0]
        mean_nonzero = np.mean(non_zero) if len(non_zero) > 0 else 1e-10
        truncated = np.minimum(img, k * mean_nonzero)
        max_val = np.max(truncated)
        if max_val > 0:
            truncated = (truncated / max_val) * 255
        return truncated.astype(np.uint8)

    def pff_fusion(self, lee_img, ipfe_img):
        """
        PFF峰值特征融合：
        R = KTN(LEE, k=3), G = IPFE, B = KTN(LEE, k=6)
        """
        r = self.ktn_algorithm(lee_img, self.ktn_k_values[0])
        g = ipfe_img
        b = self.ktn_algorithm(lee_img, self.ktn_k_values[1])
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def process(self, input_img):
        """完整LPEF处理流程"""
        if len(input_img.shape) == 3:
            gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = input_img.copy()

        lee = self.lee_filter(gray)
        ipfe = self.improved_pfe(gray)
        result = self.pff_fusion(lee, ipfe)

        return {'gray': gray, 'lee': lee, 'ipfe': ipfe, 'result': result}

if __name__ == "__main__":
    main()  
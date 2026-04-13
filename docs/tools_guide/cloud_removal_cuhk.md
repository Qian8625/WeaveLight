# CloudRemoval CUHK 接入说明

## 1. 当前结论

当前 `CloudRemoval` 已经接入 `EMRDM` 的 `CUHK` checkpoint，并且可以在本仓库内完成本地推理调用。

但需要明确以下事实：

- 当前 `CUHK` checkpoint 不是纯 RGB 模型，而是 `RGB + NIR -> 4-channel output` 的模型。
- 现在仓库里为了兼容普通 RGB 图，新增了一个“测试模式”分支：
  - 当只传 `image` 且没有 `nir_image` 时，worker 会自动合成一个 pseudo-NIR。
  - 这个分支的目标是让工具链和环境先跑通，不代表结果质量可靠。
- 在本次联调环境里，CPU 上对 `tempcode/56.png` 的全尺寸 `512x512` 推理会因为 `natten/flex_attention` 内存占用过高而失败。
- 缩小到 `64x64` 和 `128x128` 时可以跑通，说明当前代码链路、checkpoint 加载链路和采样链路本身是通的。

## 2. 当前版本的主要问题

### 2.1 模型语义问题

当前使用的配置文件：

- `models/cpk/cuhk-20260410T131636Z-3-001/cuhk/configs/2024-11-09T08-59-25-project.yaml`

关键点：

- `in_channels: 8`
- `out_channels: 4`
- `mean_key: cond_image`
- `to_rgb_config: sgm.util.nir_to_rgb`

这说明当前模型不是“普通三通道 RGB 去云模型”，而是针对 `RGB+NIR` 条件输入训练出来的模型。

### 2.2 RGB-only 只是兼容模式

当前 worker 对普通 RGB 图的支持方式是：

- 输入 RGB 图时自动合成一个 pseudo-NIR。
- 然后按 `RGB + pseudo-NIR` 的形式喂给 CUHK 模型。

这条路径的意义是：

- 验证 `CloudRemoval` 工具是否能启动。
- 验证 `EMRDM + checkpoint + worker + 输出落盘` 是否通。

这条路径的限制是：

- 结果不应被当成正式业务结果。
- 如果后续要正式提供“普通 RGB 去云”，最好换成真正的单时相纯 RGB checkpoint。

### 2.3 当前更适合 GPU，不适合 CPU 全尺寸

本次联调已确认：

- `512x512` 的 CPU 推理在当前环境中会在 `natten/flex_attention` 上触发极大内存申请，无法作为实际运行方式。
- 因此当前 `CUHK` 版本应视为“需要 GPU 的工具”。

### 2.4 预览图颜色可能不直观

当前 preview PNG 走的是模型配置里的：

- `sgm.util.nir_to_rgb`

因此 preview 的色彩表达更接近“模型内部的 4 通道结果映射”，不一定等同于普通 RGB 图像的自然颜色观感。

## 3. 本次做过的修改

下面按“主仓库改动”“EMRDM 本地补丁”“环境改动”三部分列出。

### 3.1 主仓库改动

#### 新增 worker

- `tool_server/tool_workers/online_workers/CloudRemoval_worker.py`

主要内容：

- 新增 `CloudRemovalWorker`。
- 支持加载本地 `EMRDM` 配置和 checkpoint。
- 支持普通 RGB 图、3-band TIFF、4-band RGB+NIR TIFF。
- 为当前 `CUHK` backend 增加 `nir_image` 参数。
- 当没有 `nir_image` 时，支持测试模式 pseudo-NIR 合成。
- 推理完成后输出主结果文件和 preview PNG。
- 为 `torch 2.11 + CPU/GPU` 运行时做了两项适配：
  - `load_model_from_config(..., freeze=False)`，避免 EMA 推理时断言失败。
  - 将 `model.sampler.device` 和 `ideal_sampler.device` 同步到 worker 实际 device，避免采样器内部仍然默认跑到 `cuda`。

#### 注册和工具管理

- `tool_server/tool_workers/tool_manager/base_manager.py`

主要内容：

- 将 `CloudRemoval` 加入在线工具检查列表。
- 为 `CloudRemoval` 增加较长的调用超时时间。

#### App 侧参数绑定

- `app/app_new.py`

主要内容：

- 将 `CloudRemoval.image` 绑定到主图像输入。
- 将辅助图像中的 `time1_image` 作为 `CloudRemoval.nir_image` 的候选绑定。
- 页面注释同步更新，说明 `CloudRemoval` 可选使用 NIR。

#### Agent / tf_eval 提示词

- `tool_server/tf_eval/utils/rs_agent_prompt.py`

主要内容：

- 将 `CloudRemoval` 的描述从原来的通用设想更新为当前 `CUHK RGB+NIR` 语义。
- 明确写出 RGB-only 时会进入 pseudo-NIR 测试模式。

#### 启动配置

- `tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml`
- `tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml`

主要内容：

- 新增/更新 `CloudRemoval` 服务配置。
- 本地配置已指向当前 `CUHK` config 和 checkpoint。
- 本地配置的 conda 环境使用：
  - `tool1_backup_20260410_emrdm`

#### 单工具测试脚本

- `scripts/tools_test/cloud_removal_single_test.py`

主要内容：

- 新增单工具测试入口。
- 读取：
  - `CLOUD_REMOVAL_IMAGE`
  - `CLOUD_REMOVAL_NIR_IMAGE`（可选）
- 方便走 tool manager/controller 链路验证。

### 3.2 EMRDM 本地补丁

注意：下面这些修改发生在本地克隆目录 `models/EMRDM` 内，不一定会自动出现在主仓库版本管理里。

#### 延迟初始化 LPIPS，避免推理阶段强制联网

- `models/EMRDM/sgm/modules/learning/evaluator.py`

修改内容：

- 将 `lpips.LPIPS(...)` 从模块导入时初始化，改为按需懒加载。

原因：

- 原始实现会在模型初始化时触发 `torchvision` 下载 `alexnet` 权重。
- 当前联调环境无网络，导致 checkpoint 加载卡死。
- 这个补丁不影响推理主链，只是避免评估器在推理启动时强制联网。

#### 兼容新版 natten API

- `models/EMRDM/sgm/modules/diffusionmodules/k_diffusion/image_transformer.py`
- `models/EMRDM/sgm/modules/encoders/transformer_encoder.py`

修改内容：

- 增加 `natten_has_fused_na()` 兼容函数。
- 兼容当前安装的 `natten 0.21.5`，因为它没有旧版代码依赖的 `natten.has_fused_na()` 接口。

原因：

- 不做兼容时，模型前向会直接报：
  - `AttributeError: module 'natten' has no attribute 'has_fused_na'`

### 3.3 conda 环境改动

环境：

- `tool1_backup_20260410_emrdm`

为跑通 `EMRDM + CUHK`，本次在该环境中补装过以下依赖：

- `pytorch-lightning`
- `einops`
- `natsort`
- `dctorch`
- `natten`
- `lpips`

注意：

- 这些是环境级修改，不在仓库代码里自动固化。
- 如果后续迁移机器或重建环境，需要重新核对这些包。

## 4. 为什么现在先保留 CUHK 方案

虽然 `CUHK` 不是理想的纯 RGB 方案，但它有两个现实优势：

- 当前本地已经有可用 checkpoint。
- 经过本次补丁后，整条链路已经被验证到“可加载、可采样、可落盘”。

因此如果当前目标是“先把 CloudRemoval 工具挂起来”，继续用 `CUHK` 是可以接受的。

但必须接受下面的边界：

- 正式推荐输入应是 `RGB + NIR`。
- 普通 RGB 输入只能视为测试兼容。
- 正式运行应优先在 GPU 上做。

## 5. 日后切换模型时需要做什么

后续如果切换到新的 checkpoint，不要只换权重路径，要按下面顺序重新核对。

### 5.1 先确认新模型的输入语义

必须先确认：

- 输入是纯 RGB、RGB+NIR、RGB+SAR，还是别的多模态组合。
- 输入通道数是多少。
- 输出通道数是多少。
- `conditioner` 需要哪些 key。
- `to_rgb_config` 是什么。

至少要从新模型的配置文件里看清：

- `network_config.params.in_channels`
- `network_config.params.out_channels`
- `mean_key`
- `conditioner_config`
- `to_rgb_config`

### 5.2 更新服务配置

通常至少要改：

- `tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml`
- `tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml`

主要更新：

- `config-path`
- `model-path`
- 需要时更新 `conda_env`

### 5.3 更新 worker 的输入描述和兼容逻辑

如果新模型不是 `RGB+NIR`，要同步修改：

- `tool_server/tool_workers/online_workers/CloudRemoval_worker.py`

重点检查：

- `get_tool_instruction()` 里的参数说明。
- `_prepare_input_tensor()` 的输入处理逻辑。
- `build_cuhk_input()` / pseudo-NIR 路径是否还需要保留。
- 是否还需要 `nir_image` 参数。

如果新模型是纯 RGB：

- 建议删掉或关闭 pseudo-NIR 逻辑。
- 工具输入收敛成只需要 `image`。

### 5.4 更新 app 和 agent 提示词

至少同步修改：

- `app/app_new.py`
- `tool_server/tf_eval/utils/rs_agent_prompt.py`
- `scripts/tools_test/cloud_removal_single_test.py`

原因：

- 当前这些地方都已经写成了 `CUHK + nir_image 可选` 的语义。
- 仅仅换 worker 不同步这些说明，会让 agent 和前端继续按旧接口调用。

### 5.5 重新核对 EMRDM 兼容补丁

切换模型时需要重新确认下面几项是否仍然适用：

- `freeze=False` 是否仍然需要。
- `sampler.device` 同步是否仍然需要。
- `natten` 兼容补丁是否仍然需要。
- `LPIPS` 懒加载是否仍然需要。

### 5.6 重新做两层测试

建议至少做：

1. 本地直接推理测试
2. tool server / controller 链路测试

如果新模型是正式纯 RGB 模型，还应增加：

- 一组真实 RGB 输入的视觉效果验证
- 一组 GeoTIFF 输入输出验证

## 6. 当前建议的使用方式

当前版本更适合按下面方式理解：

- 工具名称继续叫 `CloudRemoval`
- 后端暂时固定到 `EMRDM CUHK checkpoint`
- 正式推荐输入：
  - 主图像 `image`
  - 可选真实 NIR `nir_image`
- 如果只给 RGB：
  - 可以用于 smoke test
  - 不建议用于正式结果

## 7. 手动 GPU 测试指令

下面是一条不依赖 controller、直接本地调用 worker 的 GPU 手动测试指令。

注意：

- 这条命令要求目标机器上 `CUDA` 可用。
- 如果你继续用 `tempcode/56.png`，它仍然会走 pseudo-NIR 测试模式。
- 如果你要验证正式质量，应该把 `image` 和 `nir_image` 都换成真实的 `RGB + NIR` 输入。

```bash
CUDA_VISIBLE_DEVICES=1 \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp \
FONTCONFIG_PATH=/etc/fonts \
conda run -n tool1_backup_20260410_emrdm python -c "
import json
from tool_server.tool_workers.online_workers.CloudRemoval_worker import CloudRemovalWorker

worker = CloudRemovalWorker(
    controller_addr='http://127.0.0.1:1',
    worker_addr='http://127.0.0.1:40031',
    no_register=True,
    port=40031,
    model_name='CloudRemoval',
    emrdm_root='models/EMRDM',
    config_path='models/cpk/cuhk-20260410T131636Z-3-001/cuhk/configs/2024-11-09T08-59-25-project.yaml',
    model_path='models/cpk/cuhk-20260410T131636Z-3-001/cuhk/checkpoints/last.ckpt',
    device='cuda',
    pad_multiple=16,
)

result = worker.generate({
    'image': 'tempcode/56.png',
    'output_path': 'tempcode/cloud_removal_56_gpu_cuhk.tif',
    # 如果有真实 NIR，取消下一行注释并替换路径：
    # 'nir_image': '/absolute/path/to/nir.tif',
})

print(json.dumps(result, ensure_ascii=False, indent=2))
"
```

如果要做正式测试，推荐把上面的输入改成：

- `image=/absolute/path/to/cloudy_rgb.tif`
- `nir_image=/absolute/path/to/nir.tif`

## 8. 本次联调后已清理的测试产物

以下内容已经清理，不再保留在仓库工作区里：

- `tempcode` 下本次生成的缩放测试图
- `tempcode` 下本次生成的去云输出和 preview
- `tool_server/tool_workers/logs/automatic_generated` 下本次生成的临时 worker 日志


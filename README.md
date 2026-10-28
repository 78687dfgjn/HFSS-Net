# HFSS-Net：SDGW + EFC + HFP 双流目标检测

本仓库当前固定组合为 **`combo__sdgw__efc__hfp`**，已替换旧版 HFP + PFA + SCSU。目标是复现 RGBTDronePerson 官方 4,900 对训练 / 1,225 对评估划分上的 **约 49.4% Val mAP50**。仓库只包含复现源码与说明，不附带数据、训练权重或结果文件。

## 参考结果与选权口径

当前参考运行使用 seed0：训练于第 **132** 轮按 fitness 早停，**第76轮 `best_map50.pt` 的 Val mAP50 为 49.4049694%**；fitness 最佳为第82轮的 `best.pt`。复现49.4%应读取 **`best_map50.pt`**，不要用 `best.pt` 替代，也不要只读最后一轮。该分数是参考运行的单种子观察值，不保证不同软硬件环境逐位一致。

官方1,225对已经用于每轮验证和模型选型，本文称为**选型验证集**；`test` 与 `val` 指向同一批图像，复评不构成独立盲测。历史内部开发集的57%成绩与本目标不同，不能混用。

## 固定架构

| 插槽 | 模块 | 实际层号 / 通道 |
|---|---|---|
| 融合前，两条支路分别增强 | SDGW | RGB/TIR：9/11、14/16、20/22；64/128/256 |
| 双模态汇合 | EFC | 23/24/25；64/128/256 |
| 融合后 Neck 三尺度 | HFP | Detect输入32/35/38；64/128/256 |

基础配置直接使用无参数 `PairedAdd`，不再先构建旧注意力模块再替换。为保持参考运行的初始化，构造时仅重放历史初始化与尺寸推断所消耗的随机数，不保存旧模块、旧参数或注意力计算图；最终运行的融合层为 EFC 残差融合。

各插槽均采用零初始化标量残差门控，初始输出保持纯 Add；CMBM、GAFFM及PSAB关闭。HFP采用本地源码对应的**正交DCT**高通实现，并非FFT替代版；EFC保留实验中的有符号分母保护。RGBT三类模型总参数为 **7,230,712**（存储参数口径）。模块初始化按名称和层号使用独立固定随机流，与该参考组合一致。

## 1. 环境安装

实验环境：Linux、单张 NVIDIA A30 24GB、Python 3.11.13、PyTorch 2.5.0+cu118、torchvision 0.20.0+cu118、CUDA运行时11.8、OpenCV 4.10.0、NumPy 2.1.2。依赖版本见 `pyproject.toml`。集群 `torch` 环境的全部104项包版本见 [environment-pip-list.txt](environment-pip-list.txt)，于2026-10-28执行 `python -m pip list --format=freeze --disable-pip-version-check` 原样导出；该文件是完整环境快照，项目安装依赖仍以 `pyproject.toml` 为准。

```bash
conda create -n hfss-net python=3.11.13 -y
conda activate hfss-net
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/78687dfgjn/HFSS-Net.git
cd HFSS-Net
python -m pip install -e .
```

使用仓库自带的双流运行库，不要另外安装PyPI版ultralytics。安装依赖缓慢时，最后一步可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

以下命令在仓库根目录执行。数据、预训练权重和训练输出放在相邻的 `../repro-work/`，处理脚本会检查输出路径。

## 2. 预训练权重

下载官方YOLOv8n COCO预训练权重：

```bash
mkdir -p ../repro-work/assets
curl -L --fail https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt \
  -o ../repro-work/assets/yolov8n.pt
```

参考官方 `yolov8n.pt` 的SHA256为 `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`；请先核对下载文件。

程序自动向RGB/TIR双流骨干迁移324个张量，检查形状并记录权重SHA256；Neck和检测头采用固定seed0初值。单类数据集自动重建单类检测头，各数据集独立冷启动。

## 3. 数据集处理

统一使用热红外坐标系标签，转换为YOLO格式 `class cx cy width height`。保留空标注图像，越界框裁剪至图像边界，退化框跳过。RGB/TIR共同进行增强与letterbox，输入为六通道、640×640。

输出结构：

```text
prepared/
├── images/train/      # RGB
├── images/val/
├── image/train/       # TIR，目录名为单数image
├── image/val/
├── labels/train/
├── labels/val/
└── dataset.yaml
```

### RGBTDronePerson

输入 `RGBTDronePerson.zip`，其中包含 `train_thermal.json`、`val_thermal.json` 及 `train|val/visible|thermal/`，允许存在共同外层目录。

- 官方训练/评估：4900/1225对。
- 类别顺序：person、rider、crowd。
- 读取thermal JSON，跳过ignore标注；保留原图字节与尺寸。

```bash
python prepare.py rgbt --source ../repro-work/raw/RGBTDronePerson.zip \
  --output ../repro-work/data/rgbt
```

### VTUAV-det

先解开外层下载包，准备以下文件；内层ZIP不必展开：

```text
VTUAV-source/
├── VTUAV_v1.0.zip     # train/rgb、train/ir、test/rgb、test/ir
├── train_ir.json
└── val_ir.json
```

- 官方训练/评估：11392/5378对，person单类。
- 使用IR COCO标签，过滤ignore、iscrowd和非person类别。
- 图像使用INTER_AREA缩放至640×360，保存为PNG；框按原尺寸归一化。
- JSON的val对应ZIP内test，保持官方划分。

```bash
python prepare.py vtuav --source ../repro-work/raw/VTUAV-source \
  --output ../repro-work/data/vtuav
```

### VTSaR

使用 `VTSaR_Crop_640.7z`：

```bash
7z t ../repro-work/raw/VTSaR_Crop_640.7z
7z x ../repro-work/raw/VTSaR_Crop_640.7z -o../repro-work/raw/VTSaR-Crop
python prepare.py vtsar --source ../repro-work/raw/VTSaR-Crop \
  --output ../repro-work/data/vtsar
```

- 官方训练/评估：4800/533对，person单类。
- 按路径中的train/val、rgb/ir/labels定位配对文件。
- 保留640×480 JPEG原字节；标签仅在越界裁剪时重写。
- 缺失配对会报错，多余孤立IR/标签文件不进入样本集。

### 数据复核

处理时自动检查数量、图像可读性、RGB/TIR配对、尺寸、标签范围及跨训练/评估的完全重复图像对，检查记录写入数据目录。也可单独复核：

```bash
python prepare.py rgbt --source ../repro-work/data/rgbt --audit-only
python prepare.py vtuav --source ../repro-work/data/vtuav --audit-only
python prepare.py vtsar --source ../repro-work/data/vtsar --audit-only
```

## 4. 复现约49.4%的训练命令

先执行上面的 RGBTDronePerson 数据准备命令，确认训练4,900对、验证1,225对、类别顺序person/rider/crowd。不要使用旧980对筛选子集、内部980对验证集或XML替代thermal JSON。

下列为参考运行设置；不缩小batch、不使用缓存或compile，也不改输入分辨率。

| 设置 | 固定值 |
|---|---|
| 初始化 | 官方YOLOv8n COCO，骨干324张量双流迁移；基础初始化seed0 |
| 训练随机种子 | 0 |
| 学习率总计划 | 从开始按400 epoch设置，线性衰减，lrf=0.01 |
| Batch / nbs | 32 / 64；warmup后通常累计2步 |
| 优化器 | SGD，lr0=0.005，momentum=0.937，weight_decay=0.0005 |
| Warmup | 3轮；momentum=0.8，bias_lr=0.1 |
| 精度 | AMP前向；分配器分类分数、解码框与几何计算强制FP32 |
| 数据加载 | train workers4 / val workers8，每轮重新创建非持久worker，pin_memory=True |
| 输入与验证 | 640×640，rect=False，验证batch32、FP32，每轮Val |
| 验证阈值 | conf=0.25，NMS IoU=0.30，max_det=300 |
| 早停 | fitness，patience=50；fitness=0.1×mAP50+0.9×mAP50–95 |
| Mosaic | 按400轮计划最后10轮关闭，即第391轮起；不在第50轮提前关闭 |

参考运行在第50轮做过预算暂停，随后从完整状态续训；本入口默认直接长训，使用相同400轮计划和每轮加载器生命周期。恢复会保留优化器、EMA、scaler、scheduler、随机数、加载器生成器、累计最佳记录和早停历史，不在第50轮清零。跨设备/库版本仍不能承诺逐位一致。

```bash
python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name rgbt_sdgw_efc_hfp_seed0 --batch 32 --workers 4 --seed 0 --device 0

python train.py --dataset vtuav --data ../repro-work/data/vtuav/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name vtuav_hfss_seed0 --device 0

python train.py --dataset vtsar --data ../repro-work/data/vtsar/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name vtsar_hfss_seed0 --device 0
```

**不要把 `--epochs` 改成50、76或132**：这会改变整条学习率曲线，不再是参考协议；也不要从已训练的HFP底座热启动。默认从共同初始化冷启动，最多400轮，连续50轮fitness未改善时早停。若需完整运行400轮，增加 `--patience 0`。多种子实验使用 `--seed 1` / `--seed 2`，并更换运行名。

输出文件：

- `weights/best.pt`：最高fitness，计算式为0.1×mAP50 + 0.9×mAP50–95。
- `weights/best_map50.pt`：最高mAP50。
- `weights/last.pt`：完整恢复，包含FP32模型、EMA、优化器、scaler、scheduler、RNG、数据加载生成器及早停状态；只用于本版本恢复。
- `results.csv`、`protocol.json`、`evaluation.json`：逐轮指标、训练设置和两种最佳权重的FP32评估。

断点恢复沿用原总轮数与学习率计划：

```bash
python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --resume ../repro-work/runs/rgbt_sdgw_efc_hfp_seed0/weights/last.pt \
  --project ../repro-work/runs --device 0
```

## 5. 评估

```bash
python evaluate.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --weights ../repro-work/runs/rgbt_sdgw_efc_hfp_seed0/weights/best_map50.pt \
  --output ../repro-work/eval/rgbt_hfss --split test --device 0
```

比较fitness选权时另行评估 `best.pt`，不得将其成绩标为mAP50选权成绩。其他数据集同步替换dataset、data和weights。指标沿用PTANet的mAP50和mAP50–95，结果写入输出目录的 `metrics.json`。

| 数据集 | conf阈值 | NMS IoU | 输入 | 精度 | max_det |
|---|---:|---:|---|---|---:|
| RGBTDronePerson | 0.25 | 0.30 | 640×640方形letterbox | FP32 | 300 |
| VTUAV-det | 0.25 | 0.50 | 同上 | FP32 | 300 |
| VTSaR | 0.03 | 0.45 | 同上 | FP32 | 300 |

默认评估batch32、rect=False，与参考Val一致。CPU可使用 `--device cpu --workers 0`；若内存不足再显式加 `--batch 1`，但应记录batch变化，不能要求与batch32逐位相同。

## 6. 分数偏低时先检查

1. 是否在本仓库根目录安装并运行，导入的是本仓库运行库，而非PyPI版ultralytics。
2. 是否使用thermal JSON、完整4,900/1,225对及正确的RGB/TIR配对，没有混用旧内部划分。
3. 是否保持batch32/nbs64、workers4、seed0、FP32分配，以及从开始就400轮的学习率计划。
4. 是否比较第76轮附近的累计最佳 `best_map50.pt`，而非fitness最佳、最后一轮或仅50轮结果。最佳轮次可能因环境改变。
5. 查看输出 `protocol.json`、`results.csv`、`weights/best_map50.json`。本仓库不承诺多种子平均49.4%，也未用独立盲测证明该分数。

VTUAV-det和VTSaR保留数据准备与运行接口，49.4%仅对应RGBTDronePerson，不表示另外两个数据集已验证同样收益。旧版HFP+PFA+SCSU检查点与当前结构不兼容；如需旧版代码，请使用Git历史中的旧提交。

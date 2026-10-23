# HFSS-Net: High-Frequency Perception and Spatial Selection for RGB-T Object Detection

支持 RGBTDronePerson、VTUAV-det 和 VTSaR 的训练与评估。

## 1. 环境安装

实验环境：Linux、单张 NVIDIA A30 24GB、Python 3.11.13、PyTorch 2.5.0+cu118、torchvision 0.20.0+cu118、CUDA运行时11.8、OpenCV 4.10.0、NumPy 2.1.2。依赖版本见 `pyproject.toml`。

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

## 4. 正式训练

使用官方训练/评估划分。默认seed0、最多400轮、patience=50、最后10轮关闭Mosaic，每轮验证。训练采用SGD、lr0=0.005、warmup=3、batch8、nbs64、AMP前向与FP32正样本分配。

```bash
python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name rgbt_hfss_seed0 --device 0

python train.py --dataset vtuav --data ../repro-work/data/vtuav/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name vtuav_hfss_seed0 --device 0

python train.py --dataset vtsar --data ../repro-work/data/vtsar/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt \
  --project ../repro-work/runs --name vtsar_hfss_seed0 --device 0
```

连续50轮fitness未改善时早停。若需完整运行400轮，增加 `--patience 0`。多种子实验使用 `--seed 1` / `--seed 2`，并更换运行名。

输出文件：

- `weights/best.pt`：最高fitness，计算式为0.1×mAP50 + 0.9×mAP50–95。
- `weights/best_map50.pt`：最高mAP50。
- `weights/last.pt`：断点恢复，包含优化器与EMA。
- `results.csv`、`protocol.json`、`evaluation.json`：逐轮指标、训练设置和两种最佳权重的FP32评估。

断点恢复沿用原总轮数与学习率计划：

```bash
python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --resume ../repro-work/runs/rgbt_hfss_seed0/weights/last.pt \
  --project ../repro-work/runs --device 0
```

## 5. 评估

```bash
python evaluate.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --weights ../repro-work/runs/rgbt_hfss_seed0/weights/best.pt \
  --output ../repro-work/eval/rgbt_hfss --split test --device 0
```

评估最高mAP50权重时替换为 `best_map50.pt`。其他数据集同步替换dataset、data和weights。指标沿用PTANet的mAP50和mAP50–95，结果写入输出目录的 `metrics.json`。

| 数据集 | conf阈值 | NMS IoU | 输入 | 精度 | max_det |
|---|---:|---:|---|---|---:|
| RGBTDronePerson | 0.25 | 0.30 | 640×640方形letterbox | FP32 | 300 |
| VTUAV-det | 0.25 | 0.50 | 同上 | FP32 | 300 |
| VTSaR | 0.03 | 0.45 | 同上 | FP32 | 300 |

默认评估batch8、rect=False；CPU评估使用 `--device cpu --workers 0`。

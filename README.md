# YoloV8 Mouse Pose Estimation Project

基于 YOLOv8 的小鼠姿态估计与目标检测项目，使用 Ultralytics 框架。

## 项目结构

```
YoloV8/
├── MovingExtract.py          # 运动目标提取与姿态标注
├── YoloTrain.py              # YOLO 模型训练脚本
├── YoloTrainDataSplit.py     # 训练数据划分脚本
├── yoloTrain.yaml            # 目标检测模式配置文件
├── yoloPoseTrain.yaml       # 姿态估计模式配置文件
├── models/                   # 预训练模型与训练产出
├── YoloTrainData/            # 训练数据集
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── runs/detect/              # 训练输出目录
```

---

## YOLO 基础概念

### YOLO 任务模式

YOLOv8 支持多种任务模式，通过 `model.train()` / `model.val()` / `model.predict()` 调用：

| 模式 | 说明 | 输出 |
|------|------|------|
| **detect** | 目标检测（最常用） | 边界框 + 类别 |
| **pose** | 姿态估计 | 边界框 + 关键点 |
| **segment** | 实例分割 | 边界框 + 掩码 |
| **classify** | 图像分类 | 类别概率 |

本项目定义的 4 个关键点：

| ID | 名称 | tip (鼻尖) | head (头顶) | body (身体) | tailbase (尾根) |
|----|------|-----------|-------------|-------------|-----------------|

---

## 核心脚本说明

### 1. MovingExtract.py

**功能**: 从视频或图像序列中提取运动目标，支持目标检测和姿态标注。

**输入**: MP4 视频文件 或 包含 JPG 图像的文件夹

**输出**:
- `PoseExtract{时间戳}/` — 提取的 ROI 图像和完整帧
- `{文件名}.png` — 放大的关键点标注图像
- `{文件名}.jpg` — 完整帧图像
- `{文件名}.txt` — YOLO 格式标签

**主要参数** (文件开头配置):

```python
mediaName = r"xxx.mp4"  # 输入媒体
useModel = False                     # False=背景减法检测, True=YOLO模型检测
PoseExtract = True if not useModel else False  # 是否进行姿态标注
conf = 0.3                           # 检测置信度阈值 (useModel=True时)
sample = 0.8                          # 图像采样率
minArea = 650                         # 最小检测区域
maxArea = 2400                        # 最大检测区域
PoseExtractDivider = 20              # 每隔N帧进行一次姿态提取
```

**检测模式**:

| `useModel` | 检测方式 | 说明 |
|------------|---------|------|
| `False` | MOG2/KNN 背景减法 | 无需预训练模型，基于帧差法检测运动物体 |
| `True` | YOLO 模型检测 | 使用 `models/xxx.pt` 进行目标检测 |

**ROI 选择流程**:
1. 运行后首先通过 `cv2.selectROI` 选择感兴趣区域 (可多次选择，按 `s` 保存遮罩)
2. 按 `ESC` 或空格键结束 ROI 选择
3. 对于静态物体提取 (`extractStaticObject=True`)，还需选择静态物体区域

**姿态标注操作** (当 `PoseExtract=True` 时):
- 弹窗显示检测到的 ROI，放大 4 倍便于标注
- **左键**: 标注可见关键点
- **右键**: 标注不可见关键点
- **Backspace**: 删除上一个点
- **Enter**: 确认当前标注，保存并退出
- **N**: 跳过当前帧
- **Shift+ESC**: 强制退出

**标签输出格式**:

```
0 <x_center> <y_center> <width> <height> <tip_x> <tip_y> <tip_v> <head_x> <head_y> <head_v> <body_x> <body_y> <body_v> <tailbase_x> <tailbase_y> <tailbase_v>
```

---

### 2. YoloTrainDataSplit.py

**功能**: 将原始标注数据划分为训练集、验证集、测试集，并整理为 YOLO 目录结构。

**输入**: `MainDatasetPaths` 列表指定的原始数据目录，根据png文件名进行筛选，即手动筛选数据集时只需要删除不需要的png文件即可。

**输出**: 按照 YOLO 格式组织的 `YoloTrainData/` 目录

**主要参数**:

```python
MainDatasetPaths = [r'..\YoloV8\OutputMouseBodyPicxxx']  # 原始数据路径
samplePerDataset = [1]                                               # 每数据集采样率
txtsavepath = '../YoloTrainData/labels'          # 标签输出路径
imgsavepath = '../YoloTrainData/images'          # 图像输出路径
trainval_percent = 0.9   # 训练+验证集占比(剩余部分为测试集)
train_percent = 0.9      # 训练集占训练+验证集的比例
```

**输出目录结构**:

```
YoloTrainData/
├── images/
│   ├── train/    # 训练图像
│   ├── val/      # 验证图像
│   └── test/     # 测试图像
└── labels/
    ├── train/    # 训练标签 (.txt)
    ├── val/      # 验证标签
    └── test/     # 测试标签
```

**数据筛选**: 仅处理文件名以 `pose.png` 结尾的文件（由 MovingExtract.py 产生）

---

### 3. YoloTrain.py

**功能**: 使用预训练的 YOLO Pose 模型进行训练，评估后导出为 TensorRT 引擎格式。

**主要流程**:

```python
model = YOLO(r'models\yolo11n.pt')  # 加载预训练 模型
model.train(data="yoloTrain.yaml", epochs=_epochs, multi_scale=True, workers=0)
metrics = model.val()                     # 评估模型
# 找到最新训练结果中的 best.pt
model.export(format='engine', dynamic=False, half=False)  # 导出 TensorRT
```

**训练配置**:
- 预训练模型: `yolo11n.pt` (YOLOv11 最小模型，可选其他略大的模型)
- 训练轮数: 30 epochs，可自定义

**训练产物**: `runs/detect/train{N}/weights/best.pt`

**额外导出格式**: TensorRT `.engine` 格式，用于cpu推理加速

---

## YAML 配置文件

### yoloTrain.yaml — 目标检测模式

与 pose 模式不同，detect 模式仅输出类别和边界框，不含关键点：

```yaml
train: ../YoloV8/YoloTrainData/images/train
val: ../YoloV8/YoloTrainData/images/val
nc: 2                   # 类别数: mouse(小鼠) + lickSpout(舔口)
names: ['mouse', 'lickSpout']
```

标签格式为标准 YOLO 检测格式：`<class_id> <x_center> <y_center> <width> <height>`（无关键点）。

### yoloPoseTrain.yaml — 姿态估计模式

在 detect 基础上增加了关键点检测能力，本项目用于小鼠行为分析场景：4 个关键点追踪鼻尖、头顶、身体和尾根位置。

---

## 数据流转

```
原始视频/图像
    ↓
MovingExtract.py (目标检测 + 姿态标注)
    ↓
PoseExtract{时间戳}/ (*.jpg, *.txt, *.png)
    ↓
YoloTrainDataSplit.py (数据划分)
    ↓
YoloTrainData/{images,labels}/{train,val,test}/
    ↓
yoloPoseTrain.yaml (数据配置)
    ↓
YoloTrain.py (模型训练)
    ↓
runs/detect/train{N}/weights/best.pt (部署模型)
    ↓（非必要）
model.export(format='engine') → *.engine (cpu部署模型)
```

---

## YAML 配置文件

### yoloTrain.yaml — 目标检测模式

```yaml
train: ../YoloV8/YoloTrainData/images/train
val: ../YoloV8/YoloTrainData/images/val
nc: 2                    # 2个类别
names: ['mouse', 'lickSpout']  # 类别: 小鼠, 舔口，可只训练小鼠一类，根据训练种类调整nc数值
```

### yoloPoseTrain.yaml — 姿态估计模式

```yaml
train: ../YoloV8/YoloTrainData/images/train
val: ../YoloV8/YoloTrainData/images/val
kpt_shape: [4, 3]
nc: 1
names: ['mouseWithMiniscope']
```

---

## 模型说明

| 模型文件 | 类型 | 用途 |
|---------|------|------|
| `yolo11s-pose.pt` | Pose 预训练 | 训练起点 |
| `TopViewMiniscopeBodyBestWithAddition.pt` | 检测模型 | 常用小鼠位置检测模型 |
| `TopViewMiniscopeBodyBestPose.pt` | Pose 训练产出 | 已训练的小鼠姿态模型 |

---

安装依赖:
```bash
pip install ultralytics opencv-python torch numpy scikit-image keyboard
根据系统环境以及显卡版本安装对应版本的 torch 和 torchvision
```

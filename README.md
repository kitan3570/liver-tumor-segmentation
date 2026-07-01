# 肝脏与肝肿瘤 3D 分割科研项目 (MONAI / PyTorch)

> ⚠️ **免责声明（重要）**
>
> 本项目**仅用于科研学习与展示**，演示医学影像 AI 分割任务的完整流程。
> 本项目**不用于临床诊断、治疗或任何医疗决策**，其输出结果**不可**作为任何医学判断的依据。
> 本项目**不使用任何真实患者隐私数据**；所用数据为公开的 Medical Segmentation Decathlon (MSD) Task03_Liver 数据集，已脱敏。
> 使用者应遵守数据集原始许可协议，并对自己的使用行为负责。

---

## 1. 项目简介

本项目基于 **Medical Segmentation Decathlon (MSD) Task03_Liver** 公开数据集，使用 **MONAI + PyTorch** 框架构建 **3D U-Net** 模型，完成腹部 CT 中**肝脏与肝肿瘤的 3D 体积分割 (volumetric segmentation)**。

项目采用**两阶段 (two-stage) 策略**：

- **阶段一 (phase1, 二分类)**：将背景与肝脏（含肿瘤区域）区分开来；
- **阶段二 (phase2, 三分类)**：在肝脏区域内进一步区分背景 / 肝脏实质 / 肝肿瘤。

项目提供从环境自检、数据检查、数据划分、训练、推理、评价、可视化到 STL 三维模型导出的**全流程脚本**，并附带中文注释，便于阅读理解。

### 1.1 两种使用模式

本项目同时提供**命令行模式**与 **GUI 桌面模式**，二者共用同一套 `src/*.py` 核心脚本，可按需选择：

| 模式 | 适合人群 | 入口 | 优点 |
| --- | --- | --- | --- |
| **命令行模式** | 有 Python 基础、习惯终端的同学 | `src/*.py` 脚本（见 §7B） | 灵活、可批量、便于在服务器/远程跑长任务 |
| **GUI 桌面模式** | 希望可视化操作的使用者 | `python gui/app.py`（见 §6B） | 可视化参数面板、实时日志、一键打开结果 |

> 两种模式下训练得到的模型、推理结果、评估指标完全一致；GUI 不重写任何训练逻辑，
> 仅通过子进程调用 `src/*.py`。下文 §6B 介绍 GUI 模式，§7B 介绍命令行模式。

---

## 2. 项目特点

- 🎯 **两阶段分割策略**：先粗后细，降低小目标（肿瘤)分割难度；
- 🧠 **3D U-Net**：医学影像体积分割的经典网络；
- 🛠️ **全流程脚本**：环境检查 / 数据检查 / 划分 / 训练 / 推理 / 评价 / 可视化 / STL 导出；
- 🇨🇳 **中文注释**：源代码与配置文件均带有详细中文说明；
- ✅ **自检脚本 (smoke test)**：可在 CPU 上小规模跑通流程，验证环境；
- 📊 **标准指标**：Dice / IoU / Precision / Recall，分别统计 liver 与 tumor；
- 🧩 **notebooks 教程**：提供数据探索与中文分步教程 Notebook。

---

## 3. 环境要求

| 组件 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11（亦兼容 Linux） |
| Python | 3.10 |
| PyTorch | 需与本地 CUDA 版本匹配（建议 CUDA 11.8 / 12.1） |
| GPU | 推荐 NVIDIA GPU（显存 ≥ 8GB）；亦可在 CPU 上跑 smoke test |
| MONAI | 最新稳定版 |
| 其它 | numpy, nibabel, SimpleITK, pyyaml, tqdm, matplotlib, trimesh 等 |

> 训练完整模型强烈建议使用 GPU；CPU 仅用于环境验证与小规模 smoke test。

---

## 4. 数据集获取

本项目使用 **Medical Segmentation Decathlon Task03_Liver** 数据集。

- 官方网站：http://medicaldecathlon.com/
- 下载并解压后，将数据放到项目的 `data/Task03_Liver` 目录下。

解压后目录结构应如下：

```
data/Task03_Liver/
├── imagesTr/      # 训练用 CT 图像 (*.nii.gz)
├── labelsTr/      # 训练用标注 (*.nii.gz，标签: 0=背景, 1=肝脏, 2=肿瘤)
├── imagesTs/      # 测试用 CT 图像 (*.nii.gz)
└── labelsTs/      # 测试用标注 (如有)
```

> ⚠️ 请遵守 MSD 数据集许可，仅用于科研用途。

---

## 5. 安装

> 📌 **重要：PyTorch 的 CUDA 安装命令不要照搬本项目给的内容**
> PyTorch 必须与你本机的 **CUDA / 驱动版本** 匹配，否则 `torch.cuda.is_available()` 会返回 `False`。
> 请务必前往 [PyTorch 官网 Get Started](https://pytorch.org/get-started/locally/)，按照页面上的选择器勾选你的操作系统、包管理器、语言、CUDA 版本，复制官方给出的命令安装。
> `requirements.txt` / `environment.yml` 中**故意不写死** PyTorch 的 CUDA 索引源，正是为了让你自己选择合适的 CUDA 版本。

### 方式 A：pip（轻量）

```bash
# 5.1 克隆/进入项目目录
cd d:\项目1

# 5.2 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux 用 source .venv/bin/activate

# 5.3 安装 PyTorch：务必按官网对应 CUDA 选择，例如 CUDA 11.8：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5.4 安装其余依赖
pip install -r requirements.txt
```

### 方式 B：conda（推荐，参见 `environment.yml`，已固定 python=3.10）

```bash
cd d:\项目1
conda env create -f environment.yml      # 自动创建名为 liver-seg、python=3.10 的环境
conda activate liver-seg
# 仍需按官网补装对应 CUDA 版本的 PyTorch：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 环境自检：`test_env.py`

安装完后，用根目录的 `test_env.py` 检查 torch / monai / nibabel 是否能正常导入、CUDA 是否可用：

```bash
python test_env.py            # 任一依赖导入失败也优雅报告
python test_env.py --strict   # 严格模式：任一失败即以非 0 退出码结束（可用于 CI）
```

---

## 6. 目录结构

```
项目1/
├── configs/                       # 配置文件
│   ├── liver_unet_3d.yaml         # 【推荐】统一三分类 3D U-Net 训练/推理配置
│   ├── phase1_binary.yaml          # （旧版阶段一）二分类 (背景 vs 肝脏) 配置
│   └── phase2_ternary.yaml         # （旧版阶段二）三分类 (背景/肝脏/肿瘤) 配置
├── src/                           # 核心源码与新版自包含脚本
│   ├── __init__.py
│   ├── data.py                    # （模块化版）数据加载与预处理
│   ├── models.py                  # （模块化版）3D U-Net 等模型定义
│   ├── trainer.py                 # （模块化版）训练循环 / 验证 / checkpoint
│   ├── utils.py                   # 通用工具函数 (指标、可视化辅助等)
│   ├── inspect_dataset.py         # 【推荐】数据集分布检查、CSV 汇总、三方向切片图
│   ├── create_splits.py           # 【推荐】train/val/test 划分 (70/15/15, seed=42)
│   ├── train.py                   # 【推荐】自包含训练脚本 (统一三分类, DiceCELoss, AdamW, TensorBoard, resume, OOM 提示)
│   ├── infer.py                   # 【推荐】自包含推理脚本 (Invertd 恢复原空间, 保留 affine)
│   ├── evaluate.py                # 【推荐】Dice/IoU/Precision/Recall (liver/tumor 分开, CSV)
│   ├── visualize.py               # 【推荐】CT / GT / 预测 切片叠加可视化 (tumor 优先)
│   └── export_stl.py              # 【推荐】mask → STL 三维模型（仅供科研/展示）
├── scripts/                       # （旧版）模块化脚本，调用 src/{data,models,trainer,utils}
│   ├── check_env.py / check_data.py / split_data.py / train.py
│   ├── infer.py / evaluate.py / visualize.py / export_stl.py / smoke_test.py
│   └── run_nnunet_baseline.sh     # 【baseline】nnU-Net v2 一键流程（独立环境，不覆盖 MONAI 代码）
├── gui/                           # 【桌面前端】PySide6 GUI：调用 src/*.py，不重写训练逻辑
│   ├── app.py                     # GUI 入口：python gui/app.py
│   ├── main_window.py             # 主窗口（左侧导航 + 右侧 8 页面 + 日志）
│   ├── workers.py                 # QThread 后台任务（subprocess 调用 src/*.py）
│   └── widgets.py                 # 8 个页面控件（环境/数据/训练/推理/评估/可视化/STL/日志）
├── notebooks/                     # Jupyter 教程
│   ├── exploration.ipynb          # 数据探索
│   └── tutorial_zh.ipynb          # 中文分步教程
├── outputs/                       # 输出目录 (自动生成)
│   ├── models/                    # best_model.pth / last_model.pth / checkpoint_meta.json / events.out.tfevents.*
│   ├── dataset_summary.csv        # inspect_dataset.py 输出的病例分布汇总
│   ├── evaluation.csv             # evaluate.py 输出的逐例指标 + MEAN/STD
│   ├── predictions/               # 推理结果 .nii.gz
│   ├── figures/                   # 可视化图像 PNG
│   └── stl/                       # STL 三维模型
├── data/                          # 数据存放目录 (用户自行下载)
│   ├── raw/Task03_Liver/{imagesTr,labelsTr,dataset.json}
│   └── splits/{train,val,test}.json
├── requirements.txt               # 依赖列表 (torch 不写死 CUDA)
├── environment.yml                # conda 环境 (liver-seg, python=3.10)
├── test_env.py                     # 环境自检 (torch/monai/nibabel + CUDA)
├── .gitignore
├── README.md                      # 本说明文档
└── REPORT_TEMPLATE.md             # 项目报告模板
```

> 📝 **两套脚本说明**：项目同时存在两套实现。
> - **`src/*.py`（推荐，统一三分类）**：自包含、参数与 `configs/liver_unet_3d.yaml` 对应，符合 MSD Task03_Liver 标签 `0/1/2` 的统一 3 类分割。
> - **`scripts/*.py`（旧版，两阶段）**：依赖 `src/{data,models,trainer,utils}` 模块，配合 `configs/phase1_*.yaml`、`configs/phase2_*.yaml` 用于学习实践。
> 本轮新增/重写以 `src/*.py` 为准；旧版 `scripts/*.py` 与 `phase1/phase2` 配置保留不动。

---

## 6B. 桌面前端 GUI（Liver Tumor Segmentation Toolkit）

> 📌 PySide6 桌面界面，适合可视化操作。**不重写任何训练逻辑**，所有耗时任务通过
> QThread 后台调用 `src/*.py` 完成；命令日志实时显示在 `Logs` 页。

### 6B.1 安装 PySide6

```bash
pip install PySide6            # 已包含在 requirements.txt / environment.yml
```

### 6B.2 启动

```bash
python gui/app.py
```

### 6B.3 界面布局

左侧导航 + 右侧功能页：

1. **Environment 环境部署**：①一键检查环境（后台 `EnvProbeThread` 实时探测），检测区显示 Python version / Python 路径 / pip 是否可用 / torch version / CUDA available / GPU name / MONAI version / nibabel version 共 8 项；②一键安装依赖（点「一键安装依赖」会先弹窗提醒"建议先创建 conda 环境 medseg，是否继续？"，确认后执行 `python -m pip install -r requirements.txt`，日志页实时显示进度）；③**不会自动安装 CUDA 版 PyTorch** —— 若检测到 torch 未安装，会弹窗提示"PyTorch 建议根据你的 CUDA 版本从官网选择安装命令"，并在页面下方提供输入框让你粘贴官方安装命令（如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`），点「安装 PyTorch」即按你选择的 CUDA 版本执行；另有「打开 PyTorch 官网」按钮直达 https://pytorch.org/get-started/locally/。
2. **Dataset 数据管理**：① 选择原始 `Task03_Liver` 数据目录，自动检查 `imagesTr/labelsTr/dataset.json` 是否存在并分别显示两者 `.nii.gz` 文件数（数量不一致标红）；②「复制 / 导入到项目 data/raw/Task03_Liver」按钮：若已是项目目录则直接记录路径，否则弹窗让用户选「后台复制（推荐，进度条实时显示已复制/总数）」或「仅记录当前路径」；③「检查数据集」按钮调 `python src/inspect_dataset.py --data_dir <data_dir> --output_dir outputs`；④「划分数据集」按钮调 `python src/create_splits.py --data_dir <data_dir> --output_dir data/splits --train_ratio <v> --val_ratio <v> --seed <v>`，运行完成后自动检查并显示 `train.json/val.json/test.json` 是否生成；⑤ 所有命令输出实时进 Logs 页。
3. **Training 模型训练**：选择 `config / train.json / val.json`，设置 `max_epochs / batch_size / lr / roi_size / num_workers / device(auto/cuda/cpu)`，切换 liver-only 提示，开始/停止训练（调 `src/train.py`），显示 checkpoint 路径。
4. **Inference 模型推理**：选单个 `nii.gz` 或图像文件夹 + checkpoint + 输出目录 + config，开始推理（调 `src/infer.py`）。
5. **Evaluation 结果评估**：选 `pred_dir / label_dir / output_csv`，开始评估（调 `src/evaluate.py`）。
6. **Visualization 可视化**：选 `image_path / label_path(可选) / pred_path / 输出目录 / num_slices`，生成可视化（调 `src/visualize.py`），可一键打开 `outputs/figures`。
7. **STL Export 三维导出**：选 `mask_path / 输出目录 / liver(1) / tumor(2) / smooth / min_component_size`，导出 STL（调 `src/export_stl.py`）；**顶部红字提醒科研/非临床**。
8. **Logs 日志**：实时日志窗口（带时间戳）；清空 / 保存日志按钮。

### 6B.4 设计要点

- **后台执行**：`gui/workers.py` 的 `CommandWorker(QThread)` 用 `subprocess.Popen` 调 `python <src脚本> <参数>`，stdout/stderr 按行回传 Qt 信号，主线程不卡。
- **不重写逻辑**：GUI 仅负责参数采集与命令拼装，业务逻辑全部复用 `src/*.py`。
- **路径选择**：全部用 `QFileDialog`（封装为 `PathSelect` 控件）。
- **停止**：训练页「停止训练」对当前 worker 调 `request_stop()`，安全终止子进程。

### 6B.5 开发模式运行

最推荐的方式：在 conda 环境中直接运行源码（无需打包）。

```bash
# 1) 创建并激活 conda 环境（见 §5 方式 B）
conda create -n medseg python=3.10 -y
conda activate medseg

# 2) 安装依赖（PyTorch 按官网选 CUDA 版本，见 §5）
pip install -r requirements.txt

# 3) 开发模式启动 GUI（修改代码后 Ctrl+C 重启即可生效）
python gui/app.py
```

开发模式优点：① 启动快；② 改代码立即生效，便于调试；③ 直接复用 conda 环境里的
PyTorch/MONAI/CUDA，无需打包大体积依赖。

### 6B.6 打包为可执行文件（PyInstaller）

> ⚠ **重要**：不建议把 PyTorch / MONAI / CUDA 全部打包进 exe，原因：
> 1. 体积会达到数 GB，下载与分发不便；
> 2. CUDA 运行时（cudnn / cublas 等）依赖目标机器的显卡驱动，打包进去后常因
>    版本不匹配而启动失败；
> 3. **更推荐在 conda 环境中运行 GUI**（见 §6B.5）。
>
> 打包版本**主要用于界面启动**，不负责内置大型模型和数据；模型权重、数据集、
> 配置文件仍需在运行时通过 GUI 选择。

打包脚本：[`scripts/build_gui.sh`](scripts/build_gui.sh)（Linux/macOS）、
[`scripts/build_gui.bat`](scripts/build_gui.bat)（Windows）。

#### 6B.6.1 Windows 打包命令

```bat
:: 在项目根目录执行（需先 pip install pyinstaller）
scripts\build_gui.bat
```

或手动执行核心命令：

```bat
pyinstaller --name LiverTumorSegToolkit --windowed --noconfirm ^
    --collect-data PySide6 ^
    --exclude-module torch --exclude-module monai --exclude-module nibabel ^
    --exclude-module cv2 --exclude-module tensorboard --exclude-module matplotlib ^
    --add-data "configs;configs" --add-data "src;src" ^
    gui\app.py
```

产物：`dist\LiverTumorSegToolkit\LiverTumorSegToolkit.exe`（onedir 模式）。

#### 6B.6.2 Linux 打包命令

```bash
# 在项目根目录执行（需先 pip install pyinstaller）
bash scripts/build_gui.sh
```

或手动执行核心命令：

```bash
pyinstaller --name LiverTumorSegToolkit --windowed --noconfirm \
    --collect-data PySide6 \
    --exclude-module torch --exclude-module monai --exclude-module nibabel \
    --exclude-module cv2 --exclude-module tensorboard --exclude-module matplotlib \
    --add-data "configs:configs" --add-data "src:src" \
    gui/app.py
```

产物：`dist/LiverTumorSegToolkit/LiverTumorSegToolkit`（onedir 模式）。

#### 6B.6.3 打包说明

- **应用名称**：`LiverTumorSegToolkit`（由 `--name` 指定）。
- **入口**：`gui/app.py`。
- **模式**：默认 `onedir`（启动快、调试友好）；如需单文件可改 `--onefile`
  （启动稍慢，但只生成一个 exe）。
- **排除大包**：`--exclude-module torch/monai/nibabel/cv2/tensorboard/matplotlib`，
  避免 exe 体积膨胀；运行时 GUI 通过子进程调 `python src/*.py`，会自动使用
  conda 环境里的 torch/monai。
- **附加数据**：`--add-data "configs;configs"` 与 `--add-data "src;src"` 把
  配置文件与 src 脚本一并打包，便于 GUI 在子进程里调用。
  （Windows 分隔符用 `;`，Linux/macOS 用 `:`）
- **图标**：如有 `.ico` 文件，在脚本里设 `ICON=--icon=assets/app.ico`。
- **运行 exe**：双击 `LiverTumorSegToolkit.exe` 即可启动 GUI；但实际训练/推理
  仍需机器上装有 Python + torch + monai（建议在 conda 环境运行 exe，或在 exe
  同级目录放置 src/ 与 configs/）。

### 6B.7 环境安装（GUI 模式推荐流程）

> 这是 GUI 模式下最推荐的环境安装流程（命令行模式同样适用，见 §5）。请按顺序执行：

```bash
# 1) 创建 conda 环境 medseg（Python 3.10）
conda create -n medseg python=3.10 -y

# 2) 激活环境
conda activate medseg

# 3) 根据 PyTorch 官网安装匹配本机 CUDA 的 PyTorch
#    打开 https://pytorch.org/get-started/locally/ ，选择你的操作系统 / 包管理器 / CUDA 版本，
#    复制官方给出的命令执行。例如 CUDA 11.8：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4) 安装本项目其余依赖（含 PySide6 GUI 框架）
pip install -r requirements.txt
```

> ⚠️ 第 3 步**务必**按 PyTorch 官网选择与你本机 CUDA / 显卡驱动匹配的版本，否则
> `torch.cuda.is_available()` 会返回 `False`（详见 §11 常见问题）。`requirements.txt`
> 中**故意不写死** PyTorch 的 CUDA 索引源，正是为了让你自己选择。

### 6B.8 启动 GUI

```bash
# 在 medseg 环境中，于项目根目录执行：
conda activate medseg
python gui/app.py
```

启动后将出现标题为 **Liver Tumor Segmentation Toolkit** 的窗口，左侧为 8 页导航栏，
右侧为当前页的参数面板。所有耗时任务（训练 / 推理 / 评估等）在后台 QThread 中执行，
不会卡住界面；命令输出实时进入第 8 页「Logs 日志」。

### 6B.9 GUI 功能流程（10 步完整使用顺序）

下面是 GUI 模式下从 0 到完成的推荐操作顺序，每一步对应左侧导航的一个页面：

| 步骤 | 页面 | 操作 | 说明 |
| --- | --- | --- | --- |
| **1 环境检查** | Environment 环境部署 | 点「一键检查环境」 | 后台探测 Python / pip / torch / CUDA / GPU / MONAI / nibabel 共 8 项；若 torch 未装会提示去官网安装，可粘贴官方命令到输入框点「安装 PyTorch」 |
| **2 导入数据** | Dataset 数据管理 | 选 `Task03_Liver` 目录 → 点「复制/导入」 | 自动检查 `imagesTr/labelsTr/dataset.json` 结构与 `.nii.gz` 数量；可后台复制到 `data/raw/Task03_Liver`（带进度条）或仅记录路径 |
| **3 检查数据集** | Dataset 数据管理 | 点「检查数据集」 | 调 `src/inspect_dataset.py`，输出每个病例的 shape / spacing / 标签分布到 `outputs/`，便于发现异常病例 |
| **4 划分数据集** | Dataset 数据管理 | 设 `train_ratio/val_ratio/seed` → 点「划分数据集」 | 调 `src/create_splits.py`，生成 `data/splits/train.json / val.json / test.json`；完成后页面自动检查并显示三个 json 是否生成 |
| **5 设置训练参数** | Training 模型训练 | 选 `config / train.json / val.json / output_dir`；设 `max_epochs / batch_size / lr / num_workers / roi_size(x,y,z) / val_interval / device(auto/cuda/cpu) / mode(binary_liver 或 multi_class_liver_tumor)`；点「保存配置」 | 「保存配置」会把参数写回 `configs/liver_unet_3d.yaml`；`mode` 决定 `--task_mode binary_liver`（二分类，out_channels=2）或 `--task_mode multiclass`（三分类，out_channels=3） |
| **6 开始训练** | Training 模型训练 | 点「开始训练」 | 后台调 `src/train.py`，实时日志进 Logs 页；可点「停止训练」安全终止；完成后显示 `best_model.pth` 与 `last_model.pth` 路径 |
| **7 推理** | Inference 模型推理 | 选「单图 image_path」或「批量 image_dir」+ `checkpoint` + `config` + `output_dir` + `device` → 点「开始推理」 | 调 `src/infer.py`，预测 `.nii.gz` 写到 `output_dir`；完成后显示预测文件列表，可一键打开输出目录 |
| **8 评估** | Evaluation 结果评估 | 选 `pred_dir` + `label_dir` + `output_csv` → 点「开始评估」 | 调 `src/evaluate.py`，输出每个病例的 liver/tumor Dice/IoU/Precision/Recall；GUI 显示 mean±std 摘要与 QTableWidget 病例表；肿瘤 Dice 低时仅给提示不报错 |
| **9 可视化** | Visualization 可视化 | 选 `image_path` + `label_path(可选)` + `pred_path` + `output_dir` + `num_slices` → 点「生成可视化」 | 调 `src/visualize.py` 生成 PNG（GUI 不直接处理 nii.gz）；完成后扫描 `output_dir/figures` 显示缩略图列表，点击放大；可一键打开图片文件夹 |
| **10 导出 STL** | STL Export 三维导出 | 选 `mask_path` + `output_dir`；勾选 `liver(1)` / `tumor(2)`；设 `smooth` / `min_component_size` → 点「导出 STL」 | 调 `src/export_stl.py`，生成 `<stem>_liver.stl` / `<stem>_tumor.stl`；页面顶部红字提醒「仅用于科研，不可用于临床」；完成后显示 STL 文件路径 |

> 每一步的命令行都会以 `[COMMAND]` 级别写入 Logs 页（见 §6B.4），便于核对参数与复现。
> 训练 / 推理 / 评估等耗时任务可随时切到其它页面查看日志，不会阻塞界面。

---

## 7. 快速开始

在跑完整流程前，强烈建议先验证环境：

### 7.1 环境自检（根目录 `test_env.py`）

```bash
python test_env.py            # 检查 torch/monai/nibabel 导入 + CUDA 可用性
python test_env.py --strict   # 任一失败即非 0 退出（适合脚本/CI）
```

### 7.2 Smoke Test（端到端小规模自检，CPU 即可，旧版）

```bash
python scripts/smoke_test.py --phase 1 --device cpu
```

---

## 7B. 推荐工作流（统一三分类，基于 `src/*.py` + `configs/liver_unet_3d.yaml`）

该工作流对应标签 `0=背景 / 1=肝脏 / 2=肿瘤` 的统一 3 类分割，是本轮新增/重写的推荐方式。
配置统一写在 `configs/liver_unet_3d.yaml`（spacing=[1.5,1.5,2.0]、roi=[96,96,96]、CT 窗 [-200,250]、AdamW lr=1e-4 wd=1e-5、max_epochs=300、val_interval=2、sw_batch_size=4、seed=42）。

### 7B.1 数据集分布检查 — `src/inspect_dataset.py`

扫描 `imagesTr/labelsTr`，逐例打印 shape / spacing / 强度 / 标签实际取值，保存汇总到 `outputs/dataset_summary.csv`，并随机取 3 例画三方向叠加 PNG：

```bash
python src/inspect_dataset.py \
  --data_dir data/raw/Task03_Liver \
  --output_dir outputs
```

### 7B.2 训练/验证/测试划分 — `src/create_splits.py`

默认 70% / 15% / 15%，固定 `seed=42`，输出 `data/splits/{train,val,test}.json`：

```bash
python src/create_splits.py \
  --data_dir data/raw/Task03_Liver \
  --output_dir data/splits \
  --train_ratio 0.7 --val_ratio 0.15 --seed 42
```

### 7B.3 训练 — `src/train.py`

使用 `configs/liver_unet_3d.yaml` 默认配置；命令行参数可覆盖 YAML 中部分超参（如 `--max_epochs`、`--batch_size`、`--lr`、`--num_workers`）。Windows 下若 DataLoader 报错，建议 `--num_workers 0`。脚本支持从 `outputs/models/last_model.pth` 自动 resume，并把训练日志写入 TensorBoard。

```bash
python src/train.py \
  --config configs/liver_unet_3d.yaml \
  --train_json data/splits/train.json \
  --val_json data/splits/val.json \
  --output_dir outputs/models \
  --max_epochs 300 --batch_size 1 --num_workers 4
```

显存不足时脚本会捕获并提示减小 `roi_size` / `batch_size` / `num_samples`。查看 TensorBoard：

```bash
tensorboard --logdir outputs/models
```

### 7B.4 推理 — `src/infer.py`

加载 `best_model.pth`，使用 sliding window 对新 CT 推理，并用 MONAI Invertd 把预测恢复到原始空间（保留 affine/header），保存为同名 `.nii.gz`：

```bash
# 单例
python src/infer.py \
  --image_path data/raw/Task03_Liver/imagesTr/liver_0.nii.gz \
  --checkpoint outputs/models/best_model.pth \
  --output_dir outputs/predictions \
  --config configs/liver_unet_3d.yaml

# 批量
python src/infer.py \
  --image_dir data/raw/Task03_Liver/imagesTs \
  --checkpoint outputs/models/best_model.pth \
  --output_dir outputs/predictions \
  --config configs/liver_unet_3d.yaml
```

### 7B.5 评价 — `src/evaluate.py`

对 `label=1 (liver)` 与 `label=2 (tumor)` 分别计算 Dice / IoU / Precision / Recall，逐例 + MEAN / STD，保存为 `outputs/evaluation.csv`；tumor 真值为空的病例记 NaN 而不报错。

```bash
python src/evaluate.py \
  --pred_dir outputs/predictions \
  --label_dir data/raw/Task03_Liver/labelsTr \
  --output_csv outputs/evaluation.csv
```

### 7B.6 可视化 — `src/visualize.py`

优先选取含 tumor(label=2) 的若干 axial 切片，输出 CT / CT+GT / CT+Pred / GT vs Pred 四类对比图（liver 红、tumor 绿，非交互 matplotlib）：

```bash
python src/visualize.py \
  --image_path data/raw/Task03_Liver/imagesTr/liver_0.nii.gz \
  --label_path data/raw/Task03_Liver/labelsTr/liver_0.nii.gz \
  --pred_path outputs/predictions/liver_0.nii.gz \
  --output_dir outputs
```

### 7B.7 导出 STL — `src/export_stl.py`

把预测 mask 转为表面网格并保存 STL。**重要：STL 仅供科研/展示，不可用于临床治疗或手术导航**。

```bash
python src/export_stl.py \
  --mask_path outputs/predictions/liver_0.nii.gz \
  --output_dir outputs/stl \
  --labels 1 2 \
  --smooth --min_component_size 100
```

---

## 7C. 速度优化对比实验 — `scripts/benchmark_optimizations.py`

> 📌 项目已在 `src/train.py` 中加入多项训练加速优化：AMP 混合精度、OneCycleLR
> 学习率调度、梯度裁剪、DataLoader `pin_memory` / `persistent_workers` /
> `non_blocking`、滑动窗口验证 AMP、`cache` 策略。本节脚本用于**自动运行短程
> 对比实验**，量化 baseline（关闭全部优化）与 optimized（开启全部优化）两种
> 训练设置在**速度、显存、初步精度**上的差异。**所有结果均来自真实运行日志，
> 禁止编造数据。**

### 7C.1 运行命令

在项目根目录（`d:\项目1`）下，先确认已生成 `data/splits/train.json` 与
`data/splits/val.json`（见 §7B.3），然后执行：

```bash
python scripts/benchmark_optimizations.py --config configs/liver_unet_3d.yaml --train_json data/splits/train.json --val_json data/splits/val.json --max_epochs 10 --batch_size 1 --device cuda
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `configs/liver_unet_3d.yaml` | 提供 `roi_size` / `seed` / `num_workers` 等基础设置（两种模式共用） |
| `--train_json` | `data/splits/train.json` | 训练集划分 JSON |
| `--val_json` | `data/splits/val.json` | 验证集划分 JSON |
| `--max_epochs` | `10` | 短程训练轮数（仅用于速度对比） |
| `--batch_size` | `1` | 批大小（两种模式相同，保证对比公平） |
| `--device` | auto | `cuda` / `cpu`；不填自动检测 |
| `--cache` | `disk` | 缓存策略 `none` / `disk` / `memory`（两种模式共用同一策略） |
| `--output_dir` | `outputs/benchmark` | benchmark 输出根目录 |

### 7C.2 两种模式设置

两种模式使用**完全相同**的 `config / train_json / val_json / max_epochs /
batch_size / roi_size / seed / device`，仅优化开关不同：

| 开关 | baseline | optimized |
| --- | --- | --- |
| AMP 混合精度 | `--no_amp` | `--amp` |
| OneCycleLR 调度器 | `--scheduler none`（恒定 lr） | `--scheduler onecycle` |
| 梯度裁剪 | `--no_grad_clip` | `--grad_clip` |
| `pin_memory` | `--no_pin_memory` | `--pin_memory` |
| `persistent_workers` | `--no_persistent_workers` | `--persistent_workers` |
| `non_blocking` | `--no_non_blocking` | `--non_blocking` |
| 验证 AMP | `--no_val_amp` | `--val_amp` |

> Windows 下 `num_workers=0` 时，`persistent_workers` 会自动关闭，避免
> "persistent_workers needs num_workers>0" 报错。

### 7C.3 输出文件

脚本会依次跑 baseline → optimized，每个模式的训练产物单独存放：

```
outputs/benchmark/
├── baseline/                      # baseline 训练产物（权重 / 缓存 / benchmark.json）
├── optimized/                     # optimized 训练产物
├── benchmark_results.csv          # 逐模式全部指标（含失败原因列）
├── benchmark_results.md           # markdown 对比表
└── benchmark_summary.txt          # 速度/显存/Dice 对比分析与说明
```

`benchmark_results.csv` 记录每个模式以下真实指标（取自 `src/train.py
--benchmark_mode` 产出的 JSON）：

`total_train_time_sec` / `mean_epoch_time_sec` / `median_epoch_time_sec` /
`fastest_epoch_time_sec` / `slowest_epoch_time_sec` / `peak_gpu_memory_MB` /
`best_liver_dice` / `best_tumor_dice` / `final_liver_dice` /
`final_tumor_dice` / `amp_enabled` / `onecycle_enabled` / `grad_clip_enabled` /
`cache_strategy` / `pin_memory` / `persistent_workers` / `non_blocking` /
`val_amp` / `error`（失败原因）。

`benchmark_results.md` 生成如下格式的对比表：

| Setting | AMP | Cache | OneCycleLR | Grad Clip | Mean Epoch Time | Peak GPU Memory | Best Liver Dice | Best Tumor Dice |
|---|---|---|---|---|---:|---:|---:|---:|
| baseline | ❌ | disk | ❌ | ❌ | … | … | … | … |
| optimized | ✅ | disk | ✅ | ✅ | … | … | … | … |

`benchmark_summary.txt` 自动写出：① optimized 相比 baseline 的平均 epoch
时间提升百分比；② optimized 相比 baseline 的显存变化；③ 两者 Dice 差异；
④ 一段说明：**短程 10 epoch benchmark 主要用于速度比较，不代表最终模型性能。**

### 7C.4 重要说明

1. **短程实验**：默认 `max_epochs=10` 仅用于速度对比，不代表最终模型精度。
2. **无 GPU 时**：`peak_gpu_memory_MB` 记为 `NaN`，并提示 CPU benchmark 不代表
   真实训练速度（CPU 上 AMP / `pin_memory` 等优化基本不生效）。
3. **失败容错**：某个模式运行失败时，脚本会记录失败原因写入 CSV/MD，**不会**
   导致整个脚本崩溃；另一个模式仍会继续尝试。
4. **真实数据**：所有指标均来自 `src/train.py --benchmark_mode` 产出的 JSON，
   不写理论值。

### 7C.5 train.py 新增的 benchmark 相关参数

为支持本对比实验，`src/train.py` 新增以下命令行参数（均可覆盖 `config` yaml，
默认值与原项目行为一致，不影响原有训练流程）：

| 参数 | 说明 |
| --- | --- |
| `--amp` / `--no_amp` | 启用 / 关闭 AMP 混合精度（默认关闭，仅 CUDA 生效） |
| `--scheduler {none,onecycle}` | 学习率调度器（默认 `onecycle`，baseline 用 `none`） |
| `--grad_clip` / `--no_grad_clip` | 梯度裁剪开关（默认开启，`max_norm=12.0`） |
| `--grad_clip_max_norm` | 梯度裁剪最大范数（默认 `12.0`） |
| `--pin_memory` / `--no_pin_memory` | DataLoader `pin_memory`（默认 cuda→True） |
| `--persistent_workers` / `--no_persistent_workers` | DataLoader `persistent_workers`（默认 `num_workers>0`→True） |
| `--non_blocking` / `--no_non_blocking` | 数据搬运 `non_blocking`（默认 True） |
| `--val_amp` / `--no_val_amp` | 验证期滑动窗口推理 AMP（默认跟随 `--amp`） |
| `--cache {none,disk,memory}` / `--cache_strategy` | 缓存策略（默认 `disk`） |
| `--benchmark_mode` | 记录每 epoch 的 loss / dice / 耗时 / 显存 / lr |
| `--benchmark_output_json` | benchmark 结果 JSON 输出路径（仅 `--benchmark_mode` 时生效） |

`--benchmark_output_json` 中保存每个 epoch 的：`epoch` / `train_loss` /
`val_liver_dice` / `val_tumor_dice` / `epoch_time_sec` / `lr` / `gpu_memory_MB`，
以及训练结束时的汇总 summary。

---

## 8. 完整流程

以下命令按顺序执行即可完成从数据到结果的完整流程。

### 8.1 环境检查

```bash
python scripts/check_env.py
```

### 8.2 数据检查

```bash
python scripts/check_data.py --data_root ./data/Task03_Liver
```

### 8.3 数据划分

```bash
python scripts/split_data.py --data_root ./data/Task03_Liver --ratio 0.8 0.1 0.1 --seed 42 --output outputs/splits
```

### 8.4 训练（阶段一：二分类）

```bash
python scripts/train.py --config configs/phase1_binary.yaml
```

### 8.5 训练（阶段二：三分类）

```bash
python scripts/train.py --config configs/phase2_ternary.yaml
```

### 8.6 推理

```bash
python scripts/infer.py --config configs/phase1_binary.yaml --model_ckpt outputs/models/best.pt --data_list outputs/splits/test.json
```

### 8.7 评价

```bash
python scripts/evaluate.py --predictions outputs/predictions --labels data/Task03_Liver/labelsTr --config configs/phase1_binary.yaml
```

### 8.8 可视化

```bash
python scripts/visualize.py --image data/Task03_Liver/imagesTr/liver_0.nii.gz --label data/Task03_Liver/labelsTr/liver_0.nii.gz --prediction outputs/predictions/liver_0.nii.gz --output outputs/figures/case0.png
```

### 8.9 STL 三维模型导出

```bash
python scripts/export_stl.py --input outputs/predictions/liver_0.nii.gz --label_id 1 --output outputs/stl/case0_liver.stl
```

---

## 9. 两阶段说明

### 阶段一 (phase1, 二分类 phase1_binary.yaml)

- **类别**：2 类 —— `背景 (0)` 与 `肝脏 (1)`；
- **标签映射**：将原始标注中的 `肝脏 (1)` 与 `肿瘤 (2)` 合并为 `肝脏 (1)`，即 `liver + tumor → liver`；
- **目的**：先定位肝脏整体区域，作为阶段二的感兴趣区域 (ROI)。

### 阶段二 (phase2, 三分类 phase2_ternary.yaml)

- **类别**：3 类 —— `背景 (0)`、`肝脏实质 (1)`、`肿瘤 (2)`；
- **目的**：在阶段一框定的肝脏区域内进一步分割肿瘤，缓解肿瘤占比小、难学习的问题。

> 两阶段策略是医学影像中处理"大器官 + 小病灶"问题的常用思路。

---

## 9B. nnU-Net v2 baseline（与 MONAI 自写代码并行对比）

> 📌 **本节说明**
> nnU-Net v2 是医学影像分割领域的强 baseline，作为本项目自写 MONAI 3D U-Net 的"对照组"使用。
> **不覆盖、不修改**任何 `src/*.py` / `scripts/*.py`（除新增的 `scripts/run_nnunet_baseline.sh`）/ `configs/*` 等 MONAI 自写代码；nnU-Net 仅作为外部工具跑一遍 baseline，结果放在独立的 `nnunet/` 目录，便于和自写模型做 Dice/IoU 指标对比。

### 9B.1 安装：先 PyTorch，再 nnunetv2

nnU-Net v2 强依赖 PyTorch，且**必须先装好匹配本机 CUDA 的 PyTorch**，再 `pip install nnunetv2`，否则会出现 CUDA 不可用或 hidden import 失败。请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 命令：

```bash
# 1) 先装匹配 CUDA 的 PyTorch（例如 CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) 再装 nnU-Net v2（会拉取配套依赖）
pip install nnunetv2
```

> 建议在**独立 conda 环境**中安装 nnU-Net，避免和本项目的 MONAI 环境互相干扰：

```bash
conda create -n nnunet python=3.10 -y
conda activate nnunet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2
```

### 9B.2 设置三个环境变量

nnU-Net v2 通过三个环境变量定位「原始数据 / 预处理结果 / 训练结果」三个目录。请在运行任何 nnU-Net 命令前设置（Linux/macOS 用 `export`，Windows PowerShell 用 `$env:`）：

```bash
# Linux / macOS / Git Bash
export nnUNet_raw="./nnunet/nnUNet_raw"
export nnUNet_preprocessed="./nnunet/nnUNet_preprocessed"
export nnUNet_results="./nnunet/nnUNet_results"
```

```powershell
# Windows PowerShell
$env:nnUNet_raw="./nnunet/nnUNet_raw"
$env:nnUNet_preprocessed="./nnunet/nnUNet_preprocessed"
$env:nnUNet_results="./nnunet/nnUNet_results"
```

> 三个目录会自动创建；**务必在每次新开终端时重新设置**，或写进 `~/.bashrc` / 系统环境变量。

### 9B.3 转换 MSD 数据为 nnU-Net v2 格式

nnU-Net v2 提供官方转换器，把 MSD Task03_Liver 转成 nnU-Net v2 的 `imagesTr/labelsTr` 命名规则。`DATASET_ID` 在本项目约定为 `3`（与 MSD 的 Task 编号保持一致，nnU-Net 内部记为 `Dataset003_Liver`）：

```bash
nnUNetv2_convert_MSD_dataset -i data/raw/Task03_Liver -d 3
```

转换成功后会在 `$nnUNet_raw/Dataset003_Liver/` 下生成 `imagesTr/`、`labelsTr/` 与 `dataset.json`。

### 9B.4 预处理与规划（含完整性校验）

```bash
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
```

`--verify_dataset_integrity` 会检查图像/标签一一对应、形状一致、标签取值合法（Task03_Liver 应为 `{0,1,2}`）。预处理耗时较长，仅在首次运行。

### 9B.5 训练（3D full resolution，fold 0）

```bash
nnUNetv2_train 3 3d_fullres 0
```

- 第一个参数 `3` = `DATASET_ID`
- 第二个参数 `3d_fullres` = 配置类型（也可用 `2d` 或 `3d_lowres`）
- 第三个参数 `0` = fold 编号（5 折交叉验证中的第 0 折）

> ⚠️ **显存不足怎么办？** nnU-Net 3D fullres 默认配置较吃显存（建议 ≥ 11GB）。若遇 OOM，可按以下顺序降级：
> 1. 先训练 2D 配置做快速 baseline：`nnUNetv2_train 3 2d 0`
> 2. 训练 3D lowres（下采样配置，显存更小）：`nnUNetv2_train 3 3d_lowres 0`
> 3. 或在命令后加 `--c` 续训 checkpoint，减少单次显存峰值。
> nnU-Net 会自动选择适合的 patch size 与 batch size；OOM 通常是 GPU 显存本身不足，降级配置最稳妥。

### 9B.6 推理

对一批新图像（放入一个文件夹，`.nii.gz`）做推理：

```bash
nnUNetv2_predict \
  -i INPUT_FOLDER \
  -o OUTPUT_FOLDER \
  -d 3 \
  -c 3d_fullres \
  -f 0
```

- `-i INPUT_FOLDER`：待预测 CT 所在目录（仅 `.nii.gz`）
- `-o OUTPUT_FOLDER`：预测 mask 输出目录（标签 `{0,1,2}`）
- `-d 3`：`DATASET_ID`
- `-c 3d_fullres`：与训练一致的配置
- `-f 0`：使用 fold 0 的 checkpoint
- 想用多折集成可写 `-f 0 1 2 3 4`

### 9B.7 一键脚本

把以上步骤打包进 `scripts/run_nnunet_baseline.sh`，按需注释掉已跑过的步骤：

```bash
bash scripts/run_nnunet_baseline.sh
```

> 脚本默认从头跑（转换 → 预处理 → 训练 → 推理）。可编辑脚本，把已完成的步骤注释掉，只跑剩余步骤。

### 9B.8 与自写 MONAI 模型对比

拿到 nnU-Net 的预测 `OUTPUT_FOLDER/*.nii.gz` 后，可直接复用本项目的评价脚本对 liver/tumor 分别算 Dice/IoU/Precision/Recall：

```bash
python src/evaluate.py \
  --pred_dir <nnU-Net OUTPUT_FOLDER> \
  --label_dir data/raw/Task03_Liver/labelsTr \
  --output_csv outputs/evaluation_nnunet.csv
```

再把 `outputs/evaluation_nnunet.csv` 与自写 MONAI 的 `outputs/evaluation.csv` 同列对比，即可得到 baseline vs 自写模型的差距表格。

---

## 10. notebooks 教程指引

- `notebooks/exploration.ipynb`：**数据探索** —— 查看 CT 体数据、标签分布、切片浏览、HU 值直方图等；
- `notebooks/tutorial_zh.ipynb`：**中文分步教程** —— 从读图、预处理、构建模型、一步训练、一步推理到画图的全中文讲解。

> 建议先跑 `tutorial_zh.ipynb` 熟悉流程，再使用完整脚本进行正式实验。

---

## 11. 常见错误与排查

> 下表汇总命令行模式与 GUI 模式下最常见的 9 类问题。GUI 模式下所有错误都会以
> `[ERROR]` 级别写入 Logs 页（见 §6B.4），可先查看日志定位。

| 问题 | 可能原因 / 解决方案 |
| --- | --- |
| **CUDA 不可用 / `torch.cuda.is_available()` 返回 False** | 1) 未安装对应 CUDA 版本的 PyTorch；2) 显卡驱动过旧；3) 误装了 CPU 版 PyTorch。请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 重新安装匹配本机 CUDA 的版本（GUI 环境页可粘贴官方命令一键安装）。安装后用 `python -c "import torch; print(torch.cuda.is_available())"` 验证。 |
| **显存不足 (CUDA OOM)** | 3D 体积训练显存压力大。解决：① 调小 `batch_size`（最小到 1）；② 调小 `roi_size`（如 96→64）；③ 减少 `RandCropByPosNegLabeld` 的 `num_samples`（默认 4，可改 2 或 1）；④ 关闭其它占显存的程序；⑤ 在 GUI 训练页把 `device` 改为 `cpu`（仅小规模可行）。 |
| **nii.gz 路径错误 / FileNotFoundError** | 1) 数据未解压到 `data/raw/Task03_Liver/`，核对 GUI 数据页选的目录；2) splits JSON 里的路径是相对/绝对路径不一致 —— 重新跑「划分数据集」生成新 JSON；3) 推理页选的 `image_path` / `image_dir` 不存在或含中文空格，建议用纯英文路径；4) checkpoint 路径拼错（默认 `outputs/models/best_model.pth`）。 |
| **label 值不正确 / 标签异常** | MSD Task03_Liver 标签应为 `{0=背景, 1=肝脏, 2=肿瘤}`。1) 若标签文件被改成其它值，会导致 `DiceCELoss` 与 `AsDiscrete(to_onehot=N)` 维度不匹配报错；2) 二分类模式（`--task_mode binary_liver`）会自动把 1/2 都映射为 1，无需手动改标签；3) 用 `src/inspect_dataset.py` 检查每个病例的标签唯一值，发现异常值时回溯原始数据。 |
| **推理结果为空 / 预测全是背景** | 1) checkpoint 与 config 不匹配（如用二分类模型加载三分类 config，或反之），核对 `task_mode` 与训练时一致；2) `--config` 里的 `roi_size` / `ct_window` 与训练时不一致，导致输入分布偏移；3) 模型欠拟合（训练 epochs 太少 / loss 没下降），查看 TensorBoard 日志确认；4) CT 窗 `[-200,250]` 设错导致图像被截断成全黑；5) 推理时 `device` 选 cpu 但模型在 GPU 训练（或反之）通常不影响，但可检查 `map_location`。 |
| **STL 太粗糙 / 表面有锯齿** | 1) 开启 `smooth`（Taubin 平滑）让表面更光滑；2) 调大 `min_component_size` 过滤掉小碎片噪声（默认 100，可试 500/1000）；3) 原始 mask 分辨率低（spacing 大）会直接限制 STL 精度，可先用更小 spacing 重采样后再导出；4) mask 边缘本身不连续（如二值化阈值不当），检查预测 mask 的切片预览。 |
| **Windows 下 num_workers 报错 / DataLoader 卡死** | Windows 多进程在交互式环境下易出错，请在 GUI 训练页或 yaml 里把 `num_workers` 设为 `0`。 |
| **splits 未生成 / 找不到 test.json** | 先在 GUI 数据页点「划分数据集」（或运行 `src/create_splits.py`）生成 `data/splits/*.json`，再运行推理/评估。 |
| **配置字段缺失 / KeyError** | 检查 yaml 配置是否完整，参考 `configs/liver_unet_3d.yaml` 所需字段；GUI 训练页点「保存配置」会自动写回 yaml，避免手改遗漏。 |

---

## 12. 指标说明

本项目使用以下分割指标，并**对 liver 与 tumor 分别统计**：

- **Dice (Dice Similarity Coefficient)**：预测与真值的体素重叠程度，越接近 1 越好；
  `Dice = 2|A∩B| / (|A|+|B|)`
- **IoU (Intersection over Union / Jaccard)**：交并比，`IoU = |A∩B| / |A∪B|`；
- **Precision (精确率)**：预测为正的体素中真正为正的比例，衡量"误检"程度；
- **Recall (召回率)**：真值为正的体素中被正确预测为正的比例，衡量"漏检"程度。

> **空真值处理**：若某病例中该类别在真值中不存在（如某病例无肿瘤），则该类别的指标记为 `NaN`，不参与均值计算，避免误导。

---

## 13. 许可证与免责（再次强调）

> **医学声明**：本项目仅用于科研学习与展示，不用于临床诊断、治疗决策或手术导航。

- 本项目**仅用于科研学习与展示**；
- **不用于临床诊断、治疗决策或手术导航**；
- **不使用任何真实患者隐私数据**，所用数据为公开 MSD Task03_Liver；
- 使用者应遵守 MSD 数据集与相关依赖库的许可协议；
- 项目作者不对任何因使用本项目或其输出结果而造成的后果承担责任。

> 医学影像 AI 仅供学习，临床决策请以执业医师判断为准。
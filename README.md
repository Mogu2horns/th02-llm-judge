# TH02 LLM 回答偏好预测

本工程根据用户问题及匿名回答 A/B，预测 `winner_model_a`、`winner_model_b`、`winner_tie` 三类人工偏好概率。主指标采用题目提供的三分类 Log Loss，固定使用官方 train/dev。

已执行的验收、实测小规模指标和全量资源状态见 [VALIDATION.md](VALIDATION.md)。

训练方案包括 CPU 上的词级 TF-IDF 与逻辑回归，以及冻结 MiniLM、统计特征和小型 MLP 构成的语义分支。最终在训练集内部选择概率融合权重和温度，再统一评估官方 dev。

## 环境与运行状态

本机检查结果为 Ryzen 7 5800H、8 核 16 线程、约 15.86 GiB 内存、RTX 3060 Laptop 6 GiB 显存，NVIDIA 驱动 546.30。已有 `mypytorch` 环境的 CUDA 张量运算验证通过，默认 Anaconda 的 PyTorch 为 CPU 版本。

项目环境使用 Python 3.11，主要依赖固定为 PyTorch 2.5.1＋CUDA 12.1、Transformers 4.48.3、Sentence Transformers 3.4.1、scikit-learn 1.6.1。完整直接依赖见 [requirements-training.txt](requirements-training.txt)，安装完成后生成 `requirements-lock.txt`，记录实际安装的全部版本。`torch==2.5.1` 同时支持本机 Conda CUDA 构建的版本元数据，实际 CUDA 版本由安装脚本验证。

全量训练要求启动时可用内存至少 6 GiB，CUDA 模式另要求可用显存至少 2048 MiB。CPU 线程上限为 8，实验串行执行。资源不足时进程返回退出码 2，在 `run_state.json` 记录 `blocked_resources` 及原因，资源恢复后可重新执行同一命令。

实现阶段的全量预检曾记录可用内存约 0.87 GiB。当前运行状态以 `artifacts/run_state.json` 及 `artifacts/metrics.json` 为准。`metrics.json` 中的 `smoke` 标记用于区分完整实验和小规模流程验证。

## 安装与运行

### 从 GitHub 下载到 Linux 服务器

```bash
git clone https://github.com/Mogu2horns/th02-llm-judge.git
cd th02-llm-judge
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

公开仓库包含代码、依赖配置、数据切分元信息和文档。依据题目第八节的数据使用边界，`data/train.csv`、`data/dev.csv` 及包含样本文本的运行产物不上传；`.venv`、`.cache` 和训练输出也不上传。训练数据须由获授权的使用者单独传至服务器的 `data/` 目录，保留原文件名与内容。本机项目目录下可执行以下命令，将 `USER`、`SERVER` 和目标路径替换为实际值。

```powershell
scp data/train.csv data/dev.csv USER@SERVER:/path/to/th02-llm-judge/data/
```

数据就位后，在 Linux 服务器执行以下命令校验原始文件、检查资源并启动 CUDA 训练。

```bash
sha256sum -c MANIFEST.sha256
python -m th02.run --data-dir data --output-dir artifacts --seed 20260916 --preflight-only
python -m th02.run --data-dir data --output-dir artifacts --seed 20260916
```

CUDA 运行需要兼容 CUDA 12.1 的 NVIDIA 驱动；CPU 运行时在命令末尾添加 `--device cpu`。首次训练会下载固定 revision 的 MiniLM 权重，离线服务器须预先准备模型并传入 `--model-path /path/to/model`。`requirements-training.txt` 固定跨平台直接依赖，`requirements-lock.txt` 和 Conda 显式锁记录原 Windows 验收环境；Linux 完成安装后可用 `python -m pip freeze` 保存该服务器的实际依赖版本。

下文链接到 `artifacts/` 和 `artifacts-smoke*/` 的验收证据保留在本机，需单独传输或重新运行生成。

### 本机 Windows 环境

以下命令在 PowerShell 中执行，工作目录为本文件所在目录。本机推荐复用已缓存的 Conda CUDA 包。

```powershell
Set-Location -LiteralPath 'E:\lixiangExam\C-TH-02-v1.1'
.\setup.ps1 -UseCondaCache -CondaPython 'D:\anaconda\python.exe'
```

缓存模式使用 Conda 离线创建项目私有 `.cache/cuda-runtime`，包含 Python 3.11.11、PyTorch 2.5.1 和 CUDA 12.1，再从该运行时创建 `.venv`。这里的 `--system-site-packages` 指向项目私有运行时，其包和构建信息由 `environment-conda-explicit.txt` 记录。后续 pip 依赖及版本由 `requirements-lock.txt` 记录。

本机 CUDA PyTorch wheel 下载曾遇到超时，包体约 2.4 GB。缓存模式复用已下载的 Conda 包完成 CUDA 运行时准备，随后按锁定要求安装其余 Python 依赖。

常规 pip 安装入口使用指定的 Python 3.11 创建项目 `.venv`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -PythonPath 'D:\anaconda\envs\gnnProject\python.exe'
```

两种入口均将安装临时文件和 pip 缓存放入项目 `.cache`，安装依赖并生成锁定文件，最后验证实际 CUDA 12.1 运行环境。下载或安装失败时修复对应原因后重跑安装命令。缓存模式要求本机 Conda 包缓存具备所需版本。

全量训练与交付产物生成使用以下固定入口。

```powershell
.\.venv\Scripts\python.exe -m th02.run --data-dir data --output-dir artifacts --seed 20260916
```

仅检查数据指纹、硬件和资源门槛时执行以下命令。

```powershell
.\.venv\Scripts\python.exe -m th02.run --data-dir data --output-dir artifacts --seed 20260916 --preflight-only
```

CUDA 小规模验收使用前 512 条 train、前 128 条 dev，以及最多 2 轮 MLP 训练。

```powershell
.\.venv\Scripts\python.exe -m th02.run --data-dir data --output-dir artifacts-smoke-cuda --seed 20260916 --smoke
```

CPU 小规模验收可用于检查全部训练和评估接口。

```powershell
.\.venv\Scripts\python.exe -m th02.run --data-dir data --output-dir artifacts-smoke-cpu --seed 20260916 --smoke --device cpu
```

小规模验收跳过全量 6 GiB 内存门槛，仍检查所选 CUDA 设备的显存门槛。CPU 编码使用 FP32，CUDA 编码使用 FP16。比较正式模型效果时使用同一完整实验目录内的官方 dev 结果。

`--model-path` 可指定本地模型目录。默认优先使用本机缓存的 `all-MiniLM-L6-v2`，固定参考 revision 为 `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`。编码缓存记录实际模型文件指纹。

相同输出目录绑定数据指纹、随机种子、设备和模型路径，配置发生变化时使用新的 `--output-dir`。目录锁控制同一目录的并发写入。编码过程按持久化批次恢复，重新运行训练会重新拟合分类器并复用兼容的编码缓存。

## 数据处理与切分

官方 train 包含 45,982 条样本，dev 包含 5,748 条样本。读取使用 `utf-8-sig`，校验唯一 ID、三列 one-hot 标签、JSON 数组元素类型以及 Prompt/A/B 轮数一致性。数组中的 `null` 转为空文本，同时记录缺失轮数。文本保留 `[TURN i]` 轮次标记。异常包含样本 ID，便于定位修复。

所有样本保留用于指定切分。完整对话先做 Unicode NFKC 与空白折叠，保留大小写和轮次边界，再计算内容哈希。训练集内部使用 `StratifiedGroupKFold` 十折，随机种子固定为 `20260916`。第 0 折用于校准，第 1 折用于调参，其余八折用于初始拟合。

全量数据的实际分配如下。

| 用途 | 条数 |
|---|---|
| 初始拟合 | 36,804 |
| 调参 | 4,596 |
| 校准 | 4,582 |
| 配置确定后重训 | 41,400 |

调参确定逻辑回归的 C 和 MLP 轮数后，将初始拟合与调参部分合并为约 90% 训练数据，重新拟合向量器、标准化器与分类器。保留的约 10% 校准数据用于选择融合权重和温度，使校准依据来自模型拟合之外。官方 dev 用于配置锁定后的统一对照评估。

### 长文本和空回答

- TF-IDF 每字段最多保留 16,000 字符，超出时分别取头尾 8,000 字符。
- MiniLM 每字段最多 256 token，扣除特殊 token 后平均分配头尾预算。
- 每种截断记录发生比例与样本 ID，空回答及 `null` 轮数另作为统计特征保留。

### 重复与泄漏审计

更强的 NFKC 与空白折叠规则发现 train 中规范化重复额外 71 行、dev 中 2 行，train 中有 4 组同内容标签冲突。两个集合有 700 个规范化问题重合，以及 1 个完整对话规范化重合组。相关内容和标签证据写入 `data_profile.json`。

这一完整对话组存在问题尾随空格差异，且训练标签存在分歧，具体原文、样本 ID 和标签证据保留在本机审计产物中，不随公开仓库发布。官方切分用于报告主指标，内部切分按规范化内容分组。语义近似对话仍可能跨集合出现，使验证结果偏乐观，相同问题下不同回答的重合程度也需结合文本解释。

## 模型与训练参数

### 词面 Baseline

Prompt 使用独立 TF-IDF 词表，上限 20,000 特征。A/B 使用共享回答词表，上限 40,000 特征，分别编码后与 Prompt 特征拼接。TF-IDF 使用 word 1–2 gram、`min_df=3`、`sublinear_tf=True`、`float32`，允许单字符词元。

逻辑回归使用 `saga`、`max_iter=1000`、`tol=1e-3`。在内部调参集比较 `C∈{0.1,1,10}`，按 Log Loss 选择。最终模型记录词表大小、训练轮数、收敛状态、耗时和每次调参结果。预测按 512 条分批执行。

训练集类别频率构成先验预测，用于检查评估流程。Baseline 同时报告原始概率和单独温度校准后的结果。

### 冻结语义分支

`all-MiniLM-L6-v2` 全程冻结并处于评估模式。Prompt/A/B 分别编码，采用 attention mask 均值池化、FP32 汇聚和 L2 归一化，产生三侧 384 维嵌入。CUDA 编码初始 batch size 为 32，显存不足时依次降至 16、8。小规模验收初始 batch size 为 8。

语义交互特征拼接 `P、A、B、A−B、|A−B|、P×A、P×B`，共 2,688 维。另有 18 维统计特征。

- A/B 各 6 维，分别为字符数、词数、非空轮数、null 轮数、代码围栏次数和全空标记。
- Prompt 的字符数与轮数，以及 A/B 的对数字符数差，共 3 维。
- `cos(P,A)`、`cos(P,B)`、`cos(A,B)`，共 3 维。

计数使用 `log1p`，18 维统计特征在对应拟合数据的原顺序与交换顺序上标准化。交互特征按 batch 构造。三侧 float32 嵌入采用磁盘映射缓存，完整 train/dev 约占 0.24 GB。

分类头为 `2706→128→3`，中间使用 ReLU 和 dropout 0.2。优化使用交叉熵与 AdamW，学习率 `1e-3`、weight decay `1e-4`、batch size 256。最多训练 20 轮，以内部调参集的对称预测 Log Loss 选择轮数，连续 3 轮无改善时早停。重训使用选定轮数和固定种子。

每个训练样本每轮以 50% 概率交换 A/B，并同步交换 A/B 胜标签。预测时对原顺序和交换顺序分别计算概率，将交换后的 A/B 类别映射回原顺序再平均。平局类别保持对应关系。

### 融合、校准与消融

最终融合采用 `p=(1−α)p_lexical＋αp_semantic`，两分支均先做 A/B 对称预测。

在内部校准集枚举 `α∈{0,0.25,0.5,0.75,1}` 及 `T∈{0.5,0.75,1,1.25,1.5,2,3}`，按 Log Loss 选择。温度变换为 `softmax(log(p)/T)`。原始 Baseline 和对称 Baseline 分别选择各自温度。所有配置在官方 dev 预测前保存到 `models/selection_locked.json`。

实验统一输出先验、Baseline、校准 Baseline、对称 Baseline、校准对称 Baseline、语义分支、融合模型、校准融合模型。融合权重允许选择 0，报告依据实测结果说明语义增强的贡献。

## 评估、错误分析与 AI 记录

每个预测 CSV 均使用列 `id,winner_model_a,winner_model_b,winner_tie`，概率校验覆盖有限性、取值范围、行和及完整 ID 集合。Log Loss 由题目原始 `evaluation/metric.py` 的 `score()` 复算，并与工程计算结果核对。另报告准确率、各类别 precision/recall 和 Brier 分数。

完整实验可用以下命令再次复算最终融合结果。

```powershell
.\.venv\Scripts\python.exe -c "from th02.evaluation import score_official; print(score_official('data/dev.csv', 'artifacts/predictions/fusion_calibrated.csv'))"
```

报告流程生成 12 个按内容去重的错误或高不确定候选，覆盖高置信错误、低置信预测、改进后退步和可观测文本特点。每例包含真实标签、预测概率、原文及截断信息。

**错误分析需要逐例复核。** 复核时检查原始文本、标签、截断影响和两模型差异，补充判断证据，并依据实际案例归纳至少 3 类失败模式。自动生成的候选、标签预测和模板化观察需经过这一核验过程，才能形成题目要求的错误分析材料。

AI 使用材料见项目 `AI_USAGE.md`，需记录工具、用途、产出、采纳／修改／否决情况和核验方式，并保留完整 Session 或关键对话与操作记录。核验包括代码审阅、测试、实际训练日志和官方指标复算。数据与含样本文本的产物按本题授权的招聘测评范围保管和使用。

## 产物布局与验收

下列路径相对于 `--output-dir`。全量默认输出到 `artifacts`，小规模验收使用独立目录。

| 路径 | 内容 |
|---|---|
| `run_state.json` | 配置绑定、运行状态和失败原因 |
| `preflight.json` | CPU/GPU/内存检查与原始包 SHA256 校验 |
| `data_profile.json` | 类别、长度、空回答、重复、冲突与泄漏审计 |
| `splits.json` | 内部分组切分数量与样本 ID |
| `baseline_training.json` | Baseline 调参、收敛情况与耗时 |
| `models/baseline.joblib` | TF-IDF 向量器与逻辑回归 |
| `models/semantic.pt` | MLP 权重、统计标准化器与训练设置 |
| `models/calibration.json` | 融合权重、温度及校准网格结果 |
| `models/selection_locked.json` | 官方 dev 预测前锁定的完整配置 |
| `cache/train`、`cache/dev` | 编码数组、指纹、截断信息与批次恢复状态 |
| `predictions/*.csv` | 各实验的三分类预测概率 |
| `metrics.json` | 实测指标、官方复算、依赖、耗时与对称性检查 |
| `REPORT.md` | 本次实验报告 |
| `ERROR_ANALYSIS.md`、`error_cases.json` | 错误分析候选、文本证据及复核材料 |

主指标和完成状态以对应运行目录的记录为准。退出码 0 表示所请求流程成功，1 表示异常，2 表示资源门槛阻塞。模型、预测文件、报告和关键 JSON 元数据采用原子替换方式写入，以便安全重试。

单元测试使用以下入口。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖 JSON/null/多轮解析、异常 ID 定位、分组隔离、字符与 token 头尾保留、A/B 交换、预测概率、保存重载、概率融合校准和官方指标复算。小规模流程验收还应使用相同种子重复运行，核对预测和指标，并结合完整实验的对称性检查确认最终输出满足 A/B 对应关系。

原始题目、数据、模板、`requirements.txt` 与评分文件通过 [MANIFEST.sha256](MANIFEST.sha256) 验证，工程新增内容独立于这些原始文件。

本次小规模运行的逐例证据分析见 [ERROR_ANALYSIS_REVIEWED.md](artifacts-smoke-cuda/ERROR_ANALYSIS_REVIEWED.md)，包含 12 个案例及四类观测到的失败模式。完整实验生成对应错误候选后，按同样的证据核验流程完成其分析。

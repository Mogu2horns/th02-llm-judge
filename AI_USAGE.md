# AI 使用与关键操作记录

## 记录范围

本文件保存本题关键对话、操作、修改与核验摘要。用户可以结合 Codex 客户端中的本次聊天复核原始工具调用和完整对话。记录日期为 2026 年 11 月 10 日，时区为 Asia/Shanghai。

## 关键对话

1. 用户要求阅读 TH02，说明模型问题和最终交付形式。AI 阅读题目、模板、数据清单和官方评分函数，明确三分类人工偏好概率任务以及四类交付材料。
2. 用户要求检查本机 CPU/GPU，并根据必须完成第 1–6 条设定训练任务。AI 只读检查 CPU、内存、GPU、Python、Conda 环境、模型缓存及数据。
3. 用户选择 GPU 语义增强，并允许在项目中建立独立环境、下载依赖和权重。
4. AI 提交 Baseline、冻结 MiniLM、内部调参和校准、错误分析与可复现交付计划。用户随后明确要求实施完整计划。

## 工具与产出

| 工具或阶段 | 建议或产出 | 采纳／修改／否决情况 | 核验方式 |
|---|---|---|---|
| Codex 代码代理与并行子代理 | 数据与词面模型、语义分支、评估与报告模块 | 按批准计划采纳，主代理统一接口和入口 | 阅读实现、单元测试、集成运行及产物复算 |
| PowerShell、注册表、psutil、nvidia-smi | CPU、内存、GPU、Python 环境事实 | 采用实际检查结果确定资源门槛与版本 | CUDA 张量矩阵运算、真实可用内存和驱动查询 |
| Python 标准库与 hashlib | 原始文件校验、CSV/JSON 数据审计 | 采纳，保留官方数据与评分函数 | MANIFEST 全部八项 SHA256 校验 |
| scikit-learn | TF-IDF、逻辑回归、分层分组切分和数值标准化 | 采纳，并限定拟合数据范围 | 分组互斥、模型保存重载和交换视图测试 |
| PyTorch 与 Transformers | 冻结 MiniLM 和小型分类头 | 采纳，分批构造特征并缓存嵌入 | 缓存中断恢复、2706 维特征、固定种子和模型重载测试 |
| 自有校准与评估代码 | 概率融合、温度选择、预测格式和错误候选 | 采纳，使用内部校准折选择参数 | 概率合法性、交换对称性和原始 metric.score 复算 |
| 模型内存方案评估 | 将稠密语义交互特征直接转为稀疏矩阵并拼接 TF-IDF | 否决，改用独立分支与概率融合，减少稠密特征索引开销 | 检查分批特征构造与实际缓存大小 |
| 自动错误候选生成 | 12 个内容去重的错误或高不确定案例 | 作为逐例审阅材料，结论需依据原文补充 | 核对完整多轮文本、标签、概率和截断元数据 |

## 重要修改与实测事实

- 读入采用 `utf-8-sig`，保留多轮顺序，null 转为空文本并单独计数。
- 更强的 NFKC 与空白规范化发现一组 train/dev 空白近重复：问题存在尾随空格差异，两侧回答相同，训练标签存在分歧。具体原文、样本 ID 和标签保留在本机审计产物中，公开仓库仅记录统计描述。
- 对同一输出目录增加操作系统文件锁，避免多个进程并发修改缓存、模型和预测。
- 完整嵌入缓存校验样本文本和实际模型文件指纹，部分缓存通过进度元数据继续计算。
- Windows 沙箱默认临时目录存在写入权限问题，环境准备和测试临时目录改到项目内。
- 独立环境的 CUDA PyTorch 网络下载曾超时，最终复用本机缓存构建项目私有 `.cache/cuda-runtime`，再创建以它为基础的 `.venv`。实际版本为 PyTorch 2.5.1、CUDA 12.1，Conda build 为 `py3.11_cuda12.1_cudnn9_0`。Conda 显式锁与 pip 版本锁共同记录来源和依赖。
- 现有 gnnProject 环境的 CPU 集成尝试在加载 BERT 时遇到 `torchvision::nms` 版本冲突。该次失败保存在对应运行状态，后续验收使用项目独立环境。
- 最终 26 项单元测试在项目独立环境通过，日志见 `artifacts/validation-tests.log`。测试覆盖中断恢复、交换、重载、报告状态、依赖锁和评分等具体行为。
- 两次 GPU 小规模实验均使用 512 条 train 和 128 条 dev。八份预测 CSV 逐字节一致，指标完全一致，对称性最大误差为 0，见 `artifacts/reproducibility.json`。
- 小规模 Baseline Log Loss 为 1.099693312，校准融合为 1.128874178，内部选择为 alpha 0、温度 0.5。这次结果如实保留，模型配置由训练集内部选择。
- 12 个小规模案例已由 AI 辅助逐例审阅，证据分析见 `artifacts-smoke-cuda/ERROR_ANALYSIS_REVIEWED.md`。括号位置、逐字符序列和三角函数结果另用 Python 核验。
- 全量入口资源门槛为 6 GiB 可用内存及 2 GiB 可用显存，资源不足时退出码为 2，并生成 `run_state.json`。
- 独立环境完成后的全量启动曾因可用内存 1.58 GiB 而受阻，最终复检为 1.24 GiB。完整 dev 指标和对应模型错误分析等待资源满足后生成。

## 复核入口

运行命令、依赖和模型说明见 README。实际训练状态以对应输出目录的 `run_state.json` 为准。实验指标以 `metrics.json` 和预测 CSV 为准，使用原始 `evaluation/metric.py` 复算。小规模运行在报告中明确标记其样本范围。

错误分析的自动观测与最终逐例核验分别记录，避免将自动候选筛选视作已完成的证据分析。

## GitHub 上传准备

2026 年 11 月 10 日，按用户要求将项目提交至 `Mogu2horns/th02-llm-judge`。上传前确认目标仓库为公开仓库，检查源码和配置中的凭据及大文件；未发现常见密钥，`data/train.csv` 超过 GitHub 普通 Git 单文件 100 MiB 限制。

依据题目第八节的数据使用边界，将测评 CSV、包含样本文本的运行产物、本地环境和缓存排除出提交；README 和本文件中的原始样本摘录改为统计描述，原始数据文件与 MANIFEST 保持原样。README 补充 Linux 服务器克隆、安装、单独传输数据、校验与训练命令。具体提交和推送结果以 Git 历史及远程分支为准。

## Linux 服务器训练记录（2026 年 11 月 10 日）

用户在 `/disk1/wuchufeng/th02-llm-judge` 补齐原始数据和代码后，明确要求按题目完成模型训练，并强调三分类 Log Loss 及 `[1e-15, 1-1e-15]` 概率裁剪。此次由 Codex 主代理执行，子代理并行审查评估流程、修正多 GPU 预检、准备模型下载和独立验收脚本。

- 原始包八项 SHA256 均通过；训练与验证分别为 45,982 和 5,748 条。独立审计结果保存在 `logs/data-validation.json`。
- 使用现有 Python 3.11 解释器创建项目独立 `.venv`，不继承其他项目第三方包，安装 `requirements-training.txt` 固定的直接依赖。原 Windows 锁文件保留，服务器依赖另存 `requirements-lock-linux.txt`。
- 独立环境初始 pip 22.3 对新依赖元数据解析失败，升级项目 pip 后改为本地已缓存 PyTorch wheel 配合 PyPI 补齐依赖；安装日志保留在 `logs/`。
- 默认 Hugging Face 下载接口因响应元数据不兼容失败；改用固定 revision 的官方 API 和 resolve 地址下载六个必要文件。权重 SHA256 与官方 LFS 元数据一致，其余文件的 Git blob SHA1 一致。来源与哈希保存于 `models/all-MiniLM-L6-v2/provenance.json`，可用 `scripts/download_encoder.py` 复验。
- 修复 `th02/run.py` 的显存预检：使用当前 CUDA 逻辑设备 UUID 查询物理 GPU，避免忽略 `CUDA_VISIBLE_DEVICES` 后检查到另一张卡。新增三个聚焦回归测试；迁移不修改模型结构、数据切分、超参网格和主指标。
- `train_server.sh` 固定项目数据与模型路径、种子和缓存目录，通过 `TH02_GPU` 选择 GPU；正式输出为 `artifacts/`，流程验收输出为 `artifacts-smoke-server/`。
- CUDA FP16 运算、`pip check` 和 29 项测试通过；真实 MiniLM 的小规模端到端运行完成，八组指标与官方评分函数一致。此子集结果不作为全量结论。
- 首次全量尝试在 TF-IDF 阶段出现大量内核 CPU 时间。只读核查排除了 cgroup 内存上限及进程交换，发现训练内存块的大页提示和宿主巨大页整理失败计数增长；由于内核栈访问被权限拒绝，根因仍以 THP 整理推断记录。保存首次日志后停止该任务自有进程，通过 `scripts/run_server.py` 在导入 NumPy/Torch 前设置 `PR_SET_THP_DISABLE=1` 和 `NUMPY_MADVISE_HUGEPAGE=0`，核验成功后以相同配置重启；未修改主机 sysctl、其他任务或模型参数。证据与首次日志见 `logs/thp-investigation.json`、`logs/full-training-thp-first-attempt.log`。
- `scripts/verify_training.py` 独立调用未修改的官方评分器检查八组预测，记录数据、评分代码、模型和预测哈希，并补充置信度与 ECE 诊断。诊断不用于重新选择模型或校准配置。

最终训练状态、实测指标与错误分析的完成情况，以本次 `artifacts/run_state.json`、`artifacts/metrics.json`、`artifacts/verification.json` 及已复核分析文件为准；上文 Windows 小规模结果仅为历史记录。

### Linux 正式训练与最终数值核验

进程级内存规避后，完整流程正常完成（`smoke=false`），成功运行耗时 748.63 秒。TF-IDF Baseline dev Log Loss 为 1.050704936，语义分支为 1.028195140，校准融合为 1.022189513；alpha=0.5、T=0.75 均由训练集内部校准折选定，未根据 dev 再改参数。

独立官方复算八组预测全部通过（误差不超过 1e-12）。已保存 Baseline、MLP 和校准参数的全 dev 重载预测逐值一致，最大绝对差为 0。原始 Baseline 单独校准略有退步，也保留在对照表中；没有只挑选有利指标。验证记录为 `artifacts/verification.json` 与 `artifacts/reload_verification.json`。

### 全量错误分析复核

三名并行子代理分别复核 1–4、5–8、9–12 号案例，主代理阅读三份完整分析，检查证据和结论边界，并合并为 `artifacts/ERROR_ANALYSIS_REVIEWED.md`。12 例均包含具体文本差异、标签、预测概率、单例 Log Loss、截断及结论边界，归纳五类交叉失败模式。使用本地分词器定位实际丢失的否定、轮次边界和条件限定，安全自编小计算检查数量/概率关系及逐字相等性；未执行样本代码或联网外传样本。专业事实与标注理由未作超出证据的认定，长度或截断的因果机制仍标为假设。自动候选原件保留，AI 复核状态另存 `artifacts/error_review_status.json`。全部分析在正式配置锁定和 dev 评估后完成，未用于重调本轮参数。

### 内部调参指标审计补录

用户继续询问调参集 Loss、缺失的评测指标与三分类结构。代码核对确认3 logits、long类型0/1/2标签、直接交叉熵训练、推理Softmax正确。原日志没有调参Accuracy或最佳选轮checkpoint，正式模型已包含tune重训，不能回测后冒充留出表现。新增 `scripts/audit_tuning.py`，复用原fit/tune ID、种子、编码与训练函数重放，观察每轮对称概率并保存最佳模型；八轮Log Loss与原记录逐项完全一致。补算tune Accuracy、Macro-F1、分类别指标及同条件fit_eval/tune_eval，全部进入独立 `artifacts-tuning-audit/`。正式关键产物前后哈希未变，未加载dev；主代理另用标准库从8轮CSV独立复算核验。

最佳第5轮tune Accuracy48.0635%、LogLoss1.021156、Macro-F1 0.472135；第4轮Accuracy48.3681%略高但LogLoss较差，继续按照题目主指标选轮。新容量和学习率方案仅形成 `EXPERIMENTS.md`，尚未执行。做多种子实验必须固定分组ID并单独变化train_seed，不能通过修改同时影响划分的主入口seed来混淆比较。

## 改进实验与用户停止后的报告提交

用户授权改善分类准确率并整理交付，同时要求复制本项目对应的本地 Codex JSONL。主代理分配分类头实验、联合编码候选及会话导出三个并行任务，保留原正式实验。

- 完成18次固定embedding、固定分组的分类头实验：四种结构各3seed，以及阶段一优选结构的两个新增学习率各3seed；保存每轮指标、NLL及Accuracy各自最优checkpoint，三seed均值和标准差。原128结构数值锚点与旧审计逐位一致。
- 联合MiniLM候选改变问题—回答联合输入、384token预算及encoder可训练状态，不能将收益归因于单一因素。GPU2微调全部6层，CPU3项测试和小规模GPU保存重载检查通过，权重更新检查证明确实更新encoder。
- 用户于2026-11-10要求停止训练。主代理只终止确认属于本项目的PID50823并复查无本项目训练进程；保留两轮完整记录和第二轮best/resume。第三轮中途进度不算完整结果。18次head主实验已结束，计划中的额外9次未执行。没有新候选refit、校准或新dev结果。
- 原正式模型的dev Accuracy48.1559%仍是已完成完整验证的结果；新候选第2轮tune Accuracy48.7598%与它属于不同集合，未混作提升结论。
- 按父会话图导出与该项目关联的主/子会话原始JSONL到本地C-TH-02/codex_sessions，副本按完整记录边界截取，清单含来源、快照时刻、字节数、SHA256；这些含完整操作和可能样本文本的文件不公开上传。
- 用户要求将复现、设计和改进整理到REPROT并将代码推送GitHub。新增REPROT.md与results/聚合证据；公开报告只给12例脱敏证据摘要，完整案例原件留本地。补充忽略smoke权重/缓存，36项源码回归测试通过。没有公开上传两个data CSV、模型缓存、原始会话或样本文本。

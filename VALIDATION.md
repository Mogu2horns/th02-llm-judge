# TH02 实施与验收记录

## Linux 服务器全量验收（2026 年 11 月 10 日）

**本次完整训练已完成**，以下结论来自 `artifacts/` 的正式运行，非原 Windows 小规模记录。官方 train/dev 分别为 45,982/5,748 条，八项原始文件 SHA256 全部通过，随机种子为 `20260916`。

| 实验 | dev Log Loss | Accuracy |
|---|---:|---:|
| 类别先验 | 1.097351832 | 0.353340 |
| TF-IDF＋逻辑回归 | 1.050704936 | 0.450939 |
| 校准 Baseline | 1.051444524 | 0.450939 |
| A/B 对称 Baseline | 1.045375669 | 0.465031 |
| 校准对称 Baseline | 1.042299151 | 0.465031 |
| 冻结 MiniLM＋统计特征＋MLP | 1.028195140 | 0.473730 |
| 概率融合 | 1.025564947 | 0.481559 |
| 温度校准融合 | 1.022189513 | 0.481559 |

主指标按 `[1e-15, 1-1e-15]` 裁剪真实类别概率。校准融合的 Log Loss 比原始 Baseline 降低 `0.028515423`（相对 `2.7139%`）。Baseline 的单独温度校准在 dev 上略有退步，结果原样保留。模型选择未使用 dev。

实际配置：Baseline `C=0.1`，MLP 选择 5 轮；融合语义权重 `alpha=0.5`，温度 `T=0.75`。初始拟合/调参/校准/最终重训分别为 36,804/4,596/4,582/41,400 条。所有配置在 dev 预测前写入 `models/selection_locked.json`。

验收证据如下。

- 独立 `.venv`：Python 3.11.0、PyTorch 2.5.1+cu121、Transformers 4.48.3、scikit-learn 1.6.1；完整依赖见 [requirements-lock-linux.txt](requirements-lock-linux.txt)。
- GPU 3 为 RTX 3090，CUDA 12.1 FP16 矩阵运算通过；`pip check` 及 29 项测试通过，见 `logs/cuda-validation.json`、`logs/pip-check.log`、`logs/unit-tests.log`。
- 真实 MiniLM 小规模端到端流程通过，产物在 `artifacts-smoke-server/`，不用于全量结论。
- 独立验收调用未修改的官方 `evaluation/metric.py` 复算八组完整 dev 预测，全部与记录相差不超过 `1e-12`；ID 覆盖、有限性、范围与概率行和均通过，见 [verification.json](artifacts/verification.json)。
- 加载已保存的 Baseline、MLP 与校准参数，重建全 dev 的 Baseline、语义及最终融合预测，最大绝对差均为 0，见 [reload_verification.json](artifacts/reload_verification.json)。
- A/B 交换对称性最大绝对误差为 0；MiniLM 固定 revision 与权重 SHA256 通过官方校验，来源见 `models/all-MiniLM-L6-v2/provenance.json`。
- 成功运行耗时 `748.63` 秒（约 `12.48` 分钟），不含安装下载和首次性能故障尝试。首次尝试已停止；进程级 THP 规避后的 60 秒观测中约 98% CPU 时间用于用户态计算，见 `logs/thp-investigation.json`。

复现与复算入口：

```bash
TH02_GPU=3 bash train_server.sh
.venv/bin/python scripts/verify_training.py --output-dir artifacts --data-dir data
```

完整报告见 [REPORT.md](artifacts/REPORT.md)，逐例证据复核见 [ERROR_ANALYSIS_REVIEWED.md](artifacts/ERROR_ANALYSIS_REVIEWED.md)，AI 记录见 [AI_USAGE.md](AI_USAGE.md)。训练日志为 `logs/full-training.log`；状态为 `artifacts/run_state.json: complete`、`artifacts/metrics.json: smoke=false`。

## 历史 Windows 验收记录

以下保留迁移前的验收与资源阻塞事实；其中的小规模指标和资源不足状态不代表本次服务器正式运行。

## 已执行范围

环境准备、完整数据审计、内部切分、26 项单元测试和两次 GPU 端到端小规模实验均已完成。每次小规模实验使用前 512 条 train、前 128 条 dev，MLP 最多训练 2 轮，随机种子为 `20260916`。

项目 `.venv` 实际使用 Python 3.11.11、PyTorch 2.5.1 与 CUDA 12.1。为解决大文件下载超时，基础运行时采用本机 Conda 缓存，并记录在 `environment-conda-explicit.txt`。pip 依赖版本记录在 `requirements-lock.txt`。

## 实测小规模结果

| 实验 | dev 子集 Log Loss | Accuracy |
|---|---|---|
| 类别先验 | 1.107906401 | 0.375000 |
| Baseline | 1.099693312 | 0.382812 |
| 单独校准 Baseline | 1.120427493 | 0.382812 |
| 对称 Baseline | 1.104824495 | 0.351562 |
| 语义与统计分支 | 1.105122677 | 0.390625 |
| 校准融合 | 1.128874178 | 0.351562 |

内部校准集选择 `α=0`、`T=0.5`。融合在这次子集上的 Log Loss 高于原始 Baseline，差值约为 +0.02918。该结果保留原始训练内部选择，供完整实验继续检验。

完整指标、各类别表现与资源耗时见 [metrics.json](artifacts-smoke-cuda/metrics.json)，报告见 [REPORT.md](artifacts-smoke-cuda/REPORT.md)。12 个案例的逐例证据和四类失败模式见 [ERROR_ANALYSIS_REVIEWED.md](artifacts-smoke-cuda/ERROR_ANALYSIS_REVIEWED.md)。

## 已通过的验收

- 原始包全部 8 项 SHA256 与 MANIFEST 一致。
- 项目独立环境的 CUDA FP16 矩阵运算通过。
- `pip check` 通过。
- 26 项单元测试通过，日志见 [validation-tests.log](artifacts/validation-tests.log)。
- 安装脚本在 Windows PowerShell 中以离线模式重复执行通过，环境锁中的版本均已满足。
- 八个实验的工程 Log Loss 与题目原始评分函数一致。
- 两次小规模运行的八份预测 CSV 逐字节一致，全部指标相同。
- 复用真实嵌入缓存重跑后，预测与指标仍完全一致，日志见 [smoke-cuda-resume.log](artifacts/smoke-cuda-resume.log)。
- 最终融合的 A/B 交换对称性最大误差为 0。

可复现比较记录见 [reproducibility.json](artifacts/reproducibility.json)。编码缓存恢复与模型重载另由单元测试覆盖。

## 完整数据审计

官方 train 为 45,982 条，dev 为 5,748 条。内部拟合、调参、校准、重训分别为 36,804、4,596、4,582、41,400 条，内部规范化内容组互斥。

完整对话的解析后精确跨集重叠为 0，NFKC 与空白规范化后为 1 组。跨集重复问题为 700 组，训练集存在 4 组同内容标签冲突。完整证据见 [data_profile.json](artifacts/data_profile.json) 和 [splits.json](artifacts/splits.json)。

## 全量训练状态

独立环境完成后的全量启动曾记录可用内存为 1.58 GiB，最终复检为 1.24 GiB，低于批准方案的 6 GiB 门槛。入口返回退出码 2，状态为 `blocked_resources`。最新记录见 [run_state.json](artifacts/run_state.json) 和 [full-run-latest.log](artifacts/full-run-latest.log)。

完整 train/dev 模型拟合、正式 dev 指标和对应的逐例错误分析仍待执行。资源达到门槛后，在项目目录重跑既定命令即可。

```powershell
.\.venv\Scripts\python.exe -m th02.run --data-dir data --output-dir artifacts --seed 20260916
```

AI 使用、关键对话、修改和核验方式见 [AI_USAGE.md](AI_USAGE.md)。

## 内部调参指标审计补录

原选轮过程已用原划分、种子和缓存重放，八轮 tune Log Loss 与原记录逐项完全一致；真正留出的 tune Accuracy/Macro-F1 已补齐。最佳第 5 轮 tune Log Loss=1.021156、Accuracy=48.0635%、Macro-F1=0.472135。同条件 fit_eval Accuracy=51.2770%。这不是对已并入 tune 重训后的正式模型做回测。详见 [调参审计报告](artifacts-tuning-audit/REPORT.md) 与 [独立验证](artifacts-tuning-audit/verification.json)。后续实验设计见 [EXPERIMENTS.md](EXPERIMENTS.md)，容量扫描尚未执行。

## 改进实验停止状态（2026-11-10）

18次分类头主实验已完成；联合编码微调只完成2轮，第三轮按用户要求中止。新方案未完成最终重训/校准/dev评估，不能声称已有新dev提升。实际设计、命令、完整数值及12例脱敏分析见 [REPROT.md](REPROT.md)，可公开聚合证据见results/。新增源码共36项回归测试通过（8.170秒）；本地日志为logs/final-tests-stopped-study.log。原CSV与官方评分文件仍保持MANIFEST校验一致。

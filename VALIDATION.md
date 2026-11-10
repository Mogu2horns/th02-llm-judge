# TH02 实施与验收记录

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

"""Generate reproducible experiment summaries and evidence-first error candidates."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import TEXT_FIELDS, targets
from .evaluation import LABELS, select_error_indices, validate_probabilities


DISPLAY_LABELS = ("A 胜", "B 胜", "平局")
MODEL_NAMES = {
    "prior": "训练集类别先验",
    "baseline": "原始 Baseline",
    "baseline_calibrated": "校准 Baseline",
    "symmetric_baseline": "对称 Baseline",
    "symmetric_baseline_calibrated": "校准对称 Baseline",
    "semantic": "语义与统计分支",
    "fusion": "概率融合",
    "fusion_calibrated": "校准概率融合",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _json_block(value) -> list[str]:
    return ["```json", json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), "```", ""]


def _snippet(text: str, width: int = 400) -> str:
    if len(text) <= width * 2:
        return text
    return text[:width] + "\n……中间内容见 error_cases.json……\n" + text[-width:]


def _indented_text(text: str) -> list[str]:
    lines = text.splitlines()
    return ["    " + line for line in (lines or ["（空文本）"])] + [""]


def _observed_groups(sample, label: int, truncation: dict, baseline, semantic) -> list[str]:
    groups = []
    if label == 2:
        groups.append("数据集标签为平局")
    if len(sample.prompt) > 1:
        groups.append("多轮上下文")
    raw_fields = ["\n".join(getattr(sample, field)) for field in TEXT_FIELDS]
    if any(truncation.values()) or sum(len(text) for text in raw_fields) > 8000:
        groups.append("长文本与截断检查")
    if any(sample.null_counts) or any(not text.strip() for text in raw_fields[1:]):
        groups.append("空回答或缺失轮次")
    if any("```" in text for text in raw_fields):
        groups.append("含代码围栏")
    if any(ord(character) > 127 for character in raw_fields[0]):
        groups.append("问题含非 ASCII 字符")
    if semantic is not None and np.argmax(baseline) != np.argmax(semantic):
        groups.append("词面与语义分支预测分歧")
    return groups or ["一般偏好区分"]


def _review_questions(groups: list[str]) -> list[str]:
    questions = []
    if "长文本与截断检查" in groups or "多轮上下文" in groups:
        questions.append("核查决定偏好的句子是否位于截断区域，并结合前序轮次检查上下文依赖。")
    if "数据集标签为平局" in groups:
        questions.append("比较两侧有效信息、事实性和指令执行情况，说明平局标签能否由文本支持。")
    if "空回答或缺失轮次" in groups:
        questions.append("核查 null 与空文本所在轮次，并检查另一侧回答是否有效满足问题。")
    if "含代码围栏" in groups:
        questions.append("检查代码的运行条件、边界输入和题目约束，记录可验证的正确性证据。")
    if "问题含非 ASCII 字符" in groups:
        questions.append("按原文语境核查答案内容，并检查词汇覆盖和语义编码截断情况。")
    if "词面与语义分支预测分歧" in groups:
        questions.append("比较两个分支对真实类别分配的概率，核查质量差异是否被词面或语义表示遗漏。")
    if not questions:
        questions.append("逐条检查事实性、推理步骤、指令遵从和回答完整性，并引用支持判断的原文。")
    return questions


def _make_error_cases(dev, predictions: dict, embedding_metadata: dict) -> list[dict]:
    labels = targets(dev)
    baseline = validate_probabilities(predictions["baseline"])
    improved = validate_probabilities(predictions["fusion_calibrated"])
    semantic = predictions.get("semantic")
    if semantic is not None:
        semantic = validate_probabilities(semantic)
    indices = select_error_indices(dev, labels, baseline, improved, count=12)
    dev_embedding_metadata = embedding_metadata.get("dev", {})
    encoder_ids = dev_embedding_metadata.get("truncated_ids", {})
    encoder_ids = {field: set(encoder_ids.get(field, [])) for field in TEXT_FIELDS}
    cases = []
    for index in indices:
        sample = dev[index]
        lexical_truncation = {field: len(sample.texts[position]) > 16000
                              for position, field in enumerate(TEXT_FIELDS)}
        semantic_truncation = {field: sample.id in encoder_ids[field] for field in TEXT_FIELDS}
        any_truncation = {field: lexical_truncation[field] or semantic_truncation[field]
                          for field in TEXT_FIELDS}
        groups = _observed_groups(
            sample, int(labels[index]), any_truncation,
            baseline[index], semantic[index] if semantic is not None else None,
        )
        predicted_class = int(np.argmax(improved[index]))
        baseline_class = int(np.argmax(baseline[index]))
        evidence = [
            f"数据集标签为{DISPLAY_LABELS[int(labels[index])]}，原始 Baseline 预测为{DISPLAY_LABELS[baseline_class]}，"
            f"校准融合预测为{DISPLAY_LABELS[predicted_class]}。",
            f"原始 Baseline 对真实类别的概率为 {baseline[index, labels[index]]:.6f}，"
            f"校准融合对应概率为 {improved[index, labels[index]]:.6f}。",
        ]
        if len(sample.prompt) > 1:
            evidence.append(f"输入包含 {len(sample.prompt)} 轮对话，轮次来自原始 JSON 数组。")
        if any(semantic_truncation.values()) or any(lexical_truncation.values()):
            evidence.append("字段截断由编码缓存元数据或 16,000 字符阈值直接确认。")
        raw_fields = {field: getattr(sample, field) for field in TEXT_FIELDS}
        case = {
            "id": sample.id,
            "dev_index": int(index),
            "review_status": "candidate_requires_review",
            "label": LABELS[int(labels[index])],
            "label_index": int(labels[index]),
            "baseline_probabilities": baseline[index].tolist(),
            "improved_probabilities": improved[index].tolist(),
            "semantic_probabilities": semantic[index].tolist() if semantic is not None else None,
            "baseline_prediction": LABELS[baseline_class],
            "improved_prediction": LABELS[predicted_class],
            "baseline_wrong": bool(baseline_class != labels[index]),
            "improved_wrong": bool(predicted_class != labels[index]),
            "improved_max_probability": float(improved[index].max()),
            "observed_groups": groups,
            "observable_evidence": evidence,
            "review_questions": _review_questions(groups),
            "raw_fields": raw_fields,
            "null_counts": dict(zip(TEXT_FIELDS, sample.null_counts)),
            "turn_counts": {field: len(turns) for field, turns in raw_fields.items()},
            "character_counts": {field: len("\n".join(turns)) for field, turns in raw_fields.items()},
            "tfidf_truncated": lexical_truncation,
            "semantic_truncated": semantic_truncation,
        }
        cases.append(case)
    return cases


def _error_markdown(cases: list[dict], smoke: bool) -> str:
    lines = ["# 错误与高不确定样本候选", ""]
    if smoke:
        lines += ["本次使用小规模子集进行流程验收，案例和指标适用于该子集。", ""]
    lines += [
        "以下案例由固定规则自动筛选，当前状态为待逐例复核。观测分组描述输入和预测事实，"
        "失败原因需结合完整文本及可验证证据补充。", "",
        "完整多轮文本保存在同目录 error_cases.json，Markdown 展示每字段首尾摘录。"
        "null 已按训练预处理转为空字符串，缺失数量单独保留。", "",
        f"本次选取 {len(cases)} 个规范化内容互异的错误或高不确定样本。", "",
        "## 观测分组", "", "| 候选分组 | 案例数 |", "|---|---|",
    ]
    group_counts = Counter(group for case in cases for group in case["observed_groups"])
    lines.extend(f"| {group} | {count} |" for group, count in group_counts.most_common())
    lines += ["", "分组允许重叠。最终报告应从逐例证据归纳至少三类失败模式。", ""]
    for number, case in enumerate(cases, start=1):
        lines += [f"## 案例 {number}　样本 {case['id']}", "",
                  f"数据集标签为 **{DISPLAY_LABELS[case['label_index']]}**，"
                  "概率顺序为 A 胜、B 胜、平局。", "",
                  "| 模型 | A 胜 | B 胜 | 平局 |", "|---|---|---|---|"]
        for name, key in (("原始 Baseline", "baseline_probabilities"),
                          ("校准融合", "improved_probabilities")):
            values = " | ".join(f"{value:.6f}" for value in case[key])
            lines.append(f"| {name} | {values} |")
        lines += ["", "### 可直接核对的证据", ""]
        lines.extend(f"- {evidence}" for evidence in case["observable_evidence"])
        lines += ["", f"观测分组为{'、'.join(case['observed_groups'])}。", "",
                  "### 文本与处理情况", "",
                  "| 字段 | 字符数 | 轮数 | null 数 | TF-IDF 截断 | MiniLM 截断 |",
                  "|---|---|---|---|---|---|"]
        for field in TEXT_FIELDS:
            lines.append(
                f"| {field} | {case['character_counts'][field]} | {case['turn_counts'][field]} | "
                f"{case['null_counts'][field]} | {'是' if case['tfidf_truncated'][field] else '否'} | "
                f"{'是' if case['semantic_truncated'][field] else '否'} |"
            )
        lines.append("")
        for field, title in zip(TEXT_FIELDS, ("问题", "回答 A", "回答 B")):
            lines += [f"#### {title}", ""]
            text = "\n".join(f"[TURN {turn + 1}]\n{value}"
                             for turn, value in enumerate(case["raw_fields"][field]))
            lines += _indented_text(_snippet(text))
        lines += ["### 待补充的核验", ""]
        lines.extend(f"- {question}" for question in case["review_questions"])
        lines += ["", "逐例结论待复核后填写。", ""]
    return "\n".join(lines)


def _report_markdown(train, dev, profile: dict, experiment: dict, embedding_metadata: dict) -> str:
    smoke = bool(experiment.get("smoke", False))
    lines = ["# TH02 实验报告", "", "## 实验概要", ""]
    if smoke:
        lines += ["**本次为小规模流程验收。下列指标基于所选子集，完整数据实验需单独运行。**", ""]
    lines += [
        f"本次使用 {len(train):,} 条训练样本、{len(dev):,} 条验证样本，随机种子为 "
        f"`{experiment.get('seed')}`。运行状态记录为 `{experiment.get('run_status', 'unknown')}`。", "",
        "方案以 TF-IDF 与三分类逻辑回归建立词面 Baseline，使用冻结 MiniLM 嵌入、"
        "18 维统计特征和小型 MLP 建立语义分支。两分支执行 A/B 对称预测，随后进行概率融合与温度校准。", "",
        "改进假设为语义表示能够补充问题与回答关系，统计特征能够补充缺失、长度与格式信息，"
        "对称处理能够缓解位置偏差。实际增益以本报告实测结果为准。", "",
        "## 数据理解", "",
        "JSON 数组按原始轮次解析，null 转为空文本并保留缺失数量。"
        "文本使用轮次标记序列化，TF-IDF 每字段最多保留 16,000 字符并平均保留头尾，"
        "MiniLM 每字段最多 256 token，内容 token 的头尾预算平均分配。", "",
        "| 数据集 | 样本数 | 多轮样本 | 规范化重复额外行 | 相同回答样本 |",
        "|---|---|---|---|---|",
    ]
    for split in ("train", "dev"):
        current = profile.get(split, {})
        lines.append(
            f"| {split} | {current.get('rows', 0)} | {current.get('multi_turn_samples', 0)} | "
            f"{current.get('normalized_duplicate_extra_rows', 0)} | {current.get('identical_responses', 0)} |"
        )
    cross_split = profile.get("cross_split", {})
    lines += ["", f"train/dev 规范化完整对话重叠组数为 "
              f"{cross_split.get('normalized_dialogue_overlap_groups', 0)}，"
              f"重复问题重叠组数为 {cross_split.get('normalized_prompt_overlap_groups', 0)}。", "",
              "完整空值、长度分位数、冲突标签、重复与截断信息见 data_profile.json 和编码缓存元数据。", "",
              "## 验证设计", "",
              "官方 train/dev 保持给定边界。train 内部按规范化完整对话哈希分组，"
              "使用固定分层分组十折，其中八折拟合、一折调参、一折校准。"
              "C 和 MLP 轮数选定后，使用前九折重新拟合。融合权重与温度由校准折确定。", "",
              "规范化使用 Unicode NFKC、合并空白、保留大小写和轮次边界，null 按空字符串处理。"
              "语义近似对话与同问题不同回答仍可能跨集合出现，验证分数可能因此偏乐观。", "",
              "内部分配数量如下。", ""]
    lines += _json_block(experiment.get("splits", {}))
    lines += ["## 验证结果", "", "主指标为三分类 Log Loss，越低越好。"
              "Accuracy 和按类别表现用于辅助分析。", "",
              "| 模型 | Log Loss | Accuracy |", "|---|---|---|"]
    results = experiment.get("results", {})
    for name, values in results.items():
        lines.append(
            f"| {MODEL_NAMES.get(name, name)} | {values['log_loss']:.8f} | {values['accuracy']:.6f} |"
        )
    if "baseline" in results and "fusion_calibrated" in results:
        difference = results["fusion_calibrated"]["log_loss"] - results["baseline"]["log_loss"]
        lines += ["", f"校准融合相对原始 Baseline 的 Log Loss 差值为 {difference:+.8f}。"]
    lines += ["", "预测文件采用固定三列概率顺序，官方 evaluation/metric.py 用于指标复算。"
              "各类别 precision、recall、support 和完整实验配置见 metrics.json。", "",
              "## 训练与校准配置", "", "### Baseline", ""]
    lines += _json_block(experiment.get("baseline_training", {}))
    lines += ["### 语义分支", ""]
    lines += _json_block(experiment.get("semantic_training", {}))
    lines += ["### 融合与温度", ""]
    calibration = {key: value for key, value in experiment.get("calibration", {}).items() if key != "grid"}
    lines += _json_block(calibration)
    lines += ["### 字段截断比例", ""]
    lines += _json_block({split: {
        "tfidf": profile.get(split, {}).get("tfidf_truncation_rates", {}),
        "minilm": metadata.get("truncation_rates", {}),
    } for split, metadata in embedding_metadata.items()})
    lines += ["## 资源与耗时", ""]
    lines += _json_block({"resources": experiment.get("resources", {}), "timings": experiment.get("timings", {})})
    lines += ["## 错误分析与限制", "",
              "ERROR_ANALYSIS.md 与 error_cases.json 提供自动筛选的错误及高不确定候选，"
              "每例包含完整文本引用来源、概率、标签、长度、轮次、缺失和截断证据。"
              "候选需逐例复核后归纳至少三类有证据的失败模式。", "",
              "MiniLM 表示采用通用句向量，模型对事实性、细粒度推理、代码正确性和语言覆盖的表现"
              "需要通过具体案例核验。头尾截断会丢失中间内容，"
              "标注分歧及语义近似对话仍会影响偏好学习和验证解释。", "",
              "## 可复现性与 AI 使用", "",
              "环境、锁定依赖、种子及运行入口见项目 README。"
              "模型、预测、内部切分和参数选择均由代码保存，官方评分函数可复算报告指标。"
              "AI 辅助工具、采纳与修改情况及核验操作记录见项目 AI_USAGE.md。", ""]
    return "\n".join(lines)


def write_reports(
    output_dir, train, dev, profile, experiment, predictions, embedding_metadata,
) -> None:
    """Write the run summary and reviewable error candidates from saved evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _make_error_cases(dev, predictions, embedding_metadata)
    error_payload = {
        "status": "candidate_requires_review",
        "smoke": bool(experiment.get("smoke", False)),
        "probability_order": list(LABELS),
        "selection_policy": "content-distinct errors or improved maximum probability <= 0.5",
        "cases": cases,
    }
    _write_atomic(
        output_dir / "error_cases.json",
        json.dumps(error_payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )
    _write_atomic(output_dir / "ERROR_ANALYSIS.md", _error_markdown(cases, error_payload["smoke"]))
    _write_atomic(
        output_dir / "REPORT.md",
        _report_markdown(train, dev, profile, experiment, embedding_metadata),
    )

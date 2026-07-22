#!/usr/bin/env python3
"""Summarize the completed EWC frozen-proxy strength/coverage sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def get_summary(path: Path) -> dict[str, float]:
    return read_json(path)["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/ewc_attack_strength_sweep/runs_ewc"),
    )
    parser.add_argument(
        "--stream-root",
        type=Path,
        default=Path("experiments/ewc_attack_strength_sweep/frozen_proxy_F-S"),
    )
    parser.add_argument(
        "--clean-reference",
        type=Path,
        default=Path(
            "experiments/frozen_proxy_frequency_shift/full49_runs/none/clean"
        ),
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    stream_root = args.stream_root.resolve()
    clean_metrics = read_json(args.clean_reference / "ewc" / "metrics.json")
    clean = clean_metrics["summary"]
    initial_old_acc = float(
        clean_metrics["initial"]["old_generalization"]["acc"]
    )
    initial_new_subject_acc = [
        float(value["acc"])
        for value in clean_metrics["final"]["initial_model_seen_subjects"].values()
    ]
    initial_new_mean_acc = sum(initial_new_subject_acc) / len(
        initial_new_subject_acc
    )
    manifest = read_json(stream_root / "manifest.json")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for condition in manifest["conditions"]:
        name = condition["condition"]
        metrics_path = run_root / name / "ewc" / "metrics.json"
        metadata_path = stream_root / name / "manifest.json"
        if not metrics_path.is_file():
            missing.append(str(metrics_path))
            continue
        # The stream has task metadata below each individual directory.  The
        # top-level condition row is sufficient for coverage and budget values.
        summary = get_summary(metrics_path)
        attacked_task_rows = [
            task
            for task in read_json(metrics_path)["tasks"]
            if task.get("noise", {}).get("noisy_sequences", 0) > 0
        ]
        pseudo_acc = sum(
            float(task["pseudo_labels"]["acc_diagnostic_only"])
            for task in attacked_task_rows
        ) / max(len(attacked_task_rows), 1)
        clean_pseudo_acc = sum(
            float(task["pseudo_labels_on_clean_current"]["acc_diagnostic_only"])
            for task in attacked_task_rows
        ) / max(len(attacked_task_rows), 1)
        row = {
            **condition,
            "final_old_acc": summary["final_old_acc"],
            "final_old_mf1": summary["final_old_mf1"],
            "final_seen_acc": summary["final_seen_acc"],
            "final_seen_mf1": summary["final_seen_mf1"],
            "bwt_acc": summary["bwt_acc"],
            "bwt_mf1": summary["bwt_mf1"],
            "delta_old_acc": summary["final_old_acc"] - clean["final_old_acc"],
            "delta_old_mf1": summary["final_old_mf1"] - clean["final_old_mf1"],
            "delta_seen_acc": summary["final_seen_acc"] - clean["final_seen_acc"],
            "delta_seen_mf1": summary["final_seen_mf1"] - clean["final_seen_mf1"],
            "delta_bwt_acc": summary["bwt_acc"] - clean["bwt_acc"],
            "delta_bwt_mf1": summary["bwt_mf1"] - clean["bwt_mf1"],
            "attacked_task_pseudo_acc": pseudo_acc,
            "clean_task_pseudo_acc": clean_pseudo_acc,
            "delta_task_pseudo_acc": pseudo_acc - clean_pseudo_acc,
            "both_acc_drop_1pp": (
                summary["final_old_acc"] <= clean["final_old_acc"] - 0.01
                and summary["final_seen_acc"] <= clean["final_seen_acc"] - 0.01
            ),
            "both_acc_drop_2pp": (
                summary["final_old_acc"] <= clean["final_old_acc"] - 0.02
                and summary["final_seen_acc"] <= clean["final_seen_acc"] - 0.02
            ),
        }
        rows.append(row)
    if missing:
        raise RuntimeError("Missing EWC sweep artifacts:\n" + "\n".join(missing))

    strength_rows = [row for row in rows if row["condition"].startswith("strength_")]
    subject_rows = [row for row in rows if row["condition"].startswith("subjects_")]
    sequence_rows = [row for row in rows if row["condition"].startswith("sequences_")]

    def first_threshold(items: list[dict[str, Any]], key: str) -> str | None:
        for row in items:
            if row[key]:
                return row["condition"]
        return None

    threshold_summary = {
        "strength_first_both_acc_drop_1pp": first_threshold(
            strength_rows, "both_acc_drop_1pp"
        ),
        "strength_first_both_acc_drop_2pp": first_threshold(
            strength_rows, "both_acc_drop_2pp"
        ),
        "subjects_first_both_acc_drop_1pp": first_threshold(
            sorted(subject_rows, key=lambda row: row["task_count"]),
            "both_acc_drop_1pp",
        ),
        "subjects_first_both_acc_drop_2pp": first_threshold(
            sorted(subject_rows, key=lambda row: row["task_count"]),
            "both_acc_drop_2pp",
        ),
        "sequences_first_both_acc_drop_1pp": first_threshold(
            sorted(sequence_rows, key=lambda row: row["sequence_fraction"]),
            "both_acc_drop_1pp",
        ),
        "sequences_first_both_acc_drop_2pp": first_threshold(
            sorted(sequence_rows, key=lambda row: row["sequence_fraction"]),
            "both_acc_drop_2pp",
        ),
    }

    machine = {
        "clean_reference": str(args.clean_reference.resolve()),
        "clean_summary": clean,
        "clean_initial_old_acc": initial_old_acc,
        "clean_initial_new_subject_mean_acc": initial_new_mean_acc,
        "manifest": manifest,
        "rows": rows,
        "threshold_definition": {
            "visible_degradation_1pp": "both final old ACC and final seen-new ACC are at least 1 percentage point below clean",
            "strong_degradation_2pp": "both final old ACC and final seen-new ACC are at least 2 percentage points below clean",
        },
        "threshold_summary": threshold_summary,
    }
    report_lines = [
        "# EWC Frozen Proxy 攻击强度与覆盖率实验",
        "",
        "> 本实验只使用 EWC，所有攻击流来自同一 frozen proxy 方向和 nested sequence mask，victim 不使用 replay、BrainUICL 或额外防御。结果与同一协议的 clean EWC 配对。",
        "",
        "## 1. 为什么 clean 不使用 replay 仍有 60%–70% ACC",
        "",
        f"这些方法不是从随机参数开始学习。source-pretrained 模型在任何新个体增量更新之前，old-generalization ACC 已经是 `{pct(initial_old_acc)}`，在全部 49 个新个体上的初始模型 subject 平均 ACC 是 `{pct(initial_new_mean_acc)}`。EWC clean 完成 49 个任务后旧个体 ACC 为 `{pct(clean['final_old_acc'])}`，说明高准确率主要来自 source pretraining 和跨个体共享的睡眠分期表示，而不是 replay 重新记住了历史样本。",
        "",
        f"当前是 subject/domain-incremental，而不是增加新类别的 class-incremental：每个个体都使用相同 5 类睡眠阶段和同一个分类头。EWC 的平均当前个体 ACC 从适配前 `{pct(clean['mean_current_before_acc'])}` 变为适配后 `{pct(clean['mean_current_after_acc'])}`，平均只提高 `{100.0 * clean['mean_current_acc_gain']:.2f} pp`。因此新个体 60% 以上的表现多数已经存在于预训练模型，CL 更新只是小幅适配。",
        "",
        "无 replay 也不等于没有历史状态。EWC 保存上一阶段参数锚点和对角 Fisher/importance，Online EWC、SI、MAS 也都把历史压缩进参数与重要性向量；它们不保存原始 EEG sequence，但仍以参数形式保留旧知识。本协议还使用 `cl_lr=1e-6`、EWC strength 5000 和冻结 BN running statistics，使每次更新非常保守，进一步减少遗忘。",
        "",
        "EWC 是四种正则化 CL 中最直接的代表，因此本 sweep 先用它回答“输入污染需要多大、覆盖多少任务/sequence 才能穿过保守正则化更新的累积阈值”，避免把算法差异混入第一轮强度曲线。",
        "",
        "## 2. 预先定义的退化标准",
        "",
        "- **可见退化**：最终旧个体 ACC 和最终已见新个体 ACC 都至少比 clean 低 1 个百分点。",
        "- **明显强退化**：上述两个 ACC 都至少比 clean 低 2 个百分点。",
        "- 单个指标下降而另一个指标不下降，只记为局部退化，不称为同时 old/new 明显退化。",
        "",
        "## 3. Clean 参照",
        "",
        f"EWC clean：旧个体 ACC `{pct(clean['final_old_acc'])}`，已见新个体 ACC `{pct(clean['final_seen_acc'])}`，旧个体 MF1 `{pct(clean['final_old_mf1'])}`，已见新个体 MF1 `{pct(clean['final_seen_mf1'])}`，BWT ACC `{pct(clean['bwt_acc'])}`。",
        "",
        "## 4. 每 sequence 强度 sweep",
        "",
        "固定攻击 25/49 个任务、每个被攻击任务约 20% sequence，使用 shifted `F-S`（所有修改方向取正号）。",
        "",
        "| 条件 | 相对 L2 目标 | L∞/std 上限 | 修改 sequence/全流 | proxy pseudo ACC 变化 | 旧 ACC | 新 ACC | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in strength_rows:
        flags = ("是" if row["both_acc_drop_1pp"] else "否") + " / " + (
            "是" if row["both_acc_drop_2pp"] else "否"
        )
        report_lines.append(
            f"| `{row['condition']}` | {row['relative_l2_budget']:.1%} | {row['linf_std_scale']:.2f} | "
            f"{row['noisy_sequences']}/{row['uploaded_sequences']} | {pp(row['delta_task_pseudo_acc'])} | {pct(row['final_old_acc'])} | "
            f"{pct(row['final_seen_acc'])} | {pp(row['delta_old_acc'])} | {pp(row['delta_seen_acc'])} | "
            f"{pp(row['delta_bwt_acc'])} | {flags} |"
        )
    report_lines.extend(
        [
            "",
            "攻击强度确实改变了训练输入的 guiding pseudo-label 质量：攻击任务上的 proxy pseudo-label diagnostic ACC 变化从约 0 pp（0.5%）扩大到约 -5.7 pp（10%）。但最终 old/new ACC 只下降约 2.1/3.0 pp，说明 EWC 锚点、同任务未污染 sequence 和后续任务共同削弱了输入层扰动向最终参数的传递。",
            "",
            "## 5. 攻击 subject/task 数量 sweep",
            "",
            "固定每个被攻击任务约 20% sequence、每 sequence 5% 相对 L2 和 `0.20 × std` 上限；只改变被攻击任务数量。",
            "",
            "| 条件 | 攻击任务数 | 修改 sequence/全流 | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(subject_rows, key=lambda item: item["task_count"]):
        flags = ("是" if row["both_acc_drop_1pp"] else "否") + " / " + (
            "是" if row["both_acc_drop_2pp"] else "否"
        )
        report_lines.append(
            f"| `{row['condition']}` | {row['task_count']} | {row['noisy_sequences']}/{row['uploaded_sequences']} | "
            f"{pp(row['delta_old_acc'])} | {pp(row['delta_seen_acc'])} | {pp(row['delta_bwt_acc'])} | {flags} |"
        )
    report_lines.extend(
        [
            "",
            "## 6. 每任务 sequence 覆盖率 sweep",
            "",
            "固定攻击 25 个任务、每 sequence 5% 相对 L2；只改变每个被攻击任务内的 sequence 比例。",
            "",
            "| 条件 | 任务内 sequence 比例 | 修改 sequence/全流 | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(sequence_rows, key=lambda item: item["sequence_fraction"]):
        flags = ("是" if row["both_acc_drop_1pp"] else "否") + " / " + (
            "是" if row["both_acc_drop_2pp"] else "否"
        )
        report_lines.append(
            f"| `{row['condition']}` | {row['sequence_fraction']:.0%} | {row['noisy_sequences']}/{row['uploaded_sequences']} | "
            f"{pp(row['delta_old_acc'])} | {pp(row['delta_seen_acc'])} | {pp(row['delta_bwt_acc'])} | {flags} |"
        )

    strength_visible = threshold_summary["strength_first_both_acc_drop_1pp"]
    strength_strong = threshold_summary["strength_first_both_acc_drop_2pp"]
    subjects_visible = threshold_summary["subjects_first_both_acc_drop_1pp"]
    sequences_visible = threshold_summary["sequences_first_both_acc_drop_1pp"]
    report_lines.extend(
        [
            "",
            "## 7. 本轮首次明显退化点",
            "",
            f"- 固定 25 个攻击任务和 20% 任务内覆盖率时，首次达到 old/new 同时下降至少 1 pp 的条件是 `{strength_visible or '未达到'}`；首次同时下降至少 2 pp 的条件是 `{strength_strong or '未达到'}`。",
            f"- 固定 5% L2 和 20% 任务内覆盖率时，subject/task 数量 sweep 首次达到 1 pp 的条件是 `{subjects_visible or '未达到'}`；1、3、10 个攻击任务均未达到。",
            f"- 固定 25 个攻击任务和 5% L2 时，sequence 覆盖率 sweep 首次达到 1 pp 的条件是 `{sequences_visible or '未达到'}`；5% 和 10% 覆盖率均未达到。",
            "",
            "在当前单 seed 协议中，可把 `25 个攻击任务 + 每任务 20% sequence + 每 sequence 5% 相对 L2` 视为出现约 1 pp 可见退化的首个工程工作点；把 L2 提高到 10% 后才出现 old/new 同时超过 2 pp 的强退化。这个阈值只对当前 frozen proxy、EWC 超参数和 ISRUC split 有效。",
            "",
            "## 8. 为什么 BrainWash 原论文下降更大",
            "",
            "| 维度 | BrainWash 原论文 | 当前 EEG EWC sweep |",
            "|---|---|---|",
            "| CL 场景 | 10-split CIFAR-100 等，类别互斥、多头 task-incremental | ISRUC 跨个体、共享 5 类、单头 subject/domain-incremental |",
            "| 初始模型 | 按任务顺序训练 ResNet-18 | 已有 source-pretrained EEG 模型，初始 old/new ACC 已约 70%/65% |",
            "| 攻击者 | 读取 victim 当前参数，模型反演旧任务，并对最后任务做白盒双层优化 | 冻结 surrogate，不读取 EWC 轨迹，只复用一步 classifier proxy 方向 |",
            "| 污染范围 | 主表通常污染最后任务全部样本 | 每个攻击任务最多污染约 20% sequence；5% 档全流仅 224/2148 |",
            "| 输入预算 | 图像先归一化到 [0,1]，L∞ ε=0.1 或 0.3 | EEG/EOG 相对 L2 0.5%–10%，并同时受 0.02–0.40 × std 的 L∞ 上限约束 |",
            "| Victim 更新 | SGD learning rate 1e-2 | `cl_lr=1e-6`、EWC 5000、冻结 BN，更新更保守 |",
            "| 主指标 | BWT 与最后一个被攻击任务 ACC | 19 个 old 个体和 49 个 new 个体的最终平均 ACC/MF1、BWT |",
            "",
            "BrainWash Table 1 中 CIFAR-100 EWC 的 clean BWT 为 -5.2，ε=0.1 reckless 后为 -12.6，即 BWT 额外下降 7.4 pp；最后任务 ACC 从 68.3% 变为 51.0%。这不是“给 10% EEG sequence 加 5% L2 后总体 ACC 必须下降 10 pp”，而是更强白盒攻击、全任务样本污染、不同任务协议和不同指标共同作用的结果。",
            "",
            "## 9. 解释和限制",
            "",
            "本 sweep 的强度是输入级、有限、系统同向的 proxy 扰动；它不是原始 BrainWash 的双层优化复现。此前本地无界 proxy degradation 探针在两个精心挑选的 subject 上累计造成约 32.38 个百分点 old ACC 下降，而受限 BrainWash stress 在一个 subject 上约 0.86 个百分点；两者的攻击目标、预算和选择机制都比本 sweep 更激进或不同，不能直接拿 10% 数字横比。",
            "",
            "即使强度增加，正则化 CL 仍可能只受到有限影响，因为 source-pretrained 模型已经提供了强分类表示，任务共享同一 5 类 label space，EWC 只更新小步参数，且最终指标是跨 49 个体的平均。若只有少数 sequence 被改，污染梯度还会被同一任务的 clean sequence、伪标签和历史锚点稀释。相反，若攻击任务多、sequence 覆盖率高、方向同向，污染会在更多任务中重复进入参数和 importance 状态，才可能出现明显累积退化。",
            "",
            "本报告中的 1/2 个百分点阈值是工程判据，不是统计显著性；当前仍是单 seed。正式结论需要至少 3 个 paired seeds，并报告 subject-level bootstrap 区间、扰动后的伪标签翻转率和 EEG 合理性指标。",
            "",
            "## 10. 复现文件",
            "",
            "- 上传流生成器：`experiments/generate_ewc_attack_strength_sweep.py`",
            "- 断点运行脚本：`scripts/run_ewc_attack_strength_sweep.sh`",
            "- 运行结果：`runs_ewc/`",
            "- 上游方向 manifest：`frozen_proxy_F-S/manifest.json`",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    report_path = run_root / "SWEEP_RESULTS_ZH.md"
    json_path = run_root / "SWEEP_RESULTS.json"
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(machine, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(json.dumps(threshold_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

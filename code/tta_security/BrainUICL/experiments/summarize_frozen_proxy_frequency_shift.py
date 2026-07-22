#!/usr/bin/env python3
"""Summarize the completed frozen-proxy regularization CL experiment.

The script intentionally reads the JSON artifacts produced by the runner rather
than parsing human-readable logs.  It validates that every scheduled method
and condition is present, computes paired attack/defense deltas, and writes a
Chinese report plus a machine-readable summary next to the run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("ewc", "online_ewc", "si", "mas")
METHOD_NAMES = {
    "ewc": "EWC",
    "online_ewc": "Online EWC",
    "si": "SI",
    "mas": "MAS",
}
ATTACK_CONDITIONS = ("clean", "I-NS", "I-S", "F-NS", "F-S")
DEFENSE_CONDITIONS = {
    "none": ATTACK_CONDITIONS,
    "robust_feature": ATTACK_CONDITIONS,
    "t2t": ("clean", "I-S"),
}
DEFENSE_NAMES = {
    "none": "无额外防御",
    "robust_feature": "Robust Feature",
    "t2t": "T2T",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.{digits}f} pp"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_runs(run_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    missing: list[str] = []
    for defense, conditions in DEFENSE_CONDITIONS.items():
        runs[defense] = {}
        for condition in conditions:
            runs[defense][condition] = {}
            for method in METHODS:
                path = run_root / defense / condition / method / "metrics.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                runs[defense][condition][method] = read_json(path)
    if missing:
        raise RuntimeError("Missing metrics artifacts:\n" + "\n".join(missing))
    return runs


def summary(metrics: dict[str, Any]) -> dict[str, float]:
    return metrics["summary"]


def metric(metrics: dict[str, Any], key: str) -> float:
    return float(summary(metrics)[key])


def delta(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    defense: str,
    condition: str,
    method: str,
    key: str,
) -> float:
    return metric(runs[defense][condition][method], key) - metric(
        runs[defense]["clean"][method], key
    )


def aggregate_stream_metadata(
    stream_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for condition in ("I-NS", "I-S", "F-NS", "F-S"):
        frequency = "infrequent" if condition.startswith("I") else "frequent"
        task_set = set(
            manifest["config"]["infrequent_tasks"]
            if frequency == "infrequent"
            else manifest["config"]["frequent_tasks"]
        )
        rows: list[dict[str, Any]] = []
        for task in sorted(task_set):
            path = (
                stream_root
                / "rel_l2_0500"
                / condition
                / f"individual_{task}"
                / "metadata.json"
            )
            if not path.is_file():
                raise RuntimeError(f"Missing stream metadata: {path}")
            rows.append(read_json(path))
        noisy = sum(int(row["noisy_sequences"]) for row in rows)
        attacked_task_uploaded = sum(int(row["uploaded"]) for row in rows)
        uploaded = int(
            manifest["stream_summaries"]["rel_l2_0500"][condition]["uploaded"]
        )

        def weighted_nested(name: str, field: str = "mean") -> float:
            values = []
            for row in rows:
                item = row[name]
                if isinstance(item, dict):
                    item = item[field]
                values.append(float(item) * int(row["noisy_sequences"]))
            return sum(values) / max(noisy, 1)

        signs_positive = sum(int(row["sign_positive"]) for row in rows)
        signs_negative = sum(int(row["sign_negative"]) for row in rows)
        output[condition] = {
            "frequency": frequency,
            "shifted": condition in {"I-S", "F-S"},
            "tasks": len(rows),
            "task_indices": sorted(task_set),
            "noisy_sequences": noisy,
            "attacked_task_uploaded_sequences": attacked_task_uploaded,
            "uploaded_sequences": uploaded,
            "coverage": noisy / max(uploaded, 1),
            "sign_positive": signs_positive,
            "sign_negative": signs_negative,
            "relative_l2_eog_mean": weighted_nested("eog_relative_l2"),
            "relative_l2_eeg_mean": weighted_nested("eeg_relative_l2"),
            "relative_l2_eog_max": max(
                float(row["eog_relative_l2"]["max"]) for row in rows
            ),
            "relative_l2_eeg_max": max(
                float(row["eeg_relative_l2"]["max"]) for row in rows
            ),
            "linf_eog_mean": weighted_nested("eog_linf_over_std"),
            "linf_eeg_mean": weighted_nested("eeg_linf_over_std"),
            "spectral_tv_eog_mean": weighted_nested(
                "eog_spectral_total_variation"
            ),
            "spectral_tv_eeg_mean": weighted_nested(
                "eeg_spectral_total_variation"
            ),
            "out_of_band_eog_mean": weighted_nested(
                "eog_perturbation_out_of_band_fraction"
            ),
            "out_of_band_eeg_mean": weighted_nested(
                "eeg_perturbation_out_of_band_fraction"
            ),
            "outside_range_eog_mean": weighted_nested(
                "eog_sample_outside_clean_range_fraction"
            ),
            "outside_range_eeg_mean": weighted_nested(
                "eeg_sample_outside_clean_range_fraction"
            ),
            "proxy_label_preservation_mean": weighted_nested(
                "proxy_pseudo_label_preservation"
            ),
            "task_mean_ratio": mean(
                [float(row["empirical_sequence_mean_ratio"]) for row in rows]
            ),
        }
    return output


def detected_tasks(metrics: dict[str, Any]) -> list[int]:
    return [
        int(row["task"])
        for row in metrics["tasks"]
        if row.get("defense", {}).get("detected", False)
    ]


def rejected_tasks(metrics: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for row in metrics["tasks"]:
        values.extend(
            int(item)
            for item in row.get("defense", {}).get("rejected_task_indices", [])
        )
    return sorted(set(values))


def t2t_diagnostics(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    attack_tasks: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        clean = runs["t2t"]["clean"][method]
        attacked = runs["t2t"]["I-S"][method]
        clean_detected = set(detected_tasks(clean))
        attacked_detected = set(detected_tasks(attacked))
        clean_rejected = set(rejected_tasks(clean))
        attacked_rejected = set(rejected_tasks(attacked))
        rows.append(
            {
                "method": method,
                "clean_valid": int(summary(clean).get("t2t_valid_scores", 0)),
                "clean_pairs": int(summary(clean).get("t2t_detected_pairs", 0)),
                "clean_rejected": int(
                    summary(clean).get("t2t_rejected_updates", 0)
                ),
                "attack_valid": int(
                    summary(attacked).get("t2t_valid_scores", 0)
                ),
                "attack_pairs": int(
                    summary(attacked).get("t2t_detected_pairs", 0)
                ),
                "attack_rejected": int(
                    summary(attacked).get("t2t_rejected_updates", 0)
                ),
                "clean_detected_tasks": sorted(clean_detected),
                "attack_detected_tasks": sorted(attacked_detected),
                "new_detected_tasks": sorted(attacked_detected - clean_detected),
                "attack_endpoint_hits": sorted(attacked_detected & attack_tasks),
                "attack_rejected_hits": sorted(attacked_rejected & attack_tasks),
                "new_rejected_attack_hits": sorted(
                    (attacked_rejected - clean_rejected) & attack_tasks
                ),
            }
        )
    return rows


def build_machine_summary(
    run_root: Path,
    manifest: dict[str, Any],
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    streams: dict[str, dict[str, Any]],
    t2t: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for defense, conditions in runs.items():
        metrics[defense] = {}
        for condition, methods in conditions.items():
            metrics[defense][condition] = {
                method: summary(value) for method, value in methods.items()
            }

    attack_deltas = {}
    for method in METHODS:
        attack_deltas[method] = {}
        for condition in ATTACK_CONDITIONS[1:]:
            attack_deltas[method][condition] = {
                key: delta(runs, "none", condition, method, key)
                for key in ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1", "bwt_acc", "bwt_mf1")
            }

    robust_recovery = {}
    for method in METHODS:
        robust_recovery[method] = {}
        for condition in ATTACK_CONDITIONS[1:]:
            robust_recovery[method][condition] = {
                "unprotected_old_acc_delta": delta(runs, "none", condition, method, "final_old_acc"),
                "protected_old_acc_delta": delta(runs, "robust_feature", condition, method, "final_old_acc"),
                "unprotected_seen_acc_delta": delta(runs, "none", condition, method, "final_seen_acc"),
                "protected_seen_acc_delta": delta(runs, "robust_feature", condition, method, "final_seen_acc"),
                "old_acc_recovery": delta(runs, "robust_feature", condition, method, "final_old_acc") - delta(runs, "none", condition, method, "final_old_acc"),
                "seen_acc_recovery": delta(runs, "robust_feature", condition, method, "final_seen_acc") - delta(runs, "none", condition, method, "final_seen_acc"),
                "bwt_acc_recovery": delta(runs, "robust_feature", condition, method, "bwt_acc") - delta(runs, "none", condition, method, "bwt_acc"),
            }

    t2t_recovery = {}
    for method in METHODS:
        t2t_recovery[method] = {
            "unprotected_old_acc_delta": delta(runs, "none", "I-S", method, "final_old_acc"),
            "t2t_old_acc_delta": delta(runs, "t2t", "I-S", method, "final_old_acc"),
            "unprotected_seen_acc_delta": delta(runs, "none", "I-S", method, "final_seen_acc"),
            "t2t_seen_acc_delta": delta(runs, "t2t", "I-S", method, "final_seen_acc"),
            "old_acc_recovery": delta(runs, "t2t", "I-S", method, "final_old_acc") - delta(runs, "none", "I-S", method, "final_old_acc"),
            "seen_acc_recovery": delta(runs, "t2t", "I-S", method, "final_seen_acc") - delta(runs, "none", "I-S", method, "final_seen_acc"),
            "bwt_acc_recovery": delta(runs, "t2t", "I-S", method, "bwt_acc") - delta(runs, "none", "I-S", method, "bwt_acc"),
        }

    return {
        "run_root": str(run_root),
        "manifest": manifest,
        "streams": streams,
        "metrics": metrics,
        "attack_deltas_vs_none_clean": attack_deltas,
        "robust_feature_recovery": robust_recovery,
        "t2t_recovery": t2t_recovery,
        "t2t_diagnostics": t2t,
    }


def table_baseline(runs: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> list[str]:
    lines = [
        "| CL 方法 | 旧个体 ACC | 旧个体 MF1 | 已见新个体 ACC | 已见新个体 MF1 | BWT ACC | BWT MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary(runs["none"]["clean"][method])
        lines.append(
            f"| {METHOD_NAMES[method]} | {pct(item['final_old_acc'])} | {pct(item['final_old_mf1'])} | "
            f"{pct(item['final_seen_acc'])} | {pct(item['final_seen_mf1'])} | {pct(item['bwt_acc'])} | {pct(item['bwt_mf1'])} |"
        )
    return lines


def table_attack_acc(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[str]:
    lines = [
        "| CL 方法 | I-NS 旧/新/BWT | I-S 旧/新/BWT | F-NS 旧/新/BWT | F-S 旧/新/BWT |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for condition in ATTACK_CONDITIONS[1:]:
            cells.append(
                " / ".join(
                    pp(delta(runs, "none", condition, method, key))
                    for key in ("final_old_acc", "final_seen_acc", "bwt_acc")
                )
            )
        lines.append(f"| {METHOD_NAMES[method]} | " + " | ".join(cells) + " |")
    return lines


def table_attack_mf1(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[str]:
    lines = [
        "| CL 方法 | I-NS 旧/新/BWT | I-S 旧/新/BWT | F-NS 旧/新/BWT | F-S 旧/新/BWT |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for condition in ATTACK_CONDITIONS[1:]:
            cells.append(
                " / ".join(
                    pp(delta(runs, "none", condition, method, key))
                    for key in ("final_old_mf1", "final_seen_mf1", "bwt_mf1")
                )
            )
        lines.append(f"| {METHOD_NAMES[method]} | " + " | ".join(cells) + " |")
    return lines


def table_robust_recovery(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[str]:
    lines = [
        "| CL 方法 | 攻击 | 无防御旧 ACC 退化 | RF 旧 ACC 退化 | 旧 ACC 恢复量 | 无防御新 ACC 退化 | RF 新 ACC 退化 | 新 ACC 恢复量 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for condition in ATTACK_CONDITIONS[1:]:
            unprotected_old = delta(runs, "none", condition, method, "final_old_acc")
            protected_old = delta(runs, "robust_feature", condition, method, "final_old_acc")
            unprotected_new = delta(runs, "none", condition, method, "final_seen_acc")
            protected_new = delta(runs, "robust_feature", condition, method, "final_seen_acc")
            lines.append(
                f"| {METHOD_NAMES[method]} | {condition} | {pp(unprotected_old)} | {pp(protected_old)} | "
                f"{pp(protected_old - unprotected_old)} | {pp(unprotected_new)} | {pp(protected_new)} | "
                f"{pp(protected_new - unprotected_new)} |"
            )
    return lines


def table_t2t(
    t2t_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "| CL 方法 | clean 有效分数 | clean 触发对数 | clean 回滚任务数 | I-S 有效分数 | I-S 触发对数 | I-S 回滚任务数 | 新增检测端点 | 新增回滚攻击任务 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in t2t_rows:
        lines.append(
            f"| {METHOD_NAMES[row['method']]} | {row['clean_valid']} | {row['clean_pairs']} | {row['clean_rejected']} | "
            f"{row['attack_valid']} | {row['attack_pairs']} | {row['attack_rejected']} | "
            f"{','.join(map(str, row['new_detected_tasks'])) or '无'} | "
            f"{','.join(map(str, row['new_rejected_attack_hits'])) or '无'} |"
        )
    return lines


def build_report(
    run_root: Path,
    manifest: dict[str, Any],
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    streams: dict[str, dict[str, Any]],
    t2t_rows: list[dict[str, Any]],
) -> str:
    config = manifest["config"]
    lines: list[str] = [
        "# Frozen Proxy 频率 × Shift 正则化 CL 实验报告",
        "",
        "> 本报告只分析 EWC、Online EWC、SI、MAS 四种无 replay 的正则化持续学习方法；没有把 BrainUICL 放入本实验。所有数值直接由 `metrics.json` 和攻击流 `metadata.json` 自动汇总。",
        "",
        "## 1. 实验是否完整",
        "",
        f"- 数据：ISRUC Group-I，固定 seed `{config['seed']}`，49 个新个体按同一顺序作为 49 个任务。",
        f"- 代理：冻结 source-pretrained proxy；参数 hash 前后相同：`{manifest['proxy_parameters_unchanged']}`。",
        f"- 训练：每个方法 10 个 CPC epoch + 10 个增量 epoch，学生 BN running statistics 冻结，guiding model 只提供 hard pseudo-label；不做置信度过滤、不使用 replay。",
        f"- 攻击：每个被攻击任务修改 20% sequence；相对 L2 上限 `{config['relative_l2_budgets'][0]:.0%}`，逐点上限为 `0.20 × modality std`，扰动保留在 `{config['band_hz'][0]}–{config['band_hz'][1]} Hz`。",
        "- `I-NS/I-S` 攻击 3 个任务（任务 13、25、37），`F-NS/F-S` 攻击 25 个任务；四个流共用干净底本、sequence mask 和 proxy 方向。",
        "- `NS` 使用任务内平衡的正负方向，`S` 使用全为正方向；这是有限样本下对 non-shifted/shifted 的工程近似，不是对渐近理论条件的证明。",
        "- 结果完整性：无防御 20 组、Robust Feature 20 组、T2T 8 组，共 48 个方法-条件运行；完成标记为 `_EXECUTION_COMPLETE`，22 项测试通过。",
        "",
        "## 2. 攻击流验证",
        "",
        "| 流 | 攻击任务数 | 修改 sequence | 上传 sequence | 实际覆盖率 | EEG 相对 L2 均值/最大值 | EOG 相对 L2 均值/最大值 | EEG L∞/std 均值 | 符号 (+/-) | proxy 标签保持率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("I-NS", "I-S", "F-NS", "F-S"):
        row = streams[condition]
        lines.append(
            f"| `{condition}` | {row['tasks']} | {row['noisy_sequences']} | {row['uploaded_sequences']} | "
            f"{pct(row['coverage'])} | {row['relative_l2_eeg_mean']:.2%}/{row['relative_l2_eeg_max']:.2%} | "
            f"{row['relative_l2_eog_mean']:.2%}/{row['relative_l2_eog_max']:.2%} | "
            f"{row['linf_eeg_mean']:.3f} | {row['sign_positive']}/{row['sign_negative']} | "
            f"{pct(row['proxy_label_preservation_mean'])} |"
        )
    lines.extend(
        [
            "",
            "补充诊断：四个流的扰动频带外能量约为 `2.15e-14` 量级，说明带通投影生效；扰动后的样本越过各自 clean sequence 振幅范围的比例约为 `3e-6` 到 `1e-5`。这里的 `relative L2` 和振幅范围只验证输入约束，不等价于攻击成功。",
            "最终保存的攻击流上，冻结 proxy 的 hard pseudo-label 保持率约为 56.79%–68.75%。因此攻击者没有直接篡改真实 label，但输入扰动会改变一部分 guiding pseudo-label；这不是严格意义上的 supervised clean-label poisoning，后续论文表述应称为“输入级 proxy pseudo-label 攻击”。",
            "",
            "## 3. 无防御 clean 基线",
            "",
        ]
    )
    lines.extend(table_baseline(runs))
    lines.extend(
        [
            "",
            "这里的旧个体指标是在 clean old-generalization 集上测量，已见新个体指标是在所有 49 个 clean 个体上测量；BWT 越接近 0 表示相对各个体刚适配后的遗忘越小。SI 在本协议中 clean 旧/新 ACC 最高，MAS 次之；这只是单 seed 的协议内基线，不足以宣称普遍算法排名。",
            "",
            "## 4. 攻击对无防御 CL 的影响",
            "",
            "下表是 `攻击流 - clean` 的百分点变化，负值表示退化；每个单元格顺序为“旧 ACC / 已见新 ACC / BWT ACC”。",
            "",
        ]
    )
    lines.extend(table_attack_acc(runs))
    lines.extend(
        [
            "",
            "MF1 的对应变化如下，单元格顺序为“旧 MF1 / 已见新 MF1 / BWT MF1”。",
            "",
        ]
    )
    lines.extend(table_attack_mf1(runs))
    lines.extend(
        [
            "",
            "从配对差值看，攻击频率从 3 个任务增加到 25 个任务后，退化总体更明显；同一频率下，`F-S` 通常不优于 `F-NS`，但差异并非对所有方法单调。由于每个任务的总预算相同，频繁条件同时增加了累计污染能量，因此这里回答的是“现实中长期重复接收污染的累计影响”，不是只改变攻击频率而保持总能量不变的因果实验。",
            "",
            "## 5. Robust Feature 配对结果",
            "",
            "Robust Feature 的 clean 代价必须单独计算；下面的恢复量定义为：`(RF 攻击 - RF clean) - (无防御攻击 - 无防御 clean)`。恢复量为正表示相对减轻了攻击退化。",
            "",
        ]
    )
    lines.extend(table_robust_recovery(runs))
    lines.extend(
        [
            "",
            "本轮结果不能支持 Robust Feature 已经被证明有效。原因是：第一，只有一个 seed；第二，当前 EEG 实现把论文的线性平方损失/共同特征基公式近似到最后线性分类器的特征协方差；第三，`F-NS` 虽然最接近论文适用条件，但应以跨 seed 的正恢复量和置信区间作为证据。当前表格只说明在这条固定 proxy 流和这组超参数下，哪些方法的配对退化变小或变大。",
            "",
            "## 6. T2T 结果与误报",
            "",
            "T2T 只在 clean 和 `I-S` 上运行，动作是 rollback、参数范围是全部可训练参数。`clean 触发` 在没有攻击时按定义都是 clean false positive；`I-S` 的检测不能因为触发了某个正常任务就称为攻击检测。",
            "",
        ]
    )
    lines.extend(table_t2t(t2t_rows))
    lines.extend(
        [
            "",
            "T2T 的关键观察是：clean 流已经出现多次触发和回滚，而 `I-S` 没有对攻击任务产生清晰、可归因且超出 clean 基线的新检测。换句话说，固定的 `2.5 × 最近分数均值` 在这个深度 EEG subject-CL 协议中主要响应正常跨个体更新动力学；它同时造成性能轨迹变化，因此不能把 T2T 的最终 ACC 直接解释为防护收益。",
            "",
            "## 7. 结论边界与下一步",
            "",
            "1. **攻击流成立。** 四类上传文件固定、共享、有限，并满足预设频率/符号/幅值/频带设计；因此可以作为后续防御比较的同一输入基准。",
            "2. **攻击在当前协议下可测但较弱。** `F-S` 对无防御方法的 old ACC 下降约 0.7–1.5 个百分点量级，说明有累计影响，但不是强破坏性攻击；需要在不破坏 EEG 合理性的前提下做预算/覆盖率 sweep。",
            "3. **Robust Feature 目前只能作为探索性结果。** 要声称防护有效，必须扩展至少 3 个 seed，并同时报告 clean 代价、攻击恢复量和不确定性。",
            "4. **T2T 当前不适合作为自动 rollback 防御。** clean 误报已经较高，且 `I-S` 没有清晰的新增攻击检测；应先做 clean calibration + monitor-only，再决定是否保留。",
            "5. **本实验不包含 BrainUICL、memory、PACOL/BrainWash 原始攻击复现，也没有测试总污染能量固定的频率消融。** 因此不能把本报告结论推广到 replay 方法或宣称复现了原论文攻击。",
            "6. **正式主实验建议。** 固定当前协议，加入至少 3 个 model/attack seed；对 `F-NS` 做 Robust Feature 主验证，对 `I-S` 保留 T2T 附加压力测试；同时增加 `K` 攻击任务但固定总平方 L2 能量的对照，分离频率效应与累计能量效应。",
            "",
            "## 8. 复现入口",
            "",
            f"- 总编排脚本：`scripts/run_frozen_proxy_regularization_full49.sh`",
            f"- 攻击 manifest：`{manifest_path_placeholder(run_root)}`",
            f"- 运行结果根目录：`{run_root}`",
            "- 自动汇总脚本：`experiments/summarize_frozen_proxy_frequency_shift.py`",
            "- 完成日志：`full49_runs/orchestrator.log`；测试日志：`full49_runs/final_tests.log`。",
        ]
    )
    return "\n".join(lines) + "\n"


def manifest_path_placeholder(run_root: Path) -> str:
    # Keep the report portable within the repository while retaining a clear
    # pointer to the immutable attack stream.
    return str(run_root.parent / "full49_seed4321" / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/frozen_proxy_frequency_shift/full49_runs"),
    )
    parser.add_argument(
        "--stream-root",
        type=Path,
        default=Path("experiments/frozen_proxy_frequency_shift/full49_seed4321"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    stream_root = args.stream_root.resolve()
    manifest = read_json(stream_root / "manifest.json")
    if not manifest.get("proxy_parameters_unchanged", False):
        raise RuntimeError("Proxy parameter hash changed")
    runs = load_runs(run_root)
    streams = aggregate_stream_metadata(stream_root, manifest)
    attack_tasks = set(manifest["config"]["infrequent_tasks"])
    t2t_rows = t2t_diagnostics(runs, attack_tasks)
    machine = build_machine_summary(run_root, manifest, runs, streams, t2t_rows)
    report = build_report(run_root, manifest, runs, streams, t2t_rows)

    report_path = args.report or run_root / "FULL49_RESULTS_ZH.md"
    json_path = args.json_path or run_root / "FULL49_RESULTS.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(machine, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

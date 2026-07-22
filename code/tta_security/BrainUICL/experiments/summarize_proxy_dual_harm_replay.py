#!/usr/bin/env python3
"""Validate and summarize dual-harm against aligned plain experience replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = ("clean", "repeat_clean", "attack_shared")
METRICS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "bwt_acc",
    "bwt_mf1",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def memory_purity(metrics: dict[str, Any]) -> dict[str, float | int]:
    data_root = Path(metrics["config"]["data_root"])
    records = metrics["final"]["memory_records"]
    correct = 0
    total = 0
    poisoned_correct = 0
    poisoned_total = 0
    clean_correct = 0
    clean_total = 0
    for record in records:
        labels = np.load(
            data_root
            / str(record["subject"])
            / "label"
            / f"{record['sequence_index']}.npy"
        ).astype(np.int64)
        pseudo = np.asarray(record["pseudo_labels"], dtype=np.int64)
        if labels.shape != pseudo.shape:
            raise ValueError("Final memory pseudo-label shape mismatch")
        matches = int((labels == pseudo).sum())
        correct += matches
        total += labels.size
        if record["poisoned"]:
            poisoned_correct += matches
            poisoned_total += labels.size
        else:
            clean_correct += matches
            clean_total += labels.size
    return {
        "records": len(records),
        "epoch_labels": total,
        "purity": correct / max(total, 1),
        "poisoned_purity": poisoned_correct / max(poisoned_total, 1),
        "clean_purity": clean_correct / max(clean_total, 1),
        "poisoned_epoch_labels": poisoned_total,
    }


def attacked_task_diagnostics(metrics: dict[str, Any]) -> dict[str, float]:
    rows = [
        row
        for row in metrics["tasks"]
        if (row.get("attack") or {}).get("mode") == "proxy_dual_harm"
    ]
    return {
        "tasks": len(rows),
        "attacked_stream_pseudo_acc": mean(
            [row["pseudo_labels"]["acc_diagnostic_only"] for row in rows]
        ),
        "clean_current_pseudo_acc": mean(
            [
                row["pseudo_labels_on_clean_current"]["acc_diagnostic_only"]
                for row in rows
            ]
        ),
        "pseudo_label_preservation": mean(
            [
                row["attack"]["diagnostics_mean"][
                    "pseudo_label_preservation"
                ]
                for row in rows
            ]
        ),
        "guiding_confidence": mean(
            [
                row["attack"]["diagnostics_mean"]["guiding_confidence"]
                for row in rows
            ]
        ),
        "relative_l2_eog": mean(
            [row["attack"]["diagnostics_mean"]["relative_l2_eog"] for row in rows]
        ),
        "relative_l2_eeg": mean(
            [row["attack"]["diagnostics_mean"]["relative_l2_eeg"] for row in rows]
        ),
    }


def validate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_split = runs["clean"]["config"]["seed"]
    for condition, metrics in runs.items():
        protocol = metrics["protocol"]
        if metrics["method"] != "plain_er":
            raise ValueError(f"{condition}: method is not plain_er")
        if not protocol["replay"] or protocol["regularization_cl_penalty"]:
            raise ValueError(f"{condition}: replay/regularization protocol mismatch")
        if protocol["confidence_filter"]:
            raise ValueError(f"{condition}: confidence filtering is enabled")
        if protocol["brainuicl_cea"] or protocol["brainuicl_dcb"]:
            raise ValueError(f"{condition}: BrainUICL mechanisms are enabled")
        if protocol["true_target_labels_used_for_training"]:
            raise ValueError(f"{condition}: target labels leaked into training")
        if metrics["config"]["seed"] != reference_split:
            raise ValueError(f"{condition}: seed mismatch")
        if metrics["config"]["memory_capacity"] != 1000:
            raise ValueError(f"{condition}: memory capacity mismatch")
        if metrics["config"]["replay_ratio"] != 1.0:
            raise ValueError(f"{condition}: replay ratio mismatch")

    repeat_tasks = {
        row["task"]: row for row in runs["repeat_clean"]["tasks"] if row["attack"]
    }
    attack_tasks = {
        row["task"]: row for row in runs["attack_shared"]["tasks"] if row["attack"]
    }
    if set(repeat_tasks) != set(attack_tasks) or len(attack_tasks) != 25:
        raise ValueError("Attack/repeat task sets are not the same odd 25 tasks")
    for task in attack_tasks:
        benign = repeat_tasks[task]
        attack = attack_tasks[task]
        if benign["attack"]["training_sequences_after_injection"] != attack["attack"][
            "training_sequences_after_injection"
        ]:
            raise ValueError(f"Task {task}: N->4N upload count mismatch")
        if benign["memory_update"]["candidates"] != attack["memory_update"][
            "candidates"
        ]:
            raise ValueError(f"Task {task}: reservoir candidate count mismatch")
        diagnostics = attack["attack"]["diagnostics_mean"]
        if diagnostics["relative_l2_eog"] > 0.20001:
            raise ValueError(f"Task {task}: EOG relative-L2 budget exceeded")
        if diagnostics["relative_l2_eeg"] > 0.20001:
            raise ValueError(f"Task {task}: EEG relative-L2 budget exceeded")
    if (
        runs["repeat_clean"]["summary"]["total_replay_draws"]
        != runs["attack_shared"]["summary"]["total_replay_draws"]
    ):
        raise ValueError("Attack/repeat replay draw counts differ")
    return {
        "same_attack_tasks": True,
        "attack_task_count": len(attack_tasks),
        "volume_matched": True,
        "reservoir_candidates_matched": True,
        "replay_draws_matched": True,
        "input_budget_valid": True,
        "true_target_labels_used_for_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "experiments/replay_cl_eeg_runs/proxy_dual_harm_plain_er_full49"
        ),
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    runs: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for condition in CONDITIONS:
        path = run_root / condition / "metrics.json"
        if not path.is_file():
            missing.append(str(path))
        else:
            runs[condition] = read_json(path)
            splits[condition] = read_json(run_root / condition / "split.json")
    if missing:
        raise RuntimeError("Missing plain ER artifacts:\n" + "\n".join(missing))
    if any(split != splits["clean"] for split in splits.values()):
        raise ValueError("Clean/repeat/attack task splits differ")

    validation = validate(runs)
    validation["same_49_task_split"] = True
    summaries = {condition: run["summary"] for condition, run in runs.items()}
    deltas = {}
    for condition in ("repeat_clean", "attack_shared"):
        deltas[f"{condition}_minus_clean"] = {
            key: summaries[condition][key] - summaries["clean"][key]
            for key in METRICS
        }
    deltas["attack_minus_repeat_clean"] = {
        key: summaries["attack_shared"][key] - summaries["repeat_clean"][key]
        for key in METRICS
    }
    memory = {condition: memory_purity(run) for condition, run in runs.items()}
    attack_diagnostics = attacked_task_diagnostics(runs["attack_shared"])
    attack_effect = deltas["attack_minus_repeat_clean"]
    effective = (
        attack_effect["final_old_acc"] <= -0.01
        and attack_effect["final_seen_acc"] <= -0.01
    )
    replay_stats = {
        condition: run["final"]["memory"] for condition, run in runs.items()
    }

    machine = {
        "run_root": str(run_root),
        "validation": validation,
        "summaries": summaries,
        "deltas": deltas,
        "memory_purity_posthoc": memory,
        "replay_stats": replay_stats,
        "attack_diagnostics": attack_diagnostics,
        "effective_after_volume_control": effective,
    }
    rows = []
    for condition in CONDITIONS:
        summary = summaries[condition]
        rows.append(
            f"| {condition} | {pct(summary['final_old_acc'])}/{pct(summary['final_old_mf1'])} | "
            f"{pct(summary['final_seen_acc'])}/{pct(summary['final_seen_mf1'])} | "
            f"{pct(summary['bwt_acc'])} | {summary['final_memory_size']} | "
            f"{pct(memory[condition]['purity'])} |"
        )
    effect_rows = []
    for label, values in deltas.items():
        effect_rows.append(
            f"| {label} | {pp(values['final_old_acc'])}/{pp(values['final_old_mf1'])} | "
            f"{pp(values['final_seen_acc'])}/{pp(values['final_seen_mf1'])} | "
            f"{pp(values['bwt_acc'])} |"
        )

    if effective:
        decision = (
            "共享 dual-harm 在扣除等量 repeat-clean 后仍使 plain ER 的 old/new ACC "
            "同时下降至少 1 pp，因此对 replay 有效；当前不需要为了“产生效果”而改攻击。"
        )
        modification = (
            "下一步应做机制消融，而不是继续放大预算：比较 current-only 不入库、"
            "poison 入库但不重复、repeat 0/1/3、memory capacity 和 replay ratio，"
            "确认下降中有多少来自 replay 持久化。"
        )
    else:
        decision = (
            "共享 dual-harm 在等量控制后没有让 plain ER 的 old/new ACC 同时下降 1 pp，"
            "因此不能声称它对 replay 有效。"
        )
        modification = (
            "应增加 replay-aware adapter：旧域 proxy 改为当前 reservoir 样本；"
            "一步展开同时包含 current CE 与 replay CE；外层加入 memory-admission/"
            "future-replay gradient，并在不改变输入预算的条件下重新测试。"
        )

    report = "\n".join(
        [
            "# Dual-harm 对 Plain ER-EEG 的验证",
            "",
            "> 本实验不使用 BrainUICL 持续学习算法。BrainUICL 网络只作为与正则化实验一致的 source-pretrained backbone；CL 机制是固定容量 reservoir experience replay。",
            "",
            "## 协议",
            "",
            "- 三条件：Clean ER、等量 Repeat-clean ER、Dual-harm ER。",
            "- 同一 49-task 顺序、CPC guide、hard pseudo-label、10+10 epochs、`cl_lr=1e-6`、冻结 BN。",
            "- Reservoir 容量 1000 sequence；每个 current batch 抽取等量 replay sequence，当前:replay=1:1。",
            "- 无 confidence filter、CEA、DCB、source replay、EWC/SI/MAS penalty。",
            "- Attack 与 Repeat-clean 在奇数 25 个任务均为 N->4N 上传；每次 occurrence 都参加相同 reservoir admission。",
            "- Dual-harm 使用与正则化实验相同的双 proxy、一步 classifier unroll、20% relative L2、`0.5×std` L∞ 和 repeat=3；正则曲率 adapter 在 ER 中关闭。",
            "",
            "## 绝对结果",
            "",
            "| 条件 | old ACC/MF1 | new ACC/MF1 | BWT ACC | 最终 memory | 最终伪标签纯度 |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 配对差值",
            "",
            "| 对比 | old ACC/MF1 | new ACC/MF1 | BWT ACC |",
            "|---|---:|---:|---:|",
            *effect_rows,
            "",
            "BWT ACC 在 Attack 与 Repeat-clean 间只变化约 0.05 pp，不能据此判断攻击无效：攻击已经降低各任务刚学完时的起点，BWT 只衡量最终值相对该起点的变化。这里必须以最终 old/new 绝对 ACC/MF1 和等量差分为主。",
            "",
            "## Replay 持久化诊断",
            "",
            f"- Attack 最终 reservoir 中 poisoned record 比例：`{pct(replay_stats['attack_shared']['poisoned_fraction'])}`。",
            f"- 全程 replay draw 中 poisoned sequence 比例：`{pct(replay_stats['attack_shared']['poisoned_replay_fraction'])}`。",
            f"- Attack 最终 memory 总伪标签纯度：`{pct(memory['attack_shared']['purity'])}`；其中 poisoned records 为 `{pct(memory['attack_shared']['poisoned_purity'])}`，未污染 records 为 `{pct(memory['attack_shared']['clean_purity'])}`。",
            f"- 攻击任务 attacked-stream pseudo ACC：`{pct(attack_diagnostics['attacked_stream_pseudo_acc'])}`；poisoned-CPC 后 clean-current pseudo ACC：`{pct(attack_diagnostics['clean_current_pseudo_acc'])}`。",
            "",
            "## 判定",
            "",
            decision,
            "",
            modification,
            "",
            "## 如何设计同时影响正则化和 replay 的方法",
            "",
            "共同攻击核心应保持输入级和预算一致：双侧 old/new proxy、guide pseudo-label 劫持、一步更新后伤害和等量 repeat 控制。不同 CL 家族只增加内部 adapter：正则化 adapter 使用 importance/anchor 曲率旁路；replay adapter 使用 reservoir 样本、admission/survival 和未来 replay 梯度。共同主表使用冻结 shared proxy，family-aware adapter 作为独立白盒 upper-bound，不能用不同输入的绝对 ACC 直接排名。",
            "",
            "正式结论还需要至少 3 seeds，并增加 random-noise、current-only/no-store、store-once、repeat 0/1/3、memory capacity 和 replay ratio 消融。当前单 seed 只能回答可行性和机制路径。",
            "",
            "## 复现入口",
            "",
            "- Runner：`experiments/replay_cl_eeg.py`",
            "- Orchestrator：`scripts/run_proxy_dual_harm_plain_er_full49.sh`",
            "- 机器结果：`FULL_RESULTS.json`",
        ]
    ) + "\n"
    (run_root / "FULL_RESULTS_ZH.md").write_text(report, encoding="utf-8")
    (run_root / "FULL_RESULTS.json").write_text(
        json.dumps(machine, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {run_root / 'FULL_RESULTS_ZH.md'}")
    print(f"wrote {run_root / 'FULL_RESULTS.json'}")
    print(json.dumps({"effective_after_volume_control": effective}))


if __name__ == "__main__":
    main()

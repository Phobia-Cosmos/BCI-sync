#!/usr/bin/env python3
"""Validate fixed-upload replay defenses and write a Chinese result report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "plain_er",
    "spr_er",
    "puridiver_memory_ce",
    "puridiver_cru",
)
METHOD_NAMES = {
    "plain_er": "Plain ER",
    "spr_er": "SPR-style ER",
    "puridiver_memory_ce": "PuriDivER memory + CE",
    "puridiver_cru": "PuriDivER memory + C/R/U",
}
CONDITIONS = ("clean", "repeat_clean", "attack_fixed")
METRICS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "bwt_acc",
    "bwt_mf1",
)
COMMON_CONFIG_KEYS = (
    "seed",
    "batch",
    "num_worker",
    "max_subjects",
    "ssl_epoch",
    "incremental_epoch",
    "ssl_lr",
    "cl_lr",
    "beta1",
    "beta2",
    "weight_decay",
    "grad_clip",
    "freeze_bn_stats",
    "memory_capacity",
    "replay_ratio",
    "retention_milestones",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def require_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def memory_purity(metrics: dict[str, Any]) -> dict[str, float | int]:
    """Offline-only purity calculation that respects an SPR epoch mask."""

    data_root = Path(metrics["config"]["data_root"])
    result = {
        "records": 0,
        "epoch_labels": 0,
        "purity": 0.0,
        "poisoned_purity": 0.0,
        "clean_purity": 0.0,
        "poisoned_epoch_labels": 0,
        "retained_epoch_fraction": 0.0,
    }
    correct = total = poisoned_correct = poisoned_total = clean_correct = clean_total = 0
    raw_epochs = 0
    for record in metrics["final"]["memory_records"]:
        labels = np.load(
            data_root
            / str(record["subject"])
            / "label"
            / f"{record['sequence_index']}.npy"
        ).astype(np.int64)
        pseudo = np.asarray(record["pseudo_labels"], dtype=np.int64)
        if labels.shape != pseudo.shape:
            raise ValueError("Final memory pseudo-label shape mismatch")
        mask = record.get("epoch_mask")
        if mask is None:
            valid = np.ones(labels.shape, dtype=bool)
        else:
            valid = np.asarray(mask, dtype=bool)
            if valid.shape != labels.shape:
                raise ValueError("Final memory epoch mask shape mismatch")
        labels = labels[valid]
        pseudo = pseudo[valid]
        matches = int((labels == pseudo).sum())
        correct += matches
        total += labels.size
        raw_epochs += valid.size
        if record["poisoned"]:
            poisoned_correct += matches
            poisoned_total += labels.size
        else:
            clean_correct += matches
            clean_total += labels.size
    result.update(
        {
            "records": len(metrics["final"]["memory_records"]),
            "epoch_labels": total,
            "purity": correct / max(total, 1),
            "poisoned_purity": poisoned_correct / max(poisoned_total, 1),
            "clean_purity": clean_correct / max(clean_total, 1),
            "poisoned_epoch_labels": poisoned_total,
            "retained_epoch_fraction": total / max(raw_epochs, 1),
        }
    )
    return result


def attack_rows(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["task"]): row
        for row in metrics["tasks"]
        if row.get("attack") is not None
    }


def validate_summary(label: str, metrics: dict[str, Any]) -> None:
    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{label}: summary is missing")
    for key in METRICS:
        if key not in summary:
            raise ValueError(f"{label}: summary is missing {key}")
    final = metrics.get("final", {})
    old = final.get("old_generalization", {})
    require_close(label + " final_old_acc", summary["final_old_acc"], old["acc"])
    require_close(label + " final_old_mf1", summary["final_old_mf1"], old["mf1"])
    seen = final.get("seen_subjects", {})
    if len(seen) != 49:
        raise ValueError(f"{label}: expected 49 final seen-subject evaluations")
    require_close(
        label + " final_seen_acc",
        summary["final_seen_acc"],
        mean([float(row["acc"]) for row in seen.values()]),
    )
    require_close(
        label + " final_seen_mf1",
        summary["final_seen_mf1"],
        mean([float(row["mf1"]) for row in seen.values()]),
    )


def validate_run(
    method: str,
    condition: str,
    metrics: dict[str, Any],
    split: dict[str, Any],
    reference_split: dict[str, Any],
    reference_config: dict[str, Any],
) -> None:
    label = f"{method}/{condition}"
    require_equal(label + " method", metrics.get("method"), method)
    require_equal(label + " split", split, reference_split)
    config = metrics["config"]
    for key in COMMON_CONFIG_KEYS:
        require_equal(label + " config " + key, config.get(key), reference_config.get(key))
    protocol = metrics.get("protocol", {})
    if not protocol.get("replay") or protocol.get("regularization_cl_penalty"):
        raise ValueError(f"{label}: replay/regularization protocol mismatch")
    if protocol.get("brainuicl_cea") or protocol.get("brainuicl_dcb"):
        raise ValueError(f"{label}: BrainUICL-specific mechanism is enabled")
    if protocol.get("true_target_labels_used_for_training"):
        raise ValueError(f"{label}: target labels leaked into training")
    expected_subjects = [int(value) for value in split["new_order"]]
    tasks = metrics.get("tasks", [])
    if len(tasks) != 49:
        raise ValueError(f"{label}: expected 49 tasks, got {len(tasks)}")
    if [int(row["task"]) for row in tasks] != list(range(1, 50)):
        raise ValueError(f"{label}: invalid task indices")
    if [int(row["subject"]) for row in tasks] != expected_subjects:
        raise ValueError(f"{label}: subject order mismatch")
    validate_summary(label, metrics)


def validate_method(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    repeat = attack_rows(runs["repeat_clean"])
    attack = attack_rows(runs["attack_fixed"])
    expected_tasks = set(range(1, 50, 2))
    if set(repeat) != expected_tasks or set(attack) != expected_tasks:
        raise ValueError("repeat-clean and attack task sets must be the odd 25 tasks")
    hashes: dict[int, str] = {}
    source_roots: dict[int, str] = {}
    for task in sorted(expected_tasks):
        benign = repeat[task]["attack"]
        poisoned = attack[task]["attack"]
        require_equal("repeat mode", benign["mode"], "benign_repeat")
        require_equal("attack mode", poisoned["mode"], "proxy_dual_harm")
        if not poisoned.get("fixed_shared_upload"):
            raise ValueError(f"task {task}: attack is not a fixed shared upload")
        if poisoned.get("generated_inputs"):
            raise ValueError(f"task {task}: victim regenerated fixed upload")
        require_equal(
            f"task {task} upload count",
            benign["training_sequences_after_injection"],
            poisoned["training_sequences_after_injection"],
        )
        require_equal(
            f"task {task} admission candidates",
            benign["memory_admission_candidates"],
            poisoned["memory_admission_candidates"],
        )
        diagnostics = poisoned.get("source_generation_diagnostics_mean", {})
        if float(diagnostics.get("relative_l2_eog", np.inf)) > 0.200001:
            raise ValueError(f"task {task}: fixed EOG upload violates relative-L2 budget")
        if float(diagnostics.get("relative_l2_eeg", np.inf)) > 0.200001:
            raise ValueError(f"task {task}: fixed EEG upload violates relative-L2 budget")
        digest = poisoned.get("fixed_upload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"task {task}: fixed upload digest is missing")
        hashes[task] = digest
        source_roots[task] = str(poisoned.get("source_generation_run"))
    return {
        "same_25_attack_tasks": True,
        "volume_matched": True,
        "input_budget_valid": True,
        "fixed_upload_hashes": hashes,
        "fixed_upload_source_roots": source_roots,
    }


def run_root_artifacts(run_root: Path) -> tuple[
    dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]
]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    splits: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for method in METHODS:
        runs[method] = {}
        for condition in CONDITIONS:
            directory = run_root / method / condition
            metrics_path = directory / "metrics.json"
            split_path = directory / "split.json"
            if not metrics_path.is_file() or not split_path.is_file():
                missing.extend(str(path) for path in (metrics_path, split_path) if not path.is_file())
                continue
            runs[method][condition] = read_json(metrics_path)
            splits[f"{method}/{condition}"] = read_json(split_path)
    if missing:
        raise RuntimeError("Missing fixed-shared replay artifacts:\n" + "\n".join(missing))
    return runs, splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/replay_cl_eeg_runs/fixed_shared_defenses_full49"),
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    runs, splits = run_root_artifacts(run_root)
    reference_key = "plain_er/clean"
    reference_split = splits[reference_key]
    reference_config = runs["plain_er"]["clean"]["config"]

    validation: dict[str, Any] = {}
    for method in METHODS:
        for condition in CONDITIONS:
            validate_run(
                method,
                condition,
                runs[method][condition],
                splits[f"{method}/{condition}"],
                reference_split,
                reference_config,
            )
        validation[method] = validate_method(runs[method])

    shared_hashes = validation["plain_er"]["fixed_upload_hashes"]
    shared_sources = validation["plain_er"]["fixed_upload_source_roots"]
    for method in METHODS[1:]:
        require_equal(
            method + " fixed upload hashes",
            validation[method]["fixed_upload_hashes"],
            shared_hashes,
        )
        require_equal(
            method + " fixed upload source roots",
            validation[method]["fixed_upload_source_roots"],
            shared_sources,
        )

    summaries = {
        method: {
            condition: runs[method][condition]["summary"]
            for condition in CONDITIONS
        }
        for method in METHODS
    }
    deltas = {
        method: {
            "repeat_clean_minus_clean": {
                key: summaries[method]["repeat_clean"][key]
                - summaries[method]["clean"][key]
                for key in METRICS
            },
            "attack_minus_repeat_clean": {
                key: summaries[method]["attack_fixed"][key]
                - summaries[method]["repeat_clean"][key]
                for key in METRICS
            },
        }
        for method in METHODS
    }
    purity = {
        method: {
            condition: memory_purity(runs[method][condition])
            for condition in CONDITIONS
        }
        for method in METHODS
    }
    replay = {
        method: {
            condition: runs[method][condition]["final"]["memory"]
            for condition in CONDITIONS
        }
        for method in METHODS
    }

    baseline = deltas["plain_er"]["attack_minus_repeat_clean"]
    recovery = {
        method: {
            key: deltas[method]["attack_minus_repeat_clean"][key] - baseline[key]
            for key in ("final_old_acc", "final_old_mf1", "final_seen_acc", "final_seen_mf1")
        }
        for method in METHODS
    }
    rows = []
    delta_rows = []
    recovery_rows = []
    for method in METHODS:
        clean = summaries[method]["clean"]
        attack = summaries[method]["attack_fixed"]
        effect = deltas[method]["attack_minus_repeat_clean"]
        rows.append(
            f"| {METHOD_NAMES[method]} | {pct(clean['final_old_acc'])}/{pct(clean['final_old_mf1'])} | "
            f"{pct(clean['final_seen_acc'])}/{pct(clean['final_seen_mf1'])} | "
            f"{pct(attack['final_old_acc'])}/{pct(attack['final_old_mf1'])} | "
            f"{pct(attack['final_seen_acc'])}/{pct(attack['final_seen_mf1'])} | "
            f"{pct(purity[method]['attack_fixed']['purity'])} |"
        )
        delta_rows.append(
            f"| {METHOD_NAMES[method]} | {pp(effect['final_old_acc'])}/{pp(effect['final_old_mf1'])} | "
            f"{pp(effect['final_seen_acc'])}/{pp(effect['final_seen_mf1'])} | "
            f"{pp(effect['bwt_acc'])} | {pct(replay[method]['attack_fixed']['poisoned_fraction'])} | "
            f"{pct(replay[method]['attack_fixed']['poisoned_replay_fraction'])} |"
        )
        recovered = recovery[method]
        recovery_rows.append(
            f"| {METHOD_NAMES[method]} | {pp(recovered['final_old_acc'])}/{pp(recovered['final_old_mf1'])} | "
            f"{pp(recovered['final_seen_acc'])}/{pp(recovered['final_seen_mf1'])} |"
        )

    report = "\n".join(
        [
            "# 固定共享上传下的 Replay 防御验证",
            "",
            "> 所有方法从相同 source-supervised checkpoint 开始，使用相同 49-task ISRUC 顺序、CPC guide hard pseudo-label、10+10 epoch、冻结 BN、1000-sequence memory 和 current:replay=1:1。所有 target 真实标签只在评估和本报告的离线 memory-purity 诊断中读取，不进入 CPC、训练损失、过滤、memory admission 或攻击文件选择。",
            "",
            "## 公平性检查",
            "",
            "- 每种方法都完成 Clean、等量 Repeat-clean、Fixed-upload Attack 三条件；攻击与 Repeat-clean 在相同奇数 25 个任务执行 N->4N 上传。",
            "- 每个 Fixed-upload Attack 在加载前记录所有攻击 sequence 的 SHA-256 聚合摘要；四种方法在每一个攻击任务的摘要和源目录完全相同。攻击文件不会在被测方法内重新生成。",
            "- 固定文件来自早期 Plain ER source run 的已保存 upload，生成预算为每模态 relative L2 不超过 20%。因此这是固定 stream 比较，不是每个防御各自的 adaptive white-box 上界；文件最初由 Plain ER 状态生成的来源不对称性仍须在解释中保留。",
            "",
            "## 绝对结果",
            "",
            "| 方法 | Clean old ACC/MF1 | Clean new ACC/MF1 | Attack old ACC/MF1 | Attack new ACC/MF1 | Attack memory 伪标签纯度 |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 扣除上传量后的攻击效应",
            "",
            "| 方法 | Attack - Repeat old ACC/MF1 | Attack - Repeat new ACC/MF1 | BWT ACC | attack memory poisoned record | poisoned replay draw |",
            "|---|---:|---:|---:|---:|---:|",
            *delta_rows,
            "",
            "`Attack - Repeat-clean` 扣除了额外 clean 上传、额外训练步数和同量 memory admission 的影响。负数表示攻击仍造成退化；BWT 不能单独用于判断攻击是否有效，因为攻击也会降低每个任务刚适配完成时的起点。",
            "",
            "## 相对 Plain ER 的残余退化变化",
            "",
            "| 方法 | old ACC/MF1 recovery | new ACC/MF1 recovery |",
            "|---|---:|---:|",
            *recovery_rows,
            "",
            "正值表示该方法的 Attack - Repeat-clean 退化小于 Plain ER；它不是跨方法的绝对性能排名，也不是多 seed 显著性结论。",
            "",
            "## 方法边界",
            "",
            "- `SPR-style ER` 仅在任务结束时使用 student epoch embedding 和 admission pseudo-label 做 SPR 风格的 epoch mask；被拒绝 epoch 在 replay CE 中使用 ignore index。它不是原始 SPR 的完整 delayed expert/self-supervised 在线流程。",
            "- `PuriDivER memory + CE` 使用 task-end purity/diversity sequence pruning，但训练损失仍为 hard pseudo-label CE。",
            "- `PuriDivER memory + C/R/U` 在当前和 replay batch 的 student snapshot loss/uncertainty 上拟合两层 GMM，并使用 clean/relabel/unlabeled 分支损失。memory 仍是 task-end 选择，因此应称为 PuriDivER-style EEG hybrid，不能等同宣称为原论文的逐 minibatch PuriDivER 复现。",
            "- 本轮为单 seed 固定流实验。要支持部署级防御结论，仍需独立 attack-generation seed、至少 3 个 paired training seeds、低预算/低频率 sweep、clean FPR/selection cost 和真实 EEG 伪迹约束。",
            "",
            "## 产物",
            "",
            "- Runner: `experiments/replay_cl_eeg.py`",
            "- 汇总器: `experiments/summarize_fixed_shared_replay_defenses.py`",
            "- 固定攻击源: `" + str(next(iter(shared_sources.values()))) + "`",
        ]
    ) + "\n"
    machine = {
        "run_root": str(run_root),
        "validation": validation,
        "summaries": summaries,
        "deltas": deltas,
        "memory_purity_posthoc": purity,
        "replay_stats": replay,
        "recovery_vs_plain_er": recovery,
    }
    (run_root / "FULL_RESULTS_ZH.md").write_text(report, encoding="utf-8")
    (run_root / "FULL_RESULTS.json").write_text(
        json.dumps(machine, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {run_root / 'FULL_RESULTS_ZH.md'}")
    print(f"wrote {run_root / 'FULL_RESULTS.json'}")


if __name__ == "__main__":
    main()

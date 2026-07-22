#!/usr/bin/env python3
"""Validate and summarize the adaptive white-box regularization CL runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("ewc", "online_ewc", "si", "mas")
METHOD_NAMES = {
    "ewc": "EWC",
    "online_ewc": "Online EWC",
    "si": "SI",
    "mas": "MAS",
}
METRIC_KEYS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "old_aaa",
    "old_aaf1",
    "old_fr",
    "bwt_acc",
    "bwt_mf1",
    "mean_pseudo_acc_diagnostic_only",
    "mean_pseudo_mf1_diagnostic_only",
)
RUN_DIRECTORIES = {
    "ewc": "proxy_dual_harm_repeat3_odd25_full49_e10",
    "online_ewc": "proxy_dual_harm_repeat3_odd25_online_ewc_full49_e10",
    "si": "proxy_dual_harm_repeat3_odd25_si_full49_e10",
    "mas": "proxy_dual_harm_repeat3_odd25_mas_full49_e10",
}
BENIGN_REPEAT_DIRECTORY = "benign_repeat3_odd25_full49_e10"
ATTACK_TASKS = tuple(range(1, 50, 2))
COMMON_CONFIG = {
    "seed": 4321,
    "batch": 16,
    "num_worker": 0,
    "max_subjects": 0,
    "ssl_epoch": 10,
    "incremental_epoch": 10,
    "lr": 1e-4,
    "ssl_lr": 1e-6,
    "cl_lr": 1e-6,
    "beta1": 0.5,
    "beta2": 0.99,
    "weight_decay": 3e-4,
    "grad_clip": 5.0,
    "freeze_bn_stats": True,
    "eval_max_batches": 0,
    "retention_milestones": [10, 25, 49],
    "checkpoint_milestones": [10, 25, 49],
    "dataset": "ISRUC",
}
COMPARISON_CONFIG_KEYS = (
    "data_root",
    "input_checkpoint_root",
    *COMMON_CONFIG.keys(),
)
METHOD_REGULARIZER_CONFIG = {
    "ewc": {"ewc_strength": 5000.0},
    "online_ewc": {
        "online_ewc_strength": 6500.0,
        "online_ewc_decay": 1.0,
    },
    "si": {"si_strength": 1500000.0, "si_xi": 1e-6},
    "mas": {"mas_strength": 3000.0, "mas_decay": 1.0},
}
ATTACK_CONFIG = {
    "attack_mode": "proxy_dual_harm",
    "attack_tasks": list(ATTACK_TASKS),
    "attack_fraction": 1.0,
    "attack_eps_scale": 0.5,
    "attack_max_relative_l2": 0.2,
    "attack_steps": 3,
    "attack_inner_lr": 1e-4,
    "attack_param_scope": "classifier",
    "attack_reference_batch": 4,
    "attack_random_start": True,
    "attack_generation_batch": 4,
    "attack_target_weight": 5.0,
    "attack_conflict_weight": 1.0,
    "attack_gradient_norm_weight": 0.25,
    "attack_virtual_old_weight": 1.0,
    "attack_virtual_new_weight": 1.0,
    "attack_new_proxy_weight": 1.0,
    "attack_curvature_scale": 1.0,
    "attack_min_confidence": 0.85,
    "attack_confidence_weight": 2.0,
    "attack_l2_weight": 0.01,
    "attack_proxy_repeat": 3,
    "defense_mode": "none",
}
BENIGN_REPEAT_CONFIG = {
    "attack_mode": "benign_repeat",
    "attack_tasks": list(ATTACK_TASKS),
    "attack_fraction": 0.05,
    "attack_eps_scale": 0.1,
    "attack_max_relative_l2": 0.0,
    "attack_proxy_repeat": 3,
    "defense_mode": "none",
}
PROTOCOL_INVARIANTS = {
    "pseudo_labels": "all hard argmax labels from the guiding model",
    "confidence_filter": False,
    "replay": False,
    "dcb": False,
    "cea": False,
    "student_batch_norm_running_stats": "frozen",
    "true_target_labels_used_for_training": False,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: got {actual!r}, expected {expected!r}")


def assert_close(label: str, actual: Any, expected: Any) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"{label}: got {actual!r}, expected {expected!r}")


def assert_config_values(
    label: str,
    config: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            assert_close(f"{label} {key}", actual, value)
        else:
            assert_equal(f"{label} {key}", actual, value)


def validate_common_config(method: str, config: dict[str, Any]) -> None:
    assert_config_values(method, config, COMMON_CONFIG)
    assert_config_values(method, config, METHOD_REGULARIZER_CONFIG[method])


def validate_protocol(method: str, metrics: dict[str, Any], mode: str | None) -> None:
    protocol = metrics.get("protocol", {})
    for key, value in PROTOCOL_INVARIANTS.items():
        assert_equal(f"{method} protocol {key}", protocol.get(key), value)
    if mode is None:
        if protocol.get("attack") not in {None, "none"}:
            raise RuntimeError(f"{method} clean run has an attack protocol")
    else:
        assert_equal(f"{method} protocol attack", protocol.get("attack"), mode)
        assert_equal(
            f"{method} learner received attacker reference inputs",
            protocol.get("attacker_reference_inputs_used_by_learner"),
            False,
        )


def validate_task_order(
    label: str,
    split: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks = metrics.get("tasks", [])
    expected_order = [int(value) for value in split["new_order"]]
    if len(tasks) != len(expected_order) or len(tasks) != 49:
        raise RuntimeError(
            f"{label} is incomplete: expected 49 tasks, got {len(tasks)}"
        )
    if [int(row.get("task", -1)) for row in tasks] != list(range(1, 50)):
        raise RuntimeError(f"{label} has an invalid task index sequence")
    if [int(row.get("subject", -1)) for row in tasks] != expected_order:
        raise RuntimeError(f"{label} subject-order mismatch")
    return tasks


def validate_summary(method: str, metrics: dict[str, Any]) -> None:
    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"{method} has no final summary")
    missing = [key for key in METRIC_KEYS if key not in summary]
    if missing:
        raise RuntimeError(f"{method} summary is missing {missing}")
    final = metrics.get("final", {})
    final_old = final.get("old_generalization", {})
    for metric, key in (("acc", "final_old_acc"), ("mf1", "final_old_mf1")):
        assert_close(f"{method} summary {key}", summary[key], final_old.get(metric))
    seen = final.get("seen_subjects", {})
    if len(seen) != 49:
        raise RuntimeError(f"{method} final seen-subject evaluation is incomplete")
    assert_close(
        f"{method} summary final_seen_acc",
        summary["final_seen_acc"],
        mean([float(row["acc"]) for row in seen.values()]),
    )
    assert_close(
        f"{method} summary final_seen_mf1",
        summary["final_seen_mf1"],
        mean([float(row["mf1"]) for row in seen.values()]),
    )


def validate_finite_diagnostics(label: str, diagnostics: dict[str, Any]) -> None:
    for key, value in diagnostics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"{label} diagnostic {key} is not finite")


def validate_run(
    method: str,
    config: dict[str, Any],
    split: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    assert_equal(f"Attack method", metrics.get("method"), method)
    validate_common_config(method, config)
    assert_config_values(f"{method} attack", config, ATTACK_CONFIG)
    validate_protocol(method, metrics, "proxy_dual_harm")
    tasks = validate_task_order(method, split, metrics)
    attacked = [row for row in tasks if row.get("attack") is not None]
    if len(attacked) != 25:
        raise RuntimeError(
            f"{method} has {len(attacked)} attacked tasks; expected 25"
        )
    expected_tasks = set(ATTACK_TASKS)
    actual_tasks = {int(row["task"]) for row in attacked}
    if actual_tasks != expected_tasks:
        raise RuntimeError(f"Attack-task mismatch for {method}")
    for row in attacked:
        attack = row["attack"]
        total = int(attack.get("total_sequences", -1))
        expected_indices = list(range(total))
        assert_equal(f"{method} attack mode", attack.get("mode"), "proxy_dual_harm")
        assert_equal(f"{method} attack task", attack.get("task"), row["task"])
        assert_equal(f"{method} attack subject", attack.get("subject"), row["subject"])
        assert_equal(
            f"{method} attack poisoned sequence count",
            attack.get("poisoned_sequences"),
            total,
        )
        assert_close(f"{method} attack poison fraction", attack.get("poison_fraction"), 1.0)
        assert_equal(f"{method} attack indices", attack.get("poison_indices"), expected_indices)
        generated_inputs = attack.get("generated_inputs")
        if generated_inputs is not None:
            assert_equal(
                f"{method} attack generated inputs", generated_inputs, True
            )
        assert_equal(f"{method} attack learner replay", attack.get("learner_replay"), False)
        assert_equal(f"{method} attack proxy repeat", attack.get("proxy_repeat"), 3)
        assert_equal(
            f"{method} attack injected copies",
            attack.get("injected_proxy_sequences"),
            3 * total,
        )
        assert_equal(
            f"{method} attack learner sequence count",
            attack.get("training_sequences_after_injection"),
            4 * total,
        )
        diagnostics = attack.get("diagnostics_mean", {})
        validate_finite_diagnostics(method, diagnostics)
        for modality in ("eog", "eeg"):
            relative_l2 = float(diagnostics[f"relative_l2_{modality}"])
            linf = float(diagnostics[f"linf_{modality}"])
            epsilon = float(diagnostics[f"epsilon_{modality}"])
            if relative_l2 > 0.2 + 1e-7:
                raise RuntimeError(f"{method} attack exceeds relative L2 budget")
            if relative_l2 <= 0.0:
                raise RuntimeError(f"{method} attack has no input perturbation")
            if linf > epsilon + 1e-12:
                raise RuntimeError(f"{method} attack exceeds L-infinity budget")
    validate_summary(method, metrics)
    return attacked


def validate_benign_repeat(
    method: str,
    config: dict[str, Any],
    split: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the volume-matched clean control for one method."""
    assert_equal(f"Control method", metrics.get("method"), method)
    validate_common_config(method, config)
    assert_config_values(f"{method} benign-repeat", config, BENIGN_REPEAT_CONFIG)
    validate_protocol(method, metrics, "benign_repeat")
    tasks = validate_task_order(f"{method} benign-repeat", split, metrics)
    controlled = [row for row in tasks if row.get("attack") is not None]
    if len(controlled) != 25:
        raise RuntimeError(
            f"{method} benign-repeat has {len(controlled)} control tasks; expected 25"
        )
    expected_tasks = set(ATTACK_TASKS)
    actual_tasks = {int(row["task"]) for row in controlled}
    if actual_tasks != expected_tasks:
        raise RuntimeError(f"Benign-repeat task mismatch for {method}")
    for row in controlled:
        attack = row["attack"]
        diagnostics = attack.get("diagnostics_mean", {})
        total = int(attack.get("total_sequences", -1))
        assert_equal(f"{method} benign-repeat mode", attack.get("mode"), "benign_repeat")
        assert_equal(f"{method} benign-repeat task", attack.get("task"), row["task"])
        assert_equal(f"{method} benign-repeat subject", attack.get("subject"), row["subject"])
        assert_equal(f"{method} benign-repeat poisoned sequence count", attack.get("poisoned_sequences"), 0)
        assert_close(f"{method} benign-repeat poison fraction", attack.get("poison_fraction"), 0.0)
        assert_equal(
            f"{method} benign-repeat indices",
            attack.get("poison_indices"),
            list(range(total)),
        )
        assert_equal(
            f"{method} benign-repeat generated inputs",
            attack.get("generated_inputs"),
            False,
        )
        assert_equal(f"{method} benign-repeat learner replay", attack.get("learner_replay"), False)
        assert_equal(f"{method} benign-repeat proxy repeat", attack.get("proxy_repeat"), 3)
        assert_equal(
            f"{method} benign-repeat injected copies",
            attack.get("injected_proxy_sequences"),
            3 * total,
        )
        assert_equal(
            f"{method} benign-repeat learner sequence count",
            attack.get("training_sequences_after_injection"),
            4 * total,
        )
        assert_equal(
            f"{method} benign-repeat EOG perturbation",
            diagnostics.get("relative_l2_eog"),
            0.0,
        )
        assert_equal(
            f"{method} benign-repeat EEG perturbation",
            diagnostics.get("relative_l2_eeg"),
            0.0,
        )
    validate_summary(method, metrics)
    return controlled


def validate_clean_run(
    method: str,
    config: dict[str, Any],
    split: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    assert_equal("Clean method", metrics.get("method"), method)
    validate_common_config(method, config)
    validate_protocol(method, metrics, None)
    tasks = validate_task_order(f"{method} clean", split, metrics)
    if any(row.get("attack") is not None for row in tasks):
        raise RuntimeError(f"{method} clean run contains an attack task")
    validate_summary(method, metrics)


def assert_matching_context(
    method: str,
    clean_config: dict[str, Any],
    clean_split: dict[str, Any],
    clean_metrics: dict[str, Any],
    compared_label: str,
    compared_config: dict[str, Any],
    compared_split: dict[str, Any],
    compared_metrics: dict[str, Any],
) -> None:
    assert_equal(f"{method} {compared_label} split", compared_split, clean_split)
    for key in COMPARISON_CONFIG_KEYS:
        actual = compared_config.get(key)
        expected = clean_config.get(key)
        if isinstance(expected, float):
            assert_close(f"{method} {compared_label} {key}", actual, expected)
        else:
            assert_equal(f"{method} {compared_label} {key}", actual, expected)
    for name in ("old_generalization", "source_train", "validation"):
        expected_metrics = clean_metrics["initial"][name]
        actual_metrics = compared_metrics["initial"][name]
        assert_equal(
            f"{method} {compared_label} initial {name} n_epochs",
            actual_metrics["n_epochs"],
            expected_metrics["n_epochs"],
        )
        for metric in ("acc", "mf1"):
            assert_close(
                f"{method} {compared_label} initial {name} {metric}",
                actual_metrics[metric],
                expected_metrics[metric],
            )


def assert_volume_matched(
    method: str,
    attacked: list[dict[str, Any]],
    controlled: list[dict[str, Any]],
) -> None:
    attack_by_task = {int(row["task"]): row for row in attacked}
    control_by_task = {int(row["task"]): row for row in controlled}
    assert_equal(
        f"{method} attack/control task set",
        sorted(attack_by_task),
        sorted(control_by_task),
    )
    for task in sorted(attack_by_task):
        attack = attack_by_task[task]["attack"]
        control = control_by_task[task]["attack"]
        for key in (
            "total_sequences",
            "poison_indices",
            "injected_proxy_sequences",
            "training_sequences_after_injection",
            "proxy_repeat",
        ):
            assert_equal(
                f"{method} task {task} volume match {key}",
                attack.get(key),
                control.get(key),
            )


def attack_diagnostics(attacked: list[dict[str, Any]]) -> dict[str, float | int]:
    diagnostics = [row["attack"]["diagnostics_mean"] for row in attacked]
    deleted = [
        int(row["attack"].get("materialized_files_deleted", 0))
        for row in attacked
    ]
    generated = [int(row["attack"]["poisoned_sequences"]) for row in attacked]
    injected = [
        int(row["attack"].get("injected_proxy_sequences", 0))
        for row in attacked
    ]
    return {
        "attacked_tasks": len(attacked),
        "generated_sequences": sum(generated),
        "injected_proxy_copies": sum(injected),
        "materialized_files_deleted": sum(deleted),
        "mean_relative_l2_eog": mean(
            [float(row["relative_l2_eog"]) for row in diagnostics]
        ),
        "mean_relative_l2_eeg": mean(
            [float(row["relative_l2_eeg"]) for row in diagnostics]
        ),
        "mean_guiding_confidence_at_generation": mean(
            [float(row["guiding_confidence"]) for row in diagnostics]
        ),
        "mean_proxy_pseudo_preservation": mean(
            [float(row["pseudo_label_preservation"]) for row in diagnostics]
        ),
        "mean_target_hit_rate": mean(
            [float(row["target_hit_rate"]) for row in diagnostics]
        ),
        "mean_target_shift_conflict": mean(
            [float(row["target_shift_conflict"]) for row in diagnostics]
        ),
        "mean_final_gradient_conflict": mean(
            [float(row["gradient_conflict_final"]) for row in diagnostics]
        ),
        "mean_attacked_stream_pseudo_acc": mean(
            [float(row["pseudo_labels"]["acc_diagnostic_only"]) for row in attacked]
        ),
        "mean_clean_current_pseudo_acc_after_poisoned_cpc": mean(
            [
                float(row["pseudo_labels_on_clean_current"]["acc_diagnostic_only"])
                for row in attacked
            ]
        ),
    }


def benign_repeat_diagnostics(
    controlled: list[dict[str, Any]],
) -> dict[str, float]:
    return {
        "mean_clean_current_pseudo_acc_after_repeated_clean_cpc": mean(
            [
                float(row["pseudo_labels_on_clean_current"]["acc_diagnostic_only"])
                for row in controlled
            ]
        ),
        "mean_clean_current_pseudo_mf1_after_repeated_clean_cpc": mean(
            [
                float(row["pseudo_labels_on_clean_current"]["mf1_diagnostic_only"])
                for row in controlled
            ]
        ),
    }


def load_results(
    run_root: Path,
    clean_root: Path,
    benign_root: Path,
) -> dict[str, Any]:
    clean_root_summary = read_json(clean_root / "summary.json")
    clean_config = read_json(clean_root / "config.json")
    clean_split = read_json(clean_root / "split.json")
    results: dict[str, Any] = {}
    reference_order: list[int] | None = None
    reference_protocol: dict[str, Any] | None = None
    benign_root_summary = read_json(benign_root / "summary.json")
    benign_config = read_json(benign_root / "config.json")
    benign_split = read_json(benign_root / "split.json")
    benign_order: list[int] | None = None
    protocol_keys = (
        "seed",
        "batch",
        "ssl_epoch",
        "incremental_epoch",
        "cl_lr",
        "attack_mode",
        "attack_fraction",
        "attack_eps_scale",
        "attack_max_relative_l2",
        "attack_steps",
        "attack_proxy_repeat",
        "attack_tasks",
    )
    for method in METHODS:
        method_root = run_root / RUN_DIRECTORIES[method]
        config = read_json(method_root / "config.json")
        split = read_json(method_root / "split.json")
        metrics = read_json(method_root / method / "metrics.json")
        attacked = validate_run(method, config, split, metrics)
        clean_metrics = read_json(clean_root / method / "metrics.json")
        validate_clean_run(method, clean_config, clean_split, clean_metrics)
        benign_metrics = read_json(benign_root / method / "metrics.json")
        controlled = validate_benign_repeat(
            method, benign_config, benign_split, benign_metrics
        )
        assert_matching_context(
            method,
            clean_config,
            clean_split,
            clean_metrics,
            "attack",
            config,
            split,
            metrics,
        )
        assert_matching_context(
            method,
            clean_config,
            clean_split,
            clean_metrics,
            "benign-repeat",
            benign_config,
            benign_split,
            benign_metrics,
        )
        assert_volume_matched(method, attacked, controlled)
        assert_equal(
            f"{method} clean root summary",
            clean_root_summary.get(method),
            clean_metrics["summary"],
        )
        assert_equal(
            f"{method} benign-repeat root summary",
            benign_root_summary.get(method),
            benign_metrics["summary"],
        )
        method_root_summary = read_json(method_root / "summary.json")
        assert_equal(
            f"{method} attack root summary",
            method_root_summary.get(method),
            metrics["summary"],
        )
        order = [int(value) for value in split["new_order"]]
        if reference_order is None:
            reference_order = order
        elif order != reference_order:
            raise RuntimeError(f"Cross-method subject-order mismatch for {method}")
        control_order = [int(value) for value in benign_split["new_order"]]
        if benign_order is None:
            benign_order = control_order
        elif control_order != benign_order:
            raise RuntimeError(f"Cross-method control subject-order mismatch for {method}")
        if control_order != order:
            raise RuntimeError(f"Attack/control subject-order mismatch for {method}")
        protocol = {key: config[key] for key in protocol_keys}
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise RuntimeError(f"Cross-method protocol mismatch for {method}")
        clean_summary = clean_metrics["summary"]
        attacked_summary = metrics["summary"]
        benign_summary = benign_metrics["summary"]
        deltas = {
            key: float(attacked_summary[key]) - float(clean_summary[key])
            for key in METRIC_KEYS
        }
        benign_deltas = {
            key: float(benign_summary[key]) - float(clean_summary[key])
            for key in METRIC_KEYS
        }
        attack_vs_benign = {
            key: float(attacked_summary[key]) - float(benign_summary[key])
            for key in METRIC_KEYS
        }
        diagnostics = attack_diagnostics(attacked)
        control_diagnostics = benign_repeat_diagnostics(controlled)
        diagnostics["clean_current_pseudo_acc_delta_vs_benign_repeat"] = (
            diagnostics["mean_clean_current_pseudo_acc_after_poisoned_cpc"]
            - control_diagnostics[
                "mean_clean_current_pseudo_acc_after_repeated_clean_cpc"
            ]
        )
        results[method] = {
            "run_root": str(method_root),
            "clean": clean_summary,
            "attack": attacked_summary,
            "benign_repeat": benign_summary,
            "delta": deltas,
            "delta_benign_repeat_vs_clean": benign_deltas,
            "delta_attack_vs_benign_repeat": attack_vs_benign,
            "diagnostics": diagnostics,
            "benign_repeat_diagnostics": control_diagnostics,
            "benign_control_tasks": len(controlled),
        }
    return {
        "protocol": reference_protocol,
        "new_order": reference_order,
        "benign_repeat_root": str(benign_root),
        "results": results,
    }


def build_report(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    results = payload["results"]
    lines = [
        "# 正则化 CL 自适应白盒 Proxy 攻击实验",
        "",
        "> 本报告分析 EWC、Online EWC、SI、MAS 的单 seed 强白盒上界实验。该协议只修改输入流，但使用 20% relative-L2 上限、攻击 25/49 个任务并重复上传生成序列，不应表述为低预算或隐蔽攻击。",
        "",
        "## 实验协议",
        "",
        "- 数据与模型：ISRUC Group-I、seed 4321、BrainUICL source checkpoint、49 个固定顺序的新个体、冻结学生 BN running statistics。",
        "- CL：每个任务 10 个 CPC guide epoch + 10 个学生 epoch，`cl_lr=1e-6`；无 replay、无 confidence filter，目标真实标签只用于离线评估。",
        f"- 攻击频率：奇数任务，共 `{len(protocol['attack_tasks'])}/49` 个任务；每个攻击任务替换全部原始上传 sequence。",
        f"- 输入预算：逐点 `L∞ <= {protocol['attack_eps_scale']:.2f} × modality std`，每 sequence/模态 `relative L2 <= {protocol['attack_max_relative_l2']:.0%}`，{protocol['attack_steps']} 步投影符号梯度。",
        f"- 交互放大：每个生成 sequence 额外上传 `{protocol['attack_proxy_repeat']}` 份，因此一个攻击任务的 learner 输入为 1 份生成上传 + 3 份重复代理上传。",
        "- 白盒权限：攻击者读取当前 student、CPC-adapted guide 和 EWC/SI/MAS 的 importance/anchor；source-train 输入只作为私有 old proxy 计算外层目标，不进入 learner optimizer。",
        "- 攻击 surrogate：生成器只对 classifier 做一次 `inner_lr=1e-4` 的可微梯度步；真实 learner 则对全模型使用 Adam、`cl_lr=1e-6`、weight decay、gradient clipping 和 10 epochs。该 surrogate 用于寻找有害输入方向，不是对真实优化器的逐步精确复现。",
        "",
        "## 最终结果",
        "",
        "| 方法 | Clean old ACC/MF1 | Repeat-clean old ACC/MF1 | Attack old ACC/MF1 | Clean new ACC/MF1 | Repeat-clean new ACC/MF1 | Attack new ACC/MF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = results[method]
        clean = item["clean"]
        benign = item["benign_repeat"]
        attack = item["attack"]
        lines.append(
            f"| {METHOD_NAMES[method]} | {pct(clean['final_old_acc'])}/{pct(clean['final_old_mf1'])} | "
            f"{pct(benign['final_old_acc'])}/{pct(benign['final_old_mf1'])} | "
            f"{pct(attack['final_old_acc'])}/{pct(attack['final_old_mf1'])} | "
            f"{pct(clean['final_seen_acc'])}/{pct(clean['final_seen_mf1'])} | "
            f"{pct(benign['final_seen_acc'])}/{pct(benign['final_seen_mf1'])} | "
            f"{pct(attack['final_seen_acc'])}/{pct(attack['final_seen_mf1'])} |"
        )
    lines.extend(
        [
            "",
            "## 配对差值",
            "",
            "| 方法 | Repeat-clean − clean old | Attack − clean old | Attack − repeat-clean old | Repeat-clean − clean new | Attack − clean new | Attack − repeat-clean new |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = results[method]
        delta = item["delta"]
        delta_control = item["delta_benign_repeat_vs_clean"]
        delta_residual = item["delta_attack_vs_benign_repeat"]
        lines.append(
            f"| {METHOD_NAMES[method]} | "
            f"{pp(delta_control['final_old_acc'])}/{pp(delta_control['final_old_mf1'])} | "
            f"{pp(delta['final_old_acc'])}/{pp(delta['final_old_mf1'])} | "
            f"{pp(delta_residual['final_old_acc'])}/{pp(delta_residual['final_old_mf1'])} | "
            f"{pp(delta_control['final_seen_acc'])}/{pp(delta_control['final_seen_mf1'])} | "
            f"{pp(delta['final_seen_acc'])}/{pp(delta['final_seen_mf1'])} | "
            f"{pp(delta_residual['final_seen_acc'])}/{pp(delta_residual['final_seen_mf1'])} |"
        )
    lines.extend(
        [
            "",
            "## 攻击诊断",
            "",
            "| 方法 | 生成 sequence | 额外上传副本 | EEG/EOG relative L2 | 生成输入上的 guide 最大类置信度 | proxy 标签保持率 | 目标标签命中率 | attacked-stream pseudo ACC | poisoned-CPC 后 clean-current pseudo ACC | 相对 repeat-clean 变化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = results[method]["diagnostics"]
        lines.append(
            f"| {METHOD_NAMES[method]} | {row['generated_sequences']} | {row['injected_proxy_copies']} | "
            f"{pct(row['mean_relative_l2_eeg'])}/{pct(row['mean_relative_l2_eog'])} | "
            f"{pct(row['mean_guiding_confidence_at_generation'])} | "
            f"{pct(row['mean_proxy_pseudo_preservation'])} | {pct(row['mean_target_hit_rate'])} | "
            f"{pct(row['mean_attacked_stream_pseudo_acc'])} | "
            f"{pct(row['mean_clean_current_pseudo_acc_after_poisoned_cpc'])} | "
            f"{pp(row['clean_current_pseudo_acc_delta_vs_benign_repeat'])} |"
        )
    lines.extend(
        [
            "",
            "## 组合设计与观测证据",
            "",
            "1. **组合目标。** 生成器把 PACOL 式梯度方向项、BrainWash 式一步 surrogate、source/current 双代理损失和正则曲率权重放在同一目标中；设计意图是同时影响旧域稳定性与新域可塑性，并寻找较少受历史正则保护的方向。",
            "2. **两级交互。** 同一上传先进入 CPC guide，再由 guide 生成 hard pseudo-label 训练 student；攻击任务上 pseudo-label ACC 约为 27%–29%，而 poisoned-CPC 后 untouched clean-current pseudo ACC 约为 57%–60%，说明影响不仅停留在单个输入的瞬时预测。",
            "3. **流式累积。** 25 个奇数任务和每 sequence 3 个额外副本让有害上传在极低 `cl_lr` 下获得足够采样次数，并把更新后的模型与 importance/anchor 带入后续任务。",
            "4. **等量控制。** repeat-clean 使用完全相同的任务、索引和 `N -> 4N` 输入量；四方法 `attack - repeat-clean` 的 old/new ACC 仍下降 4.60–17.28/5.48–17.15 pp，因此主要退化不能只由数据量解释。",
            "5. **证据边界。** 当前实验验证的是完整组合相对 clean 与等量 clean-repeat 的效果；没有 random-noise、target-only、no-curvature、no-unroll、repeat 0/1/3 等组件消融，因此不能把下降量分别归因给任何单个设计项。",
            "",
            "## 结论边界",
            "",
            "- 这是单 seed、强白盒、data-stream upper-bound；证明当前正则化 CL 在足够强的自适应输入与上传频率控制下会发生明显 old/new 双侧退化。",
            "- volume-matched benign-repeat 控制只重复 clean sequence，不改变输入值；因此 `attack - benign-repeat` 是扣除数据量/采样次数后的残余退化，`benign-repeat - clean` 则报告重复上传本身的影响。",
            "- 20% relative L2 较大，尚未证明生理不可察觉。后续应固定本攻击结构，逐步降低预算，并加入频带、幅值、EDF 伪迹和跨 seed 约束。",
            "- source proxy 比 BrainWash 原论文的模型反演权限更强；本实验应称为 adaptive white-box proxy upper bound，而不是 BrainWash 原样复现。",
            "- 历史正式产物中的攻击生成诊断是先对 generation batch 做等权平均、再对攻击任务做等权平均；性能 ACC/MF1 来自完整评估集，不受该诊断聚合口径影响。当前代码已为后续新运行增加 sequence-weighted 诊断。",
            "",
            "## 复现文件",
            "",
            "- 攻击实现：`experiments/regularization_cl_attacks.py`",
            "- CL runner：`experiments/regularization_cl_eeg.py`",
            "- 汇总器：`experiments/summarize_proxy_dual_harm.py`",
            "- 机器可读结果：`experiments/regularization_cl_eeg_runs/proxy_dual_harm_repeat3_odd25_FULL_RESULTS.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/regularization_cl_eeg_runs"),
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path(
            "experiments/regularization_cl_eeg_runs/"
            "clean49_bn_frozen_e10_lr1e6_seed4321"
        ),
    )
    parser.add_argument(
        "--benign-root",
        type=Path,
        default=Path(
            "experiments/regularization_cl_eeg_runs/"
            "benign_repeat3_odd25_full49_e10"
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    clean_root = args.clean_root.resolve()
    benign_root = args.benign_root.resolve()
    payload = load_results(run_root, clean_root, benign_root)
    json_path = args.json_path or (
        run_root / "proxy_dual_harm_repeat3_odd25_FULL_RESULTS.json"
    )
    report_path = args.report or (
        run_root / "proxy_dual_harm_repeat3_odd25_FULL_RESULTS_ZH.md"
    )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(build_report(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

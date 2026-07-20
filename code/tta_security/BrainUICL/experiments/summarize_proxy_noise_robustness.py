#!/usr/bin/env python3
"""Summarize the aligned full49 proxy-noise robustness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "experiments" / "proxy_noise_robustness" / "full49"
METHODS = (
    "brainuicl",
    "finetune",
    "ewc",
    "online_ewc",
    "si",
    "mas",
    "spr_eeg",
    "puridiver_eeg",
)
METHOD_LABELS = {
    "brainuicl": "BrainUICL",
    "finetune": "Finetune",
    "ewc": "EWC",
    "online_ewc": "Online EWC",
    "si": "SI",
    "mas": "MAS",
    "spr_eeg": "SPR-EEG",
    "puridiver_eeg": "PuriDivER-EEG",
}
LEVELS = ("loose", "medium", "strict")
LEVEL_LABELS = {"loose": "宽松", "medium": "中等", "strict": "严格"}
LEVEL_CONFIGS = {
    "loose": {
        "requested_noise_fraction": 1.0,
        "eps_over_modality_std": 0.5,
        "max_relative_l2_eog": None,
        "max_relative_l2_eeg": None,
    },
    "medium": {
        "requested_noise_fraction": 0.5,
        "eps_over_modality_std": 0.5,
        "max_relative_l2_eog": 0.2,
        "max_relative_l2_eeg": 0.2,
    },
    "strict": {
        "requested_noise_fraction": 0.2,
        "eps_over_modality_std": 0.1,
        "max_relative_l2_eog": 0.05,
        "max_relative_l2_eeg": 0.05,
    },
}
METRIC_KEYS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "bwt_acc",
    "bwt_mf1",
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment result: {path}")
    return json.loads(path.read_text())


def metric_subset(payload: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in METRIC_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Missing metrics {missing} in payload with keys {sorted(payload)}")
    return {key: float(payload[key]) for key in METRIC_KEYS}


def load_clean_metrics(run_root: Path) -> dict[str, dict[str, float]]:
    clean: dict[str, dict[str, float]] = {}
    clean["brainuicl"] = metric_subset(
        read_json(
            REPO_ROOT
            / "experiments"
            / "rttdp_brainuicl_runs"
            / "aligned_full49_bn_frozen_lr1e6_seed4321"
            / "clean"
            / "metrics.json"
        )["summary"]
    )
    regularization = read_json(
        REPO_ROOT
        / "experiments"
        / "regularization_cl_eeg_runs"
        / "clean49_bn_frozen_e10_lr1e6_seed4321"
        / "summary.json"
    )
    for method in ("finetune", "ewc", "online_ewc", "si", "mas"):
        clean[method] = metric_subset(regularization[method])
    for method, directory in (("spr_eeg", "spr_eeg"), ("puridiver_eeg", "puridiver_eeg")):
        payload = read_json(
            run_root / "clean_baselines_seed4321" / directory / "defense_summary.json"
        )
        clean[method] = metric_subset(payload["plasticity"])
    return clean


def load_noisy_metrics(level_root: Path) -> dict[str, dict[str, float]]:
    noisy: dict[str, dict[str, float]] = {}
    noisy["brainuicl"] = metric_subset(
        read_json(
            level_root
            / "generator_brainuicl"
            / "noise_proxy_guided"
            / "metrics.json"
        )["summary"]
    )
    for method in ("finetune", "ewc", "online_ewc", "si", "mas"):
        payload = read_json(
            level_root / "regularization" / method / "metrics.json"
        )
        noisy[method] = metric_subset(payload["summary"])
    for method, directory in (("spr_eeg", "spr_eeg"), ("puridiver_eeg", "puridiver_eeg")):
        payload = read_json(level_root / directory / "defense_summary.json")
        noisy[method] = metric_subset(payload["plasticity"])
    return noisy


def refresh_regularization_summaries(run_root: Path) -> None:
    for level in LEVELS:
        regularization_root = run_root / f"{level}_seed4321" / "regularization"
        combined = {}
        for method in ("finetune", "ewc", "online_ewc", "si", "mas"):
            payload = read_json(regularization_root / method / "metrics.json")
            combined[method] = payload["summary"]
        (regularization_root / "summary.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
        )


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row["attempted"]) for row in rows)
    if total == 0:
        return 0.0
    return sum(float(row[key]) * int(row["attempted"]) for row in rows) / total


def load_noise_stream(level_root: Path) -> dict[str, Any]:
    variant_root = level_root / "generator_brainuicl" / "noise_proxy_guided"
    metadata_paths = sorted(
        (variant_root / "noisy_uploads").glob("individual_*/metadata.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if len(metadata_paths) != 49:
        raise ValueError(f"Expected 49 upload metadata files under {variant_root}, found {len(metadata_paths)}")
    metadata = [read_json(path) for path in metadata_paths]
    tasks = [int(row["task"]) for row in metadata]
    if tasks != list(range(1, 50)):
        raise ValueError(f"Unexpected task order in {variant_root}: {tasks}")
    uploaded = sum(int(row["uploaded"]) for row in metadata)
    noisy = sum(int(row["noisy_sequences"]) for row in metadata)

    generator = read_json(variant_root / "metrics.json")
    diagnostics = generator["performance"]["noise_diagnostics"]
    if len(diagnostics) != 49:
        raise ValueError(f"Expected 49 noise diagnostic rows under {variant_root}, found {len(diagnostics)}")
    return {
        "subjects": len(metadata),
        "uploaded_sequences": uploaded,
        "noisy_sequences": noisy,
        "noise_coverage": noisy / uploaded,
        "candidate_mean_rel_eog": weighted_mean(diagnostics, "mean_rel_eog"),
        "candidate_mean_rel_eeg": weighted_mean(diagnostics, "mean_rel_eeg"),
        "candidate_confidence_pass_rate": weighted_mean(diagnostics, "adv_pass_rate"),
    }


def load_replay_metrics(level_root: Path) -> dict[str, dict[str, float | int]]:
    paths = {
        "brainuicl": level_root / "generator_brainuicl" / "noise_proxy_guided" / "metrics.json",
        "spr_eeg": level_root / "spr_eeg" / "defense_spr_external_proxy_noise" / "metrics.json",
        "puridiver_eeg": level_root
        / "puridiver_eeg"
        / "defense_puridiver_external_proxy_noise"
        / "metrics.json",
    }
    replay: dict[str, dict[str, float | int]] = {}
    for method, path in paths.items():
        summary = read_json(path)["summary"]
        replay[method] = {
            "high_confidence_candidate_rate": float(summary["high_confidence_candidate_rate"]),
            "candidate_acceptance_rate": float(summary["candidate_acceptance_rate"]),
            "pseudo_sequence_coverage": float(summary["pseudo_sequence_coverage"]),
            "final_buffer_sequences": int(summary["final_buffer_sequences"]),
        }
    return replay


def build_summary(run_root: Path) -> dict[str, Any]:
    clean = load_clean_metrics(run_root)
    levels: dict[str, Any] = {}
    for level in LEVELS:
        level_root = run_root / f"{level}_seed4321"
        noisy = load_noisy_metrics(level_root)
        methods: dict[str, Any] = {}
        for method in METHODS:
            methods[method] = {
                "clean": clean[method],
                "noise": noisy[method],
                "delta_noise_minus_clean": {
                    key: noisy[method][key] - clean[method][key] for key in METRIC_KEYS
                },
            }
        levels[level] = {
            "configuration": LEVEL_CONFIGS[level],
            "stream": load_noise_stream(level_root),
            "methods": methods,
            "replay": load_replay_metrics(level_root),
        }
    return {
        "protocol": {
            "dataset": "ISRUC Group-I",
            "seed": 4321,
            "new_subjects": 49,
            "ssl_epochs": 10,
            "incremental_epochs": 10,
            "ssl_lr": 1e-6,
            "cl_lr": 1e-6,
            "student_batch_norm_running_stats": "frozen",
            "evaluation_inputs": "clean",
            "labels_modified": False,
            "shared_proxy_noise_stream_across_methods": True,
            "proxy": {
                "steps": 5,
                "reference": "base_train",
                "reference_labels": "pseudo",
                "parameter_scope": "classifier",
                "conflict_weight": 5.0,
                "confidence_weight": 0.1,
                "gradient_norm_weight": 0.0,
                "raw_weight": 0.001,
                "l2_weight": 0.0005,
                "random_start": True,
            },
            "regularization": {
                "ewc_strength": 5000.0,
                "online_ewc_strength": 6500.0,
                "online_ewc_decay": 1.0,
                "si_strength": 1500000.0,
                "si_xi": 0.000001,
                "mas_strength": 3000.0,
                "mas_decay": 1.0,
            },
        },
        "clean": clean,
        "levels": levels,
    }


def pct(value: float, signed: bool = False) -> str:
    return f"{value * 100:+.2f}" if signed else f"{value * 100:.2f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Proxy 模型噪声下的 EEG 持续学习 full49 结果",
        "",
        "所有方法使用同一 ISRUC Group-I、seed 4321、49 个体顺序、10 个 CPC epoch、10 个增量 epoch、`ssl_lr=cl_lr=1e-6` 和冻结学生 BN running statistics。噪声只替换增量阶段上传的输入，标签、模型参数、优化器和损失设置不变；所有最终指标均在 clean 数据上评估。",
        "",
        "正则化参数与 clean 对照严格对齐：EWC 5000，Online EWC 6500、decay 1，SI 1500000、xi 0.000001，MAS 3000、decay 1。",
        "",
        "## 噪声流",
        "",
        "| 配置 | 名义覆盖率 | eps / modality std | relative L2 上限 | noisy / uploaded sequence | 实际覆盖率 | proxy 候选 EOG 相对 L2 | proxy 候选 EEG 相对 L2 | 候选置信度通过率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        config = summary["levels"][level]["configuration"]
        stream = summary["levels"][level]["stream"]
        rel_cap = config["max_relative_l2_eeg"]
        rel_cap_text = "无" if rel_cap is None else f"{pct(rel_cap)}%"
        lines.append(
            f"| {LEVEL_LABELS[level]} | {pct(config['requested_noise_fraction'])}% | "
            f"{config['eps_over_modality_std']:.1f} | {rel_cap_text} | "
            f"{stream['noisy_sequences']} / {stream['uploaded_sequences']} | "
            f"{pct(stream['noise_coverage'])}% | {pct(stream['candidate_mean_rel_eog'])}% | "
            f"{pct(stream['candidate_mean_rel_eeg'])}% | {pct(stream['candidate_confidence_pass_rate'])}% |"
        )
    lines.extend(
        [
            "",
            "相对 L2 是 proxy 为全部候选 sequence 生成扰动时的按 sequence 数加权均值；中等和严格配置只把随机选中的 50%/20% 候选写入上传流，其余 sequence 保持 clean。",
            "",
            "## Clean 基线",
            "",
            "| 方法 | Old ACC | Old MF1 | Final seen-new ACC | Final seen-new MF1 | BWT ACC | BWT MF1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = summary["clean"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {pct(row['final_old_acc'])}% | {pct(row['final_old_mf1'])}% | "
            f"{pct(row['final_seen_acc'])}% | {pct(row['final_seen_mf1'])}% | "
            f"{pct(row['bwt_acc'], signed=True)} pp | {pct(row['bwt_mf1'], signed=True)} pp |"
        )
    for level in LEVELS:
        lines.extend(
            [
                "",
                f"## {LEVEL_LABELS[level]}配置",
                "",
                "表内括号为 `noise - clean`，负值表示噪声下性能下降。",
                "",
                "| 方法 | Old ACC | Old MF1 | Final seen-new ACC | Final seen-new MF1 | BWT ACC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        methods = summary["levels"][level]["methods"]
        for method in METHODS:
            noisy = methods[method]["noise"]
            delta = methods[method]["delta_noise_minus_clean"]
            lines.append(
                f"| {METHOD_LABELS[method]} | {pct(noisy['final_old_acc'])}% ({pct(delta['final_old_acc'], True)} pp) | "
                f"{pct(noisy['final_old_mf1'])}% ({pct(delta['final_old_mf1'], True)} pp) | "
                f"{pct(noisy['final_seen_acc'])}% ({pct(delta['final_seen_acc'], True)} pp) | "
                f"{pct(noisy['final_seen_mf1'])}% ({pct(delta['final_seen_mf1'], True)} pp) | "
                f"{pct(noisy['bwt_acc'], True)} pp |"
            )
        lines.extend(
            [
                "",
                "| Replay 方法 | 高置信候选率 | 候选接收率 | 上传 sequence 进入 replay 的覆盖率 | 最终 buffer sequence |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        replay = summary["levels"][level]["replay"]
        for method in ("brainuicl", "spr_eeg", "puridiver_eeg"):
            row = replay[method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {pct(float(row['high_confidence_candidate_rate']))}% | "
                f"{pct(float(row['candidate_acceptance_rate']))}% | "
                f"{pct(float(row['pseudo_sequence_coverage']))}% | {row['final_buffer_sequences']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    json_output = args.json_output or run_root / "SUMMARY.json"
    markdown_output = args.markdown_output or run_root / "SUMMARY_ZH.md"
    refresh_regularization_summaries(run_root)
    summary = build_summary(run_root)
    json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    markdown_output.write_text(render_markdown(summary))
    print(json_output)
    print(markdown_output)


if __name__ == "__main__":
    main()

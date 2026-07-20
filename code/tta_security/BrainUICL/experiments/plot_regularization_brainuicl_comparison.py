#!/usr/bin/env python3
"""Plot clean EEG CL results for regularizers and the original BrainUICL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGULARIZATION = (
    REPO_ROOT
    / "experiments"
    / "regularization_cl_eeg_runs"
    / "clean49_bn_frozen_e10_lr1e6_seed4321"
    / "summary.json"
)
DEFAULT_BRAINUICL = (
    REPO_ROOT
    / "experiments"
    / "rttdp_brainuicl_runs"
    / "aligned_full49_bn_frozen_lr1e6_seed4321"
    / "clean"
    / "metrics.json"
)
DEFAULT_OUTPUT = DEFAULT_REGULARIZATION.parent


METHODS = (
    ("finetune", "Finetune", "#6B7280"),
    ("ewc", "EWC", "#2563EB"),
    ("online_ewc", "Online EWC", "#0F766E"),
    ("si", "SI", "#2F855A"),
    ("mas", "MAS", "#C2410C"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regularization-summary",
        type=Path,
        default=DEFAULT_REGULARIZATION,
    )
    parser.add_argument(
        "--brainuicl-metrics",
        type=Path,
        default=DEFAULT_BRAINUICL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(regularization_path: Path, brainuicl_path: Path) -> list[dict]:
    regularization = json.loads(regularization_path.read_text())
    brainuicl = json.loads(brainuicl_path.read_text())
    rows = []
    for key, label, color in METHODS:
        summary = regularization[key]
        rows.append(
            {
                "method": label,
                "color": color,
                "protocol": "No replay; all hard pseudo labels; frozen student BN stats",
                "final_old_acc": summary["final_old_acc"],
                "final_old_mf1": summary["final_old_mf1"],
                "mean_current_after_acc": summary["mean_current_after_acc"],
                "mean_current_after_mf1": summary["mean_current_after_mf1"],
                "final_seen_acc": summary["final_seen_acc"],
                "final_seen_mf1": summary["final_seen_mf1"],
                "bwt_acc": summary["bwt_acc"],
            }
        )

    summary = brainuicl["summary"]
    rows.append(
        {
            "method": "BrainUICL",
            "color": "#B91C1C",
            "protocol": "Replay; confidence threshold 0.9; frozen student BN stats",
            "final_old_acc": summary["final_old_acc"],
            "final_old_mf1": summary["final_old_mf1"],
            "mean_current_after_acc": summary["mean_current_after_acc"],
            "mean_current_after_mf1": summary["mean_current_after_mf1"],
            "final_seen_acc": summary["final_seen_acc"],
            "final_seen_mf1": summary["final_seen_mf1"],
            "bwt_acc": summary["bwt_acc"],
            "pseudo_sequence_coverage": summary["pseudo_sequence_coverage"],
            "final_buffer_sequences": summary["final_buffer_sequences"],
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "method",
        "protocol",
        "final_old_acc",
        "final_old_mf1",
        "mean_current_after_acc",
        "mean_current_after_mf1",
        "final_seen_acc",
        "final_seen_mf1",
        "bwt_acc",
        "pseudo_sequence_coverage",
        "final_buffer_sequences",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_panel(
    ax,
    rows: list[dict],
    key: str,
    title: str,
    *,
    chinese: bool,
) -> None:
    chinese_names = {
        "Finetune": "微调",
        "Online EWC": "在线 EWC",
        "BrainUICL": "BrainUICL（回放）",
    }
    y = np.arange(len(rows))
    ax.axhspan(len(rows) - 1.42, len(rows) - 0.58, color="#FEE2E2", alpha=0.55)
    for index, row in enumerate(rows):
        value = 100.0 * row[key]
        marker = "D" if row["method"] == "BrainUICL" else "o"
        size = 70 if marker == "D" else 62
        ax.hlines(index, 50.0, value, color=row["color"], linewidth=2.2, alpha=0.55)
        ax.scatter(
            value,
            index,
            s=size,
            marker=marker,
            color=row["color"],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            value + 0.45,
            index,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#111827",
        )

    labels = [
        chinese_names.get(row["method"], row["method"])
        if chinese
        else row["method"]
        for row in rows
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(45.0, 76.5)
    ax.set_xlabel("性能（%）" if chinese else "Performance (%)")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#111827", pad=9)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.tick_params(axis="x", colors="#4B5563")


def draw_bwt_panel(ax, rows: list[dict], *, chinese: bool) -> None:
    y = np.arange(len(rows))
    ax.axhspan(len(rows) - 1.42, len(rows) - 0.58, color="#FEE2E2", alpha=0.55)
    ax.axvline(0.0, color="#6B7280", linewidth=1.1)
    chinese_names = {
        "Finetune": "微调",
        "Online EWC": "在线 EWC",
        "BrainUICL": "BrainUICL（回放）",
    }
    for index, row in enumerate(rows):
        value = 100.0 * row["bwt_acc"]
        marker = "D" if row["method"] == "BrainUICL" else "o"
        ax.hlines(index, min(0.0, value), max(0.0, value), color=row["color"], linewidth=2.2, alpha=0.6)
        ax.scatter(value, index, s=70 if marker == "D" else 62, marker=marker, color=row["color"], edgecolor="white", linewidth=1.1, zorder=3)
        offset = 0.25 if value >= 0 else -0.25
        ax.text(value + offset, index, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9, color="#111827")
    labels = [chinese_names.get(row["method"], row["method"]) if chinese else row["method"] for row in rows]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-10.5, 2.5)
    ax.set_xlabel("BWT ACC（百分点）" if chinese else "BWT ACC (percentage points)")
    ax.set_title("后向迁移（BWT）" if chinese else "Backward Transfer (BWT)", fontsize=12, fontweight="bold", color="#111827", pad=9)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.tick_params(axis="x", colors="#4B5563")


def draw_protocol_panel(ax, rows: list[dict], *, chinese: bool) -> None:
    brainuicl = rows[-1]
    ax.axis("off")
    if chinese:
        title = "对齐协议与资源差异"
        lines = (
            "共同条件",
            "同一数据、split、个体顺序、checkpoint、seed、10+10 轮、学习率与冻结 BN",
            "",
            "五种无回放方法",
            "当前个体全部硬伪标签；历史 buffer：0",
            "",
            "BrainUICL",
            f"置信度阈值：0.9；新序列进入 replay：{100.0 * brainuicl['pseudo_sequence_coverage']:.2f}%",
            f"最终 replay buffer：{brainuicl['final_buffer_sequences']} 个序列（含初始历史训练数据）",
        )
    else:
        title = "Aligned Protocol and Resource Differences"
        lines = (
            "Common conditions",
            "Same data, split, order, checkpoint, seed, 10+10 epochs, learning rate, and frozen BN",
            "",
            "Five no-replay methods",
            "All current hard pseudo labels; historical buffer: 0",
            "",
            "BrainUICL",
            f"Confidence threshold: 0.9; new sequences entering replay: {100.0 * brainuicl['pseudo_sequence_coverage']:.2f}%",
            f"Final replay buffer: {brainuicl['final_buffer_sequences']} sequences (including initial history)",
        )
    ax.set_title(title, fontsize=12, fontweight="bold", color="#111827", pad=9)
    y = 0.88
    for line in lines:
        is_heading = line in {"共同条件", "五种无回放方法", "BrainUICL", "Common conditions", "Five no-replay methods"}
        ax.text(0.03, y, line, transform=ax.transAxes, fontsize=10 if is_heading else 9.2, fontweight="bold" if is_heading else "normal", color="#111827" if is_heading else "#4B5563", va="top")
        y -= 0.09 if line else 0.055


def plot(path_stem: Path, rows: list[dict], *, chinese: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK SC" if chinese else "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.6), constrained_layout=False)
    if chinese:
        panels = (
            ("final_old_acc", "最终旧个体 ACC"),
            ("final_old_mf1", "最终旧个体宏平均 F1"),
            ("final_seen_acc", "最终新个体 ACC"),
            ("final_seen_mf1", "最终新个体宏平均 F1"),
        )
    else:
        panels = (
            ("final_old_acc", "Final Old-Subject ACC"),
            ("final_old_mf1", "Final Old-Subject Macro-F1"),
            ("final_seen_acc", "Final Seen New-Subject ACC"),
            ("final_seen_mf1", "Final Seen New-Subject Macro-F1"),
        )
    for ax, (key, title) in zip(axes.flat[:4], panels):
        draw_panel(ax, rows, key, title, chinese=chinese)
    draw_bwt_panel(axes.flat[4], rows, chinese=chinese)
    draw_protocol_panel(axes.flat[5], rows, chinese=chinese)

    fig.suptitle(
        "无攻击 EEG 持续学习性能" if chinese else "Clean EEG Continual Learning Performance",
        fontsize=18,
        fontweight="bold",
        color="#111827",
        y=0.975,
    )
    fig.text(
        0.5,
        0.932,
        (
            "ISRUC Group 1｜随机种子 4321｜49 个新个体｜引导模型 10 轮 + 学生模型 10 轮"
            if chinese
            else "ISRUC Group 1 | seed 4321 | 49 new subjects | 10 guide + 10 student epochs"
        ),
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.018,
        (
            "单随机种子结果，无误差条。BrainUICL（菱形）使用回放和置信度筛选；其余五种方法不使用回放，硬伪标签覆盖率为 100%。性能横轴从 45% 开始。"
            if chinese
            else (
                "Single-seed results; no uncertainty bars. BrainUICL (diamond) uses replay and confidence-based "
                "selection; the five comparison methods use no replay and 100% hard pseudo-label coverage. "
                "Performance x-axes begin at 45%."
            )
        ),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.08, wspace=0.28, hspace=0.36)
    for extension in ("png", "pdf", "svg"):
        target = path_stem.with_suffix(f".{extension}")
        fig.savefig(target, dpi=260 if extension == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.regularization_summary, args.brainuicl_metrics)
    output_stem = args.output_dir / "aligned_regularization_vs_brainuicl_clean49"
    write_csv(output_stem.with_suffix(".csv"), rows)
    plot(output_stem, rows)
    plot(args.output_dir / "aligned_regularization_vs_brainuicl_clean49_zh", rows, chinese=True)
    print(output_stem)


if __name__ == "__main__":
    main()

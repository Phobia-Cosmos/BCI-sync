#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "experiments" / "regularization_cl_eeg_runs"
METHODS = ("finetune", "ewc", "online_ewc", "si", "mas")
DISPLAY_NAMES = {
    "finetune": "Finetune",
    "ewc": "EWC",
    "online_ewc": "Online EWC",
    "si": "SI",
    "mas": "MAS",
}
METRICS = (
    "final_old_acc",
    "final_old_mf1",
    "final_seen_acc",
    "final_seen_mf1",
    "bwt_acc",
    "bwt_mf1",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def signed_points(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def metric_cell(value: float, baseline: float | None = None) -> str:
    if baseline is None:
        return percent(value)
    return f"{percent(value)} ({signed_points(value - baseline)})"


def validate_run(
    root: Path,
    expected_mode: str,
    methods: tuple[str, ...],
) -> None:
    config = read_json(root / "config.json")
    if config.get("defense_mode", "none") != expected_mode:
        raise ValueError(f"Unexpected defense mode in {root}")
    for method in methods:
        payload = read_json(root / method / "metrics.json")
        if len(payload["tasks"]) != 49:
            raise ValueError(f"{root.name}/{method} does not contain 49 tasks")
        if len(payload["final"]["seen_subjects"]) != 49:
            raise ValueError(f"{root.name}/{method} lacks final 49-subject evaluation")


def comparison_rows(base: dict, t2t: dict, robust: dict) -> list[dict]:
    rows: list[dict] = []
    for method in METHODS:
        variants = (("none", base[method]), ("robust_feature", robust[method]))
        if method != "finetune":
            variants = (
                ("none", base[method]),
                ("t2t", t2t[method]),
                ("robust_feature", robust[method]),
            )
        for defense, summary in variants:
            row = {"method": method, "defense": defense}
            for metric in METRICS:
                row[metric] = float(summary[metric])
                row[f"delta_{metric}"] = (
                    0.0
                    if defense == "none"
                    else float(summary[metric] - base[method][metric])
                )
            row["t2t_valid_scores"] = summary.get("t2t_valid_scores")
            row["t2t_detected_pairs"] = summary.get("t2t_detected_pairs")
            row["t2t_rejected_updates"] = summary.get("t2t_rejected_updates")
            row["robust_mean_protected_fraction"] = summary.get(
                "robust_mean_protected_fraction"
            )
            row["robust_mean_lambda"] = summary.get("robust_mean_lambda")
            row["robust_mean_defense_loss"] = summary.get(
                "robust_mean_defense_loss"
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    base: dict,
    t2t: dict,
    robust: dict,
    base_root: Path,
    t2t_root: Path,
    robust_root: Path,
) -> str:
    lines = [
        "# ICML 2026 两种持续学习防御在 clean EEG full49 上的迁移结果\n",
        "\n",
        "本报告只回答两个问题：论文的 Task-to-Task（T2T）验证和 Robust Feature Defense 能否接入当前无 replay 的正则化 CL-EEG；在没有人为噪声或投毒时，加入防御相对原 clean 基线会产生什么代价。这里没有攻击，因此结果不能证明任何攻击鲁棒性。\n",
        "\n",
        "## 直接结论\n",
        "\n",
        "T2T 可以接入 EWC、Online EWC、SI 和 MAS，因为这些方法在任务间都有可解释为二次曲率的历史正则项。它不能原样接入 Finetune：Finetune 的历史正则矩阵为零，论文式 (6) 中 A 和 B 的共同投影子空间为空，无法产生有效检测分数。BrainUICL、SPR-EEG 和 PuriDivER-EEG 以 replay/memory 为核心，也不属于论文式 (2) 的无 replay 正则化框架，本轮没有把论文的理论保证外推到它们。\n",
        "\n",
        "Robust Feature Defense 可以作为额外二次正则项接到五种方法。当前实现将它放在 BrainUICL 睡眠分类器的最后线性层：对每个新个体，仅用输入 EEG 提取 128 维倒数第二层特征，估计特征二阶矩阵，在其特征方向上求论文式 (14)–(15) 的 protected set 和正则特征值。它不读取 target 真实标签、不使用 replay，也不做置信度过滤。\n",
        "\n",
        "clean full49 的核心结果是：两种防御都没有产生跨算法一致的免费增益。T2T 会把正常个体漂移误判为异常并回滚 6–12 个任务；Robust Feature 不拒绝任务，性能变化大多小于 0.35 个百分点，但额外损失远小于原训练损失。后续攻击实验必须分别按论文适用条件验证，不能仅凭本轮 clean 运行声称防御有效。\n",
        "\n",
        "## 对齐协议\n",
        "\n",
        "三组结果使用同一 ISRUC Group-I、seed 4321、49 个体顺序、BrainUICL backbone/checkpoint、10 个 CPC epoch、10 个增量 epoch、`ssl_lr=cl_lr=1e-6`、batch 16、冻结学生 BatchNorm running statistics、全部 guiding-model 硬伪标签和任务 49 后重评全部 49 个体。正则参数仍为 EWC 5000，Online EWC 6500/decay 1，SI 1500000/xi 0.000001，MAS 3000/decay 1。\n",
        "\n",
        f"- 无防御：`{base_root.relative_to(REPO_ROOT)}`\n",
        f"- T2T：`{t2t_root.relative_to(REPO_ROOT)}`\n",
        f"- Robust Feature：`{robust_root.relative_to(REPO_ROOT)}`\n",
        "\n",
        "## T2T clean 结果\n",
        "\n",
        "括号内是相对同一算法无防御基线的变化；`pp` 表示百分点。\n",
        "\n",
        "| 算法 | 旧个体 ACC | 旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for method in METHODS[1:]:
        current = t2t[method]
        baseline = base[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} + T2T | "
            f"{metric_cell(current['final_old_acc'], baseline['final_old_acc'])} | "
            f"{metric_cell(current['final_old_mf1'], baseline['final_old_mf1'])} | "
            f"{metric_cell(current['final_seen_acc'], baseline['final_seen_acc'])} | "
            f"{metric_cell(current['final_seen_mf1'], baseline['final_seen_mf1'])} | "
            f"{metric_cell(current['bwt_acc'], baseline['bwt_acc'])} |\n"
        )

    lines.extend(
        [
            "\n",
            "| 算法 | 有效检测分数 | 触发次数 | 被回滚的 clean 任务 | clean 任务回滚率 |\n",
            "|---|---:|---:|---:|---:|\n",
        ]
    )
    for method in METHODS[1:]:
        current = t2t[method]
        rejected = int(current["t2t_rejected_updates"])
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {int(current['t2t_valid_scores'])} | "
            f"{int(current['t2t_detected_pairs'])} | {rejected}/49 | "
            f"{100.0 * rejected / 49.0:.2f}% |\n"
        )

    lines.extend(
        [
            "\n",
            "论文实验采用对角 Hessian 近似和固定启发式阈值：当前分数至少是此前最多 5 个可用分数均值的 2.5 倍时触发。我们没有根据 EEG clean test 结果事后修改该阈值。结果表明 EEG 的合法跨个体 domain shift 足以产生检测峰值：T2T 在所有四种方法上都发生 clean 误报。Online EWC 的 ACC 和多种方法的部分 MF1 有小幅上升，是因为回滚偶然过滤了部分低质量伪标签任务，而不是因为本轮存在攻击；它同时拒绝了 24.49% 的正常任务，不能据此称为 clean 性能改进算法。\n",
            "\n",
            "## Robust Feature clean 结果\n",
            "\n",
            "| 算法 | 旧个体 ACC | 旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |\n",
            "|---|---:|---:|---:|---:|---:|\n",
        ]
    )
    for method in METHODS:
        current = robust[method]
        baseline = base[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} + Robust Feature | "
            f"{metric_cell(current['final_old_acc'], baseline['final_old_acc'])} | "
            f"{metric_cell(current['final_old_mf1'], baseline['final_old_mf1'])} | "
            f"{metric_cell(current['final_seen_acc'], baseline['final_seen_acc'])} | "
            f"{metric_cell(current['final_seen_mf1'], baseline['final_seen_mf1'])} | "
            f"{metric_cell(current['bwt_acc'], baseline['bwt_acc'])} |\n"
        )

    lines.extend(
        [
            "\n",
            "| 算法 | 平均 protected directions | 平均正则特征值 | 最后 epoch 平均防御损失 |\n",
            "|---|---:|---:|---:|\n",
        ]
    )
    for method in METHODS:
        current = robust[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | "
            f"{100.0 * current['robust_mean_protected_fraction']:.2f}% | "
            f"{current['robust_mean_lambda']:.4f} | "
            f"{current['robust_mean_defense_loss']:.3e} |\n"
        )

    lines.extend(
        [
            "\n",
            "默认预算按论文 CIFAR-100 设置做维度归一化：论文 `M=2000`、线性头 `768×100`，因此 EEG 使用每参数预算 `2000/(768×100)`，总预算随 `128×5` 分类器维度缩放。平均约 11% 的输出-特征方向进入 protected set。对 Finetune/EWC/Online EWC，最后 epoch 的额外防御损失约为 `1e-6`；对 SI/MAS 约为 `1e-8`，远小于伪标签交叉熵和已有正则项。因此本轮主要证明公式、状态和训练接口已经迁移，尚未证明该默认预算足以抵抗 EEG 投毒。\n",
            "\n",
            "## 实现与论文保证的边界\n",
            "\n",
            "1. T2T 使用每个任务的对角 empirical Fisher 近似损失 Hessian，并用现有正则器 penalty 对参数的二阶导数作为 H。触发后严格回滚到两次更新之前，同时恢复模型参数、BatchNorm buffers 和 EWC/SI/MAS 状态。EWC 的累计加权中心并不总等于论文中的上一模型点，因此这是论文非线性实验风格的近似迁移，不继承线性定理。\n",
            "2. Robust Feature 的论文闭式解要求线性平方损失和任务 Hessian 可同时对角化。EEG 使用非线性 backbone、交叉熵和持续变化的特征。当前实现只在最终线性分类器上使用当前特征协方差的特征基，并把风险协方差旋转到新基底；这是可运行的 hybrid approximation，不是定理 5.3 的严格实例。\n",
            "3. Robust Feature 作为额外二次项叠加在原 EWC/Online EWC/SI/MAS 上，以保留原算法身份。论文的 H 本身是系统要设计的唯一正则矩阵，因此“叠加版”不能直接称作论文最优 H。\n",
            "4. 所有 target 真实标签只用于训练后诊断 ACC/MF1，不参与 T2T 分数、特征协方差、protected set、伪标签或梯度更新。\n",
            "5. 当前只有一个 seed。表中小于约 0.5 pp 的变化不能视为统计显著结论。\n",
            "\n",
            "## 下一步攻击验证应如何分开\n",
            "\n",
            "T2T 只应先测试少量任务上的 shifted 强攻击，并在独立 clean calibration subjects 上固定阈值后再锁定攻击任务和预算。评价必须同时报告检测率、clean 误报率、回滚任务数和最终性能。\n",
            "\n",
            "Robust Feature 只应先测试频繁、幅度有界、条件零均值的 non-shifted 扰动。开始攻击实验前，应做预算 sweep 并检查防御梯度/原损失梯度比；如果额外项始终只有 `1e-8`–`1e-6`，即使 clean 指标不下降也不能期待可测防护。对 shifted proxy noise 使用 Robust Feature 不符合论文定理的适用条件。\n",
            "\n",
            "## 复现命令\n",
            "\n",
            "```bash\n",
            "/home/undefined/Disk/python-envs/brainuicl/bin/python \\\n",
            "  experiments/regularization_cl_eeg.py \\\n",
            "  --methods ewc,online_ewc,si,mas --defense-mode t2t \\\n",
            "  --ssl-epoch 10 --incremental-epoch 10 \\\n",
            "  --ssl-lr 1e-6 --cl-lr 1e-6 --freeze-bn-stats \\\n",
            "  --ewc-strength 5000 --online-ewc-strength 6500 \\\n",
            "  --si-strength 1500000 --si-xi 1e-6 --mas-strength 3000 \\\n",
            "  --no-save-checkpoints \\\n",
            "  --output-root experiments/regularization_cl_eeg_runs/icml2026_t2t_clean49_bn_frozen_e10_lr1e6_seed4321\n",
            "\n",
            "/home/undefined/Disk/python-envs/brainuicl/bin/python \\\n",
            "  experiments/regularization_cl_eeg.py \\\n",
            "  --methods finetune,ewc,online_ewc,si,mas \\\n",
            "  --defense-mode robust_feature \\\n",
            "  --ssl-epoch 10 --incremental-epoch 10 \\\n",
            "  --ssl-lr 1e-6 --cl-lr 1e-6 --freeze-bn-stats \\\n",
            "  --ewc-strength 5000 --online-ewc-strength 6500 \\\n",
            "  --si-strength 1500000 --si-xi 1e-6 --mas-strength 3000 \\\n",
            "  --no-save-checkpoints \\\n",
            "  --output-root experiments/regularization_cl_eeg_runs/icml2026_robust_feature_clean49_bn_frozen_e10_lr1e6_seed4321\n",
            "\n",
            "/home/undefined/Disk/python-envs/brainuicl/bin/python \\\n",
            "  experiments/summarize_icml2026_clean_defenses.py\n",
            "```\n",
        ]
    )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=RUN_ROOT / "clean49_bn_frozen_e10_lr1e6_seed4321",
    )
    parser.add_argument(
        "--t2t-root",
        type=Path,
        default=RUN_ROOT / "icml2026_t2t_clean49_bn_frozen_e10_lr1e6_seed4321",
    )
    parser.add_argument(
        "--robust-root",
        type=Path,
        default=(
            RUN_ROOT / "icml2026_robust_feature_clean49_bn_frozen_e10_lr1e6_seed4321"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "icml2026_cl_defenses" / "full49",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_run(args.baseline_root, "none", METHODS)
    validate_run(args.t2t_root, "t2t", METHODS[1:])
    validate_run(args.robust_root, "robust_feature", METHODS)
    base = read_json(args.baseline_root / "summary.json")
    t2t = read_json(args.t2t_root / "summary.json")
    robust = read_json(args.robust_root / "summary.json")
    rows = comparison_rows(base, t2t, robust)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "comparison.csv", rows)
    report = build_report(
        base,
        t2t,
        robust,
        args.baseline_root,
        args.t2t_root,
        args.robust_root,
    )
    (args.output_root / "SUMMARY_ZH.md").write_text(report)
    print(args.output_root / "SUMMARY_ZH.md")


if __name__ == "__main__":
    main()

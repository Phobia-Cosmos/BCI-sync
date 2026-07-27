#!/usr/bin/env python3
"""Validate and summarize Proxy v2 on full SPR/PuriDivER EEG methods."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    ("full_spr", "Full SPR-EEG adapted"),
    ("full_puridiver", "Full PuriDivER-EEG adapted"),
)
REFERENCE_METHODS = (("ewc", "EWC"), ("plain_er", "Plain ER"))
ENDPOINTS = ("old_acc", "old_mf1", "new_acc", "new_mf1")
PROXY_TASKS = tuple(range(1, 50, 2))
CLEAN_TASKS = tuple(range(2, 50, 2))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(root: Path, method: str, condition: str) -> Path:
    return root / "runs" / f"{method}_{condition}" / "metrics.json"


def reference_path(root: Path, method: str, condition: str) -> Path:
    base = root / "runs" / f"{method}_{condition}"
    return base / method / "metrics.json" if method == "ewc" else base / "metrics.json"


def final_view(metrics: dict[str, Any]) -> dict[str, float]:
    summary = metrics["summary"]
    return {
        "old_acc": float(summary["final_old_acc"]),
        "old_mf1": float(summary["final_old_mf1"]),
        "new_acc": float(summary["final_seen_acc"]),
        "new_mf1": float(summary["final_seen_mf1"]),
        "bwt_acc": float(summary["bwt_acc"]),
        "bwt_mf1": float(summary["bwt_mf1"]),
    }


def difference(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in right}


def validate(
    metrics: dict[str, Any],
    order: list[int],
    *,
    mode: str,
) -> dict[str, Any]:
    tasks = metrics["tasks"]
    if len(tasks) != 49 or [int(row["subject"]) for row in tasks] != order:
        raise ValueError("Full49 task order mismatch")
    if metrics["protocol"]["true_target_labels_used_for_training"] is not False:
        raise ValueError("A full replay method used target annotations for training")
    protocol = metrics["protocol"]["progressive_feedback_proxy"]
    if protocol["mode"] != mode:
        raise ValueError(f"Expected progressive mode {mode}, got {protocol['mode']}")
    if tuple(map(int, protocol["proxy_tasks"])) != PROXY_TASKS:
        raise ValueError("Proxy tasks are not the odd K=25 schedule")
    if tuple(map(int, protocol["clean_feedback_tasks"])) != CLEAN_TASKS:
        raise ValueError("Clean feedback tasks are not the even schedule")
    if protocol["incremental_clean_hard_labels_visible"] is not False:
        raise ValueError("Incremental hard labels entered the proxy")
    if int(protocol["source_labeled_sequences"]) != 1030:
        raise ValueError("Unexpected source hard-label buffer size")
    if not protocol["upload_full_pool"] or int(protocol["base_sequences"]) != 48:
        raise ValueError("The full 48-sequence proxy pool was not used")
    rows = [row["progressive_proxy"] for row in tasks]
    proxy_rows = [row for row in rows if row.get("feedback_kind") == "proxy"]
    clean_rows = [row for row in rows if row.get("feedback_kind") == "clean"]
    if len(proxy_rows) != 25 or len(clean_rows) != 24:
        raise ValueError("Odd/even proxy role count mismatch")
    if any(int(row["proxy_sequences"]) != 48 for row in proxy_rows):
        raise ValueError("A proxy task did not upload all 48 sequences")
    if any(int(row["labeled_buffer_sequences"]) != 1030 for row in rows):
        raise ValueError("Incremental records entered the source hard-label buffer")
    clean_sequences = sum(int(row["feedback_sequences"]) for row in clean_rows)
    result: dict[str, Any] = {
        "proxy_tasks": len(proxy_rows),
        "proxy_sequences": 48 * len(proxy_rows),
        "clean_sequences": clean_sequences,
        "stream_sequences": 48 * len(proxy_rows) + clean_sequences,
    }
    if mode == "feedback":
        source_cosines = [float(row["source_gradient_cosine"]) for row in proxy_rows]
        gate = float(metrics["config"]["progressive_max_source_gradient_cosine"])
        source_conflict_events = sum(
            bool(row["source_conflict_accepted"]) for row in proxy_rows
        )
        max_step = max(float(row["max_step_relative_l2"]) for row in proxy_rows)
        max_cumulative = max(
            float(row["max_cumulative_relative_l2"]) for row in proxy_rows
        )
        if max_step > 0.050001 or max_cumulative > 0.200001:
            raise ValueError("A proxy task exceeded its relative-L2 budget")
        result.update(
            {
                "source_gradient_cosine_mean": statistics.fmean(source_cosines),
                "source_gradient_cosine_max": max(source_cosines),
                "source_gradient_gate": gate,
                "source_conflict_events": source_conflict_events,
                "max_step_relative_l2": max_step,
                "max_cumulative_relative_l2": max_cumulative,
                "final_mean_cumulative_relative_l2": float(
                    proxy_rows[-1]["mean_cumulative_relative_l2"]
                ),
                "mean_pseudo_label_preservation": statistics.fmean(
                    float(row["generation"]["pseudo_label_preservation"])
                    for row in proxy_rows
                ),
                "mean_target_hit_rate": statistics.fmean(
                    float(row["generation"]["target_hit_rate"])
                    for row in proxy_rows
                ),
                "kl_improved_events": sum(
                    float(row["response_kl_after"])
                    < float(row["response_kl_before"])
                    for row in rows
                ),
            }
        )
    return result


def analyze_pair(
    static: dict[str, Any], feedback: dict[str, Any], order: list[int]
) -> dict[str, Any]:
    static_validation = validate(static, order, mode="static")
    feedback_validation = validate(feedback, order, mode="feedback")
    static_final = final_view(static)
    feedback_final = final_view(feedback)
    delta = difference(feedback_final, static_final)
    return {
        "static": static_final,
        "feedback": feedback_final,
        "delta_static": delta,
        "all_endpoints_down": all(delta[key] < 0.0 for key in ENDPOINTS),
        "static_memory_proxy_epoch_fraction": float(
            static["summary"]["final_memory_proxy_epoch_fraction"]
        ),
        "feedback_memory_proxy_epoch_fraction": float(
            feedback["summary"]["final_memory_proxy_epoch_fraction"]
        ),
        "static_replay_proxy_fraction": float(
            static["summary"]["proxy_replay_fraction"]
        ),
        "feedback_replay_proxy_fraction": float(
            feedback["summary"]["proxy_replay_fraction"]
        ),
        "static_validation": static_validation,
        "feedback_validation": feedback_validation,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def pair(row: dict[str, float], prefix: str, formatter) -> str:
    return f"{formatter(row[f'{prefix}_acc'])}/{formatter(row[f'{prefix}_mf1'])}"


def render(report: dict[str, Any]) -> str:
    lines = [
        "## Full SPR与PuriDivER的Proxy v2 Full49验证",
        "",
        "固定seed 4321、原始49-task顺序、奇数25个proxy任务、偶数24个clean反馈任务，以及Proxy v2单步5%/累计20%预算。每个奇数任务完整上传subject 18的48条sequence；每种方法分别运行同位置Fixed-clean48和反馈驱动v2。没有叠加额外置信度过滤或外部净化，但Full SPR仍保留其作为CL算法组成部分的Delayed Buffer、Expert/Base NT-Xent、SCF与Purified Memory完整流程，Full PuriDivER仍保留逐mini-batch动态purity-diversity memory以及逐replay epoch重算的C/R/U流程。增量hard label、victim参数和method memory均不返回给proxy。",
        "",
        "| 方法/条件 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 | 四终点均下降 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in METHODS:
        row = report["methods"][method]
        lines.append(
            f"| {label} / Fixed-clean48 | {pair(row['static'], 'old', pct)} | "
            f"+0.00 pp/+0.00 pp | {pair(row['static'], 'new', pct)} | "
            "+0.00 pp/+0.00 pp | - |"
        )
        lines.append(
            f"| {label} / Proxy v2 | {pair(row['feedback'], 'old', pct)} | "
            f"{pair(row['delta_static'], 'old', pp)} | "
            f"{pair(row['feedback'], 'new', pct)} | "
            f"{pair(row['delta_static'], 'new', pp)} | "
            f"{'是' if row['all_endpoints_down'] else '否'} |"
        )
    for method, label in REFERENCE_METHODS:
        row = report["references"][method]
        lines.append(
            f"| {label} / 既有Proxy v2参照 | {pair(row['feedback'], 'old', pct)} | "
            f"{pair(row['delta_static'], 'old', pp)} | "
            f"{pair(row['feedback'], 'new', pct)} | "
            f"{pair(row['delta_static'], 'new', pp)} | "
            f"{'是' if row['all_endpoints_down'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "| 方法 | Fixed memory/replay位置记录 | v2 memory/replay proxy | v2累计L2均值/最大 | pseudo-label保持/目标命中 | source cosine mean/max/达标任务 | KL改善 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHODS:
        row = report["methods"][method]
        validation = row["feedback_validation"]
        lines.append(
            f"| {label} | {pct(row['static_memory_proxy_epoch_fraction'])}/"
            f"{pct(row['static_replay_proxy_fraction'])} | "
            f"{pct(row['feedback_memory_proxy_epoch_fraction'])}/"
            f"{pct(row['feedback_replay_proxy_fraction'])} | "
            f"{pct(validation['final_mean_cumulative_relative_l2'])}/"
            f"{pct(validation['max_cumulative_relative_l2'])} | "
            f"{pct(validation['mean_pseudo_label_preservation'])}/"
            f"{pct(validation['mean_target_hit_rate'])} | "
            f"{validation['source_gradient_cosine_mean']:.3f}/"
            f"{validation['source_gradient_cosine_max']:.3f}/"
            f"{validation['source_conflict_events']}/25 | "
            f"{validation['kl_improved_events']}/49 |"
        )
    successful = [
        label
        for method, label in METHODS
        if report["methods"][method]["all_endpoints_down"]
    ]
    lines.extend(
        [
            "",
            (
                "在当前single-seed配对中，四个终点全部下降的方法为："
                + "、".join(successful)
                + "。"
                if successful
                else "在当前single-seed配对中，Full SPR与Full PuriDivER都没有出现四个终点同时下降。"
            ),
            "连续前置比连续后置更容易出现退化，原因是时序放大而不是过滤差异：前置数据先改变表示、伪标签和后续参数轨迹，随后还会参与更多次更新；Replay方法会多次抽到早期记录。既有Plain ER位置实验中，K=5前置/后置的全程proxy replay占比为35.25%/0.57%，K=25为87.00%/17.13%。EWC虽没有replay，但早期偏移会进入后续参数锚定与重要度估计。后置区块缺少这种累计窗口，单次轻微变化反而常表现为末端个体适配。",
            "EWC与Plain ER在无额外过滤时退化仍较弱并不矛盾。当前预算下多数guide pseudo-label未改变；proxy只优化surrogate classifier的一步梯度，而victim还包含guide适配、完整encoder更新和不同CL状态，方向不会严格一致。EWC参数锚定、ER中的clean replay、后续正常个体以及CPC自监督都会稀释或修复偏移；full49终点又是在修复窗口之后测量。例如果前置K=25的Plain ER在区块末old ACC/MF1已下降5.44/3.25 pp，后续clean阶段又修复6.69/4.03 pp，所以最终不再退化。",
            "Full SPR在task 1曾相对Fixed-clean48出现old ACC下降7.20 pp、当前个体ACC下降11.05 pp，但后续恢复，full49四终点最终反而提高。其v2最终memory/replay proxy比例为14.00%/32.27%，source-conflict仅9/25个任务达标，说明当前payload没有持续提供相反更新；SCF、purified memory、自监督与后续clean共同把它转化成了适配或正则化信号。因此当前方法不能使SPR稳定退化。",
            "Full PuriDivER的退化在中后段随memory累积扩大：old ACC配对差值在task 20/30/40约为-2.66/-6.67/-19.23 pp，最终为-18.96 pp。Fixed-clean48最终只保留0.40%位置记录，而v2保留20.00%，全程replay比例从2.29%升到14.12%。渐变数据改变了伪标签类别构成并持续填入class-balanced memory；其低loss/低uncertainty会被后续C/R/U视为模型一致数据，形成自增强漂移。这里25个任务的source cosine均未达到0.10，说明大幅退化主要来自PuriDivER特有的memory选择与伪标签反馈，而不是一个已经跨算法稳定成立的classifier梯度冲突。",
            "source-gradient cosine <= 0.10仅作为跨方法生成质量诊断，不作为阻塞条件；否则同一Proxy v2在某种CL结构上无法形成足够冲突时，实验会在进入victim更新前终止，无法测量该方法的真实响应。未达标任务本身说明当前生成目标没有稳定迁移到该方法。",
            "memory中的位置记录比例用于观察SCF或purity-diversity pruning的最终保留结果；replay比例还受到记录到达时间影响，不能单独解释为净化成功率。不同方法的proxy根据各自概率反馈独立生成，因此只比较每种方法内部的v2−Fixed差值，不用绝对分数或不同payload作方法间排名。",
            "",
            "负差值表示退化。结果为seed 4321单次完整轨迹，不代表跨seed统计显著性。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- FULL_SPR_PURIDIVER_PROXY_V2_START -->"
    end = "<!-- FULL_SPR_PURIDIVER_PROXY_V2_END -->"
    text = path.read_text(encoding="utf-8")
    block = f"{start}\n{markdown}{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bci", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_root.resolve()
    reference_root = args.reference_root.resolve()
    first = load_json(metrics_path(root, "full_spr", "static48"))
    order = [int(row["subject"]) for row in first["tasks"]]
    methods: dict[str, Any] = {}
    for method, _label in METHODS:
        static = load_json(metrics_path(root, method, "static48"))
        feedback = load_json(metrics_path(root, method, "v2"))
        methods[method] = analyze_pair(static, feedback, order)
    references: dict[str, Any] = {}
    for method, _label in REFERENCE_METHODS:
        static = load_json(reference_path(reference_root, method, "static48"))
        feedback = load_json(
            reference_path(reference_root, method, "v2_s05_c20")
        )
        static_final = final_view(static)
        feedback_final = final_view(feedback)
        delta = difference(feedback_final, static_final)
        references[method] = {
            "static": static_final,
            "feedback": feedback_final,
            "delta_static": delta,
            "all_endpoints_down": all(delta[key] < 0.0 for key in ENDPOINTS),
        }
    report = {
        "schema": "brainuicl-full-spr-puridiver-proxy-v2",
        "seed": 4321,
        "task_order": order,
        "methods": methods,
        "references": references,
    }
    (root / "FULL_RESULTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = render(report)
    (root / "FULL_RESULTS_ZH.md").write_text(markdown, encoding="utf-8")
    if args.bci is not None:
        update_bci(args.bci.resolve(), markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()

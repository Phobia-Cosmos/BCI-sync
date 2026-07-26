#!/usr/bin/env python3
"""Validate and summarize the full49 score-feedback proxy experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    ("ewc", "EWC"),
    ("plain_er", "Plain ER"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(root: Path, method: str, condition: str) -> Path:
    path = root / "runs" / f"{method}_{condition}"
    return path / method / "metrics.json" if method == "ewc" else path / "metrics.json"


def clean_metrics_path(root: Path, method: str) -> Path:
    if method == "ewc":
        return root / "runs" / "regularization" / "ewc" / "clean" / "ewc" / "metrics.json"
    return root / "runs" / "replay" / "clean" / "metrics.json"


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


def delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in right}


def validate_run(
    metrics: dict[str, Any],
    *,
    method: str,
    condition: str,
    expected_order: list[int],
) -> dict[str, Any]:
    tasks = metrics["tasks"]
    order = [int(row["subject"]) for row in tasks]
    if len(tasks) != 49 or order != expected_order:
        raise ValueError(f"{method}/{condition}: full49 order mismatch")
    protocol = metrics["protocol"]["progressive_feedback_proxy"]
    if protocol["proxy_tasks"] != list(range(1, 50, 2)):
        raise ValueError(f"{method}/{condition}: proxy schedule mismatch")
    if protocol["clean_feedback_tasks"] != list(range(2, 50, 2)):
        raise ValueError(f"{method}/{condition}: clean feedback schedule mismatch")
    if protocol["base_subject"] != 18 or protocol["base_sequences"] != 48:
        raise ValueError(f"{method}/{condition}: fixed source mismatch")
    if protocol["victim_parameters_visible"] is not False:
        raise ValueError(f"{method}/{condition}: invalid visibility declaration")
    rows = [row["progressive_proxy"] for row in tasks]
    proxy_rows = [row for row in rows if row and row.get("feedback_kind") == "proxy"]
    clean_rows = [row for row in rows if row and row.get("feedback_kind") == "clean"]
    if len(proxy_rows) != 25 or len(clean_rows) != 24:
        raise ValueError(f"{method}/{condition}: observed role count mismatch")
    if condition == "feedback":
        max_step = max(float(row["max_step_relative_l2"]) for row in proxy_rows)
        max_cumulative = max(
            float(row["max_cumulative_relative_l2"]) for row in proxy_rows
        )
        if max_step > 0.010001 or max_cumulative > 0.200001:
            raise ValueError(f"{method}: progressive budget violation")
    else:
        max_step = 0.0
        max_cumulative = 0.0
    uploaded_sequences = sum(
        int(row["current_after"]["n_epochs"]) // 20 for row in tasks
    )
    proxy_summary = metrics["final"]["progressive_proxy"]
    kl_pairs = [
        (float(row["response_kl_before"]), float(row["response_kl_after"]))
        for row in rows
        if row and row.get("response_kl_after") is not None
    ]
    return {
        "task_count": len(tasks),
        "proxy_tasks": len(proxy_rows),
        "clean_feedback_tasks": len(clean_rows),
        "proxy_sequences": sum(int(row["proxy_sequences"]) for row in proxy_rows),
        "uploaded_sequences": uploaded_sequences,
        "clean_feedback_sequences": uploaded_sequences
        - sum(int(row["proxy_sequences"]) for row in proxy_rows),
        "labeled_buffer_sequences": int(proxy_summary["labeled_buffer_sequences"]),
        "feedback_buffer_sequences": int(proxy_summary["feedback_buffer_sequences"]),
        "max_step_relative_l2": max_step,
        "max_cumulative_relative_l2": max_cumulative,
        "mean_response_kl_before": (
            statistics.fmean(pair[0] for pair in kl_pairs) if kl_pairs else None
        ),
        "mean_response_kl_after": (
            statistics.fmean(pair[1] for pair in kl_pairs) if kl_pairs else None
        ),
        "kl_improved_events": sum(after < before for before, after in kl_pairs),
        "kl_events": len(kl_pairs),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def pp_magnitude(value: float) -> str:
    return f"{100.0 * abs(value):.2f} pp"


def metric_pair(view: dict[str, float], prefix: str) -> str:
    return f"{pct(view[f'{prefix}_acc'])}/{pct(view[f'{prefix}_mf1'])}"


def delta_pair(view: dict[str, float], prefix: str) -> str:
    return f"{pp(view[f'{prefix}_acc'])}/{pp(view[f'{prefix}_mf1'])}"


def render(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "## 输出反馈驱动的渐进式 Proxy：EWC 与 Plain ER 完整验证",
        "",
        "固定 seed 4321 和原始 49-task subject 顺序；奇数 25 个位置由 pretrain-train subject 18 的固定 48-sequence pool 按当前 task 原数量替换，偶数 24 个位置仍上传原 clean subject。偶数 clean 上传不是额外 probe、不会增加任务或数据量，但其 EEG/EOG、hard label 和 victim 返回的五类概率可被本地 proxy 使用。victim 仅返回本用户上传后的概率，不暴露参数、梯度、optimizer、正则器或 replay memory；victim 学习率、训练轮数和算法逻辑均未修改。",
        "",
        f"Progressive 条件中，proxy 与 victim 从同一公开 pretrain checkpoint 初始化；本地模型用 pretrain train clean+hard label 保持 source 能力，并用此前偶数 clean 与奇数 proxy 上传的返回概率蒸馏。完整流仍为 {protocol['uploaded_sequences']} 条 sequence，其中 proxy 为 {protocol['proxy_sequences']} 条，原 clean 为 {protocol['clean_feedback_sequences']} 条；最终本地 labeled clean buffer 为 {protocol['labeled_buffer_sequences']} 条，即 {protocol['source_labeled_sequences']} 条 pretrain train clean 加 {protocol['clean_feedback_sequences']} 条后续 clean，概率反馈 buffer 为全部 {protocol['feedback_buffer_sequences']} 条 clean/proxy 上传。clean 部分不计入 proxy 预算。每个奇数版本从上一版固定 pool 继续更新，单步 relative-L2≤1%、累计≤20%、单步 L∞/std≤0.025、累计≤0.50，并约束输入方向锥和历史 classifier-gradient 方向。Static 条件在相同奇数位置重复使用未变化的 subject 18 pool，不使用反馈更新或渐进偏移。",
        "",
        "| 方法/条件 | old ACC/MF1 | Δold vs Clean | new ACC/MF1 | Δnew vs Clean |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in METHODS:
        method_row = report["methods"][method]
        clean = method_row["clean"]
        lines.append(
            f"| {label} / Clean | {metric_pair(clean, 'old')} | +0.00 pp/+0.00 pp | "
            f"{metric_pair(clean, 'new')} | +0.00 pp/+0.00 pp |"
        )
        for condition, condition_label in (("static", "Static"), ("feedback", "Progressive")):
            row = method_row[condition]
            lines.append(
                f"| {label} / {condition_label} | {metric_pair(row['final'], 'old')} | "
                f"{delta_pair(row['delta_clean'], 'old')} | {metric_pair(row['final'], 'new')} | "
                f"{delta_pair(row['delta_clean'], 'new')} |"
            )

    lines.extend(
        [
            "",
            "| 方法 | Progressive − Static old ACC/MF1 | Progressive − Static new ACC/MF1 | 反馈 KL 改善事件 | 最大单步/累计 relative-L2 | ER memory/replay proxy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in METHODS:
        row = report["methods"][method]
        diagnostics = row["feedback"]["validation"]
        er_text = "—"
        if method == "plain_er":
            er_text = (
                f"{pct(row['feedback']['memory_proxy_fraction'])}/"
                f"{pct(row['feedback']['replay_proxy_fraction'])}"
            )
        lines.append(
            f"| {label} | {delta_pair(row['feedback_minus_static'], 'old')} | "
            f"{delta_pair(row['feedback_minus_static'], 'new')} | "
            f"{diagnostics['kl_improved_events']}/{diagnostics['kl_events']} | "
            f"{pct(diagnostics['max_step_relative_l2'])}/{pct(diagnostics['max_cumulative_relative_l2'])} | "
            f"{er_text} |"
        )

    ewc = report["methods"]["ewc"]
    er = report["methods"]["plain_er"]
    lines.extend(
        [
            "",
            "判断标准分两层：`Progressive − Clean` 回答完整部署轨迹是否退化，`Progressive − Static` 才回答概率反馈、历史 buffer 与同向渐进是否比固定 subject 替换更有效。负差值表示退化。",
            "",
            f"EWC 上方法成立：Progressive 相对 Clean 的 old/new ACC 分别下降 {pp_magnitude(ewc['feedback']['delta_clean']['old_acc'])}/{pp_magnitude(ewc['feedback']['delta_clean']['new_acc'])}，相对 Static 又下降 {pp_magnitude(ewc['feedback_minus_static']['old_acc'])}/{pp_magnitude(ewc['feedback_minus_static']['new_acc'])}。这说明概率反馈、历史 buffer 和渐进同向更新在该 EWC 轨迹上确实增加了总退化。",
            "",
            f"Plain ER 上方法未成立：Progressive 相对 Clean 的 old/new ACC 仍提高 {pp_magnitude(er['feedback']['delta_clean']['old_acc'])}/{pp_magnitude(er['feedback']['delta_clean']['new_acc'])}；相对 Static 虽使 old ACC 下降 {pp_magnitude(er['feedback_minus_static']['old_acc'])}，new ACC 却提高 {pp_magnitude(er['feedback_minus_static']['new_acc'])}。因此不能把它认定为 Plain ER 上有效的退化方法，更不能声称当前方法已经跨两种 CL 算法成立。ER 中约一半 memory/replay 都来自 proxy，仍未形成最终退化，说明仅提高进入 replay 的覆盖率和 surrogate 跟踪精度并不足够。",
            "",
            f"概率跟踪机制本身运行正常：EWC 有 {ewc['feedback']['validation']['kl_improved_events']}/{ewc['feedback']['validation']['kl_events']} 次、Plain ER 有 {er['feedback']['validation']['kl_improved_events']}/{er['feedback']['validation']['kl_events']} 次反馈后 KL 下降。但 KL 下降只证明本地 proxy 更接近 victim 输出，不等价于生成的数据一定让 victim 最终退化；Plain ER 结果正是这一边界。",
            "",
            f"既有同 seed `k25_q50` frozen N→N 结果为：EWC Δold/new ACC {pp(report['prior']['k25_q50']['ewc']['old_acc'])}/{pp(report['prior']['k25_q50']['ewc']['new_acc'])}，Plain ER 为 {pp(report['prior']['k25_q50']['plain_er']['old_acc'])}/{pp(report['prior']['k25_q50']['plain_er']['new_acc'])}。`k25_q100` 为：EWC {pp(report['prior']['k25_q100']['ewc']['old_acc'])}/{pp(report['prior']['k25_q100']['ewc']['new_acc'])}，Plain ER {pp(report['prior']['k25_q100']['plain_er']['old_acc'])}/{pp(report['prior']['k25_q100']['plain_er']['new_acc'])}。这些旧条件的数据来源和生成轨迹不同，只作为幅度参照，不作为严格配对消融。",
            "",
            "所有结果均为 seed 4321 的单次严格方法内比较。动态反馈为 victim-specific，EWC 和 Plain ER 的 Progressive payload 不相同，不能用两者绝对分数作算法间优劣结论。",
        ]
    )
    return "\n".join(lines) + "\n"


def update_bci(path: Path, markdown: str) -> None:
    start = "<!-- OUTPUT_FEEDBACK_PROGRESSIVE_PROXY_START -->"
    end = "<!-- OUTPUT_FEEDBACK_PROGRESSIVE_PROXY_END -->"
    text = path.read_text(encoding="utf-8")
    block = f"{start}\n{markdown}{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def prior_rows(dose: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stage_name in ("k25_q50", "k25_q100"):
        stage = next(row for row in dose["stages"] if row["stage"] == stage_name)
        output[stage_name] = {
            method: stage["methods"][method]["delta"] for method, _label in METHODS
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--dose-results", type=Path, required=True)
    parser.add_argument("--bci", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_root.resolve()
    clean_root = args.clean_root.resolve()
    split = load_json(root / "runs" / "ewc_feedback" / "split.json")
    expected_order = [int(subject) for subject in split["new_order"]]
    if len(expected_order) != 49:
        raise ValueError("Expected a full49 split")

    methods: dict[str, Any] = {}
    for method, label in METHODS:
        clean_path = clean_metrics_path(clean_root, method)
        clean_metrics = load_json(clean_path)
        clean_order = [int(row["subject"]) for row in clean_metrics["tasks"]]
        if clean_order != expected_order:
            raise ValueError(f"{method}: clean baseline task order mismatch")
        clean = final_view(clean_metrics)
        row: dict[str, Any] = {
            "label": label,
            "clean_metrics": str(clean_path),
            "clean": clean,
        }
        for condition in ("static", "feedback"):
            path = metrics_path(root, method, condition)
            metrics = load_json(path)
            final = final_view(metrics)
            condition_row: dict[str, Any] = {
                "metrics": str(path),
                "final": final,
                "delta_clean": delta(final, clean),
                "validation": validate_run(
                    metrics,
                    method=method,
                    condition=condition,
                    expected_order=expected_order,
                ),
            }
            if method == "plain_er":
                condition_row["memory_proxy_fraction"] = float(
                    metrics["summary"]["final_memory_poisoned_fraction"]
                )
                condition_row["replay_proxy_fraction"] = float(
                    metrics["summary"]["replay_poisoned_fraction"]
                )
            row[condition] = condition_row
        row["feedback_minus_static"] = delta(
            row["feedback"]["final"], row["static"]["final"]
        )
        methods[method] = row

    ewc_validation = methods["ewc"]["feedback"]["validation"]
    er_validation = methods["plain_er"]["feedback"]["validation"]
    count_fields = (
        "proxy_sequences",
        "uploaded_sequences",
        "clean_feedback_sequences",
        "labeled_buffer_sequences",
        "feedback_buffer_sequences",
    )
    if any(ewc_validation[key] != er_validation[key] for key in count_fields):
        raise ValueError("EWC and Plain ER stream/buffer counts differ")

    report = {
        "schema": "brainuicl-progressive-score-feedback-v1",
        "protocol": {
            "seed": 4321,
            "tasks": 49,
            "proxy_tasks": list(range(1, 50, 2)),
            "clean_feedback_tasks": list(range(2, 50, 2)),
            "base_subject": 18,
            "base_sequences": 48,
            "extra_tasks": 0,
            "victim_parameters_visible": False,
            "uploaded_sequences": ewc_validation["uploaded_sequences"],
            "proxy_sequences": ewc_validation["proxy_sequences"],
            "clean_feedback_sequences": ewc_validation[
                "clean_feedback_sequences"
            ],
            "labeled_buffer_sequences": ewc_validation[
                "labeled_buffer_sequences"
            ],
            "source_labeled_sequences": ewc_validation[
                "labeled_buffer_sequences"
            ]
            - ewc_validation["clean_feedback_sequences"],
            "feedback_buffer_sequences": ewc_validation[
                "feedback_buffer_sequences"
            ],
        },
        "methods": methods,
        "prior": prior_rows(load_json(args.dose_results.resolve())),
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

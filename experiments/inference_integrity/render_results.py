"""Read completed experiment JSONs and render a descriptive Chinese report.

Inputs are read-only. Markdown goes to stdout or a new file outside the run
directory. Formal mode requires the frozen 256/32/32, seed-0 GPU protocol;
--diagnostic accepts a completed, audited gate and visibly labels its output.
This utility is outside the twelve frozen experiment implementation files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any


CONDITIONS = (
    "kv_cache", "serving_config", "model_mismatch", "token_selection",
    "computation", "input_mismatch", "serving_config_r2", "serving_config_r10",
)
LABELS = {
    "kv_cache": "KV",
    "serving_config": "Serving r6",
    "model_mismatch": "Model",
    "token_selection": "Token Selection",
    "computation": "Computation",
    "input_mismatch": "Input",
    "serving_config_r2": "Serving r2",
    "serving_config_r10": "Serving r10",
}
ORIGIN_LABELS = {
    "generation_live": "D",
    "final_recompute": "R",
    "model_recompute_live": "Llama 重算",
}
FORMAL_CONFIG = {
    "m": 256, "n": 32, "calibration_size": 32, "seeds": [0],
    "prefill_tokens": 32, "max_length": 128, "sample_repeats": 1000,
    "generation_device": "cuda:0", "verification_device": "cuda:0",
    "dtype": "float32", "attention_backend": "sdpa",
    "layers_per_block": 1, "activation_interval": 1,
    "serving_round_bits": 6, "serving_round_bits_sweep": [2, 10],
    "computation_relative_l2": 1e-2,
    "sample_rates": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return value


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def group_keys(config: dict[str, Any]) -> list[str]:
    return [
        f"{seed}/{condition}/{origin}"
        for condition in CONDITIONS
        for seed in config["seeds"]
        for origin in (("model_recompute_live",) if condition == "model_mismatch"
                       else ("generation_live", "final_recompute"))
    ]


def validate_inputs(
    root: Path, manifest: dict[str, Any], summary: dict[str, Any],
    audit: dict[str, Any], *, diagnostic: bool = False,
) -> None:
    """Reject partial, unaudited, mismatched, or nonformal inputs before output."""
    if summary.get("complete") is not True or summary.get("missing_count") != 0:
        raise ValueError("运行尚未完成；不能生成完成报告")
    if audit.get("passed") is not True or audit.get("issue_counts"):
        raise ValueError("completion_audit 未通过；不能生成完成报告")
    if Path(audit.get("run_root", "")).expanduser().resolve() != root.resolve():
        raise ValueError("completion_audit 的 run_root 与报告输入不符")
    audited_files = audit.get("audited_file_sha256", {})
    expected_files = {"manifest.json", "corpus.json", "thresholds.json", "summary.json"}
    if set(audited_files) != expected_files:
        raise ValueError("completion_audit 缺少输入文件哈希；请重新执行完成审计")
    for name in expected_files:
        if audited_files[name] != _digest(root / name):
            raise ValueError(f"输入文件在完成审计后发生变化：{name}")
    config = manifest["config"]
    config_hash = sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (manifest.get("config_sha256") != config_hash
            or summary.get("config_sha256") != config_hash
            or audit.get("observed", {}).get("config_sha256") != config_hash):
        raise ValueError("manifest、summary、completion_audit 配置哈希不一致")
    if not diagnostic:
        differences = [key for key, value in FORMAL_CONFIG.items() if config.get(key) != value]
        if differences:
            raise ValueError("不符合正式实验配置（可用 --diagnostic 仅生成诊断报告）：" + ", ".join(differences))
    try:
        audited_at = datetime.fromisoformat(audit["audited_at"])
        summarized_at = datetime.fromisoformat(summary["generated_at"])
        if audited_at < summarized_at:
            raise ValueError("completion_audit 早于 summary；请先重新执行完成审计")
    except (KeyError, TypeError) as exc:
        raise ValueError("缺少有效的摘要/审计时间戳") from exc
    expected = audit.get("expected", {})
    for key in ("m", "n", "calibration_size", "seeds", "prefill_tokens", "max_length",
                "sample_repeats", "generation_device", "verification_device"):
        if expected.get(key) != config.get(key):
            raise ValueError(f"完成审计采用了不同的期望配置：{key}")
    keys = group_keys(config)
    if set(summary.get("groups", {})) != set(keys):
        raise ValueError("summary 的证据组不完整或存在额外组")
    observed = audit["observed"]
    expected_cases = len(config["seeds"]) * len(CONDITIONS) * config["m"]
    for key, value in {
        "calibration_controls": config["calibration_size"],
        "expected_cases": expected_cases, "observed_cases": expected_cases,
        "expected_evidence_groups": len(keys), "observed_summary_groups": len(keys),
        "expected_origin_results": len(keys) * config["m"],
        "observed_origin_results": len(keys) * config["m"],
    }.items():
        if observed.get(key) != value:
            raise ValueError(f"完成审计的总数不符：{key}")
    if not diagnostic:
        stage_audits = observed.get("stage_audits", {})
        stage_keys = {f"{seed}/{condition}" for seed in config["seeds"] for condition in CONDITIONS}
        if not isinstance(stage_audits, dict) or set(stage_audits) != stage_keys:
            raise ValueError("完成审计缺少全部条件的块结果复核")
        for stage_key, stage in stage_audits.items():
            seed, condition = stage_key.split("/", 1)
            origins = ("model_recompute_live",) if condition == "model_mismatch" else (
                "generation_live", "final_recompute")
            if (stage.get("passed") is not True or stage.get("root_results") != config["m"]
                    or stage.get("origin_results") != config["m"] * len(origins)
                    or stage.get("block_result_files") != 2 * config["m"] * len(origins)
                    or set(stage.get("groups", {})) != set(origins)):
                raise ValueError(f"条件块结果复核不完整：{stage_key}")
            for origin in origins:
                for mode in ("numeric", "strict_hash"):
                    derived = stage["groups"][origin][mode]
                    saved = summary["groups"][f"{stage_key}/{origin}"][mode]["counts"]
                    if any(derived.get(name) != saved.get(name) for name in
                           ("TP", "FN", "FP", "TN", "ERROR")):
                        raise ValueError(f"块结果与摘要统计不一致：{stage_key}/{origin}/{mode}")
    for key in keys:
        group = summary["groups"][key]
        if group.get("observed_cases") != config["m"] or group.get("expected_cases") != config["m"]:
            raise ValueError(f"证据组请求数量不完整：{key}")
        for mode in ("numeric", "strict_hash"):
            metric = group[mode]
            if metric.get("planned_sequences") != config["m"]:
                raise ValueError(f"计划序列数量不符：{key}/{mode}")
            counts = metric["counts"]
            total = sum(counts.get(name, 0) for name in
                        ("TP", "FN", "FP", "TN", "ERROR", "UNSUPPORTED", "INJECTION_FAILED"))
            if total != config["m"]:
                raise ValueError(f"四格/错误/注入失败计数之和不符：{key}/{mode}")
        sampling = group["sampling"]
        if set(sampling) != {str(rate) for rate in config["sample_rates"]}:
            raise ValueError(f"采样率不完整：{key}")
        for rate, row in sampling.items():
            if row.get("repeat_count_per_sequence") != config["sample_repeats"]:
                raise ValueError(f"采样重复数不符：{key}/{rate}")


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("报告包含非有限统计量")
    return f"{value * 100:.2f}%"


def _interval(value: Any) -> str:
    return "—" if value is None else f"[{_pct(value[0])}, {_pct(value[1])}]"


def _f1_from_counts(counts: dict[str, Any]) -> float | None:
    true_positives = counts.get("TP", 0)
    denominator = 2 * true_positives + counts.get("FP", 0) + counts.get("FN", 0)
    return 2 * true_positives / denominator if denominator else None


def _label(key: str) -> str:
    seed, condition, origin = key.split("/")
    return f"{LABELS[condition]} / {ORIGIN_LABELS[origin]}（seed {seed}）"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def render_report(
    root: Path, manifest: dict[str, Any], summary: dict[str, Any],
    audit: dict[str, Any], audit_path: Path, *, diagnostic: bool = False,
) -> str:
    validate_inputs(root, manifest, summary, audit, diagnostic=diagnostic)
    config = manifest["config"]
    environment = manifest["environment"]
    keys = group_keys(config)
    groups = summary["groups"]
    lines = ["# " + ("诊断样例：AFTUNE 推理完整性报告格式验证" if diagnostic
                       else "AFTUNE 推理完整性实验结果"), ""]
    if diagnostic:
        lines += ["> 本报告仅展示已完成诊断 gate 的实际数据，用于验证报告格式；不是正式实验结论。", ""]
    lines += [
        f"本报告读取 `{root}` 的已完成摘要和通过的完成审计。"
        f"共 {audit['observed']['observed_cases']} 个条件内请求、{len(keys)} 个证据组、"
        f"{audit['observed']['observed_origin_results']} 条序列证据；Model 只计一次。",
        f"摘要时间：`{summary['generated_at']}`；完成审计时间：`{audit['audited_at']}`。", "",
        "## 1. 设置与判定口径", "",
        _table(["项目", "实际配置"], [
            ["工作负载", f"M={config['m']}；N={config['n']}；P={config['prefill_tokens']}；max_length={config['max_length']}；seed={config['seeds']}"],
            ["独立正常校准", f"{config['calibration_size']} 个请求，D/R/Model 分别冻结最大序列分数阈值"],
            ["声明/替代模型", f"{config['declared_model_path']} / {config['substitute_model_path']}"],
            ["数据集", f"{config['dataset_name']}；revision={config['dataset_revision']}"],
            ["精度与设备", f"{config['dtype']}；生成={config['generation_device']}；审计={config['verification_device']}；{environment.get('gpu_name')}"],
            ["执行栈", f"PyTorch {environment.get('torch')}；Transformers {environment.get('transformers')}；CUDA {environment.get('cuda_runtime')}；attention={config['attention_backend']}"],
            ["块与注入", f"B_L={config['layers_per_block']}；I_A={config['activation_interval']}；Computation 相对 L2={config['computation_relative_l2']}；Serving r2/r6/r10"],
            ["抽样", f"{', '.join(_pct(rate) for rate in config['sample_rates'])}；每条证据每个比例重复 {config['sample_repeats']} 次"],
        ]), "",
        "D 是生成时每次选词前的实际 forward 记录；R 是生成完成后无注入的声明模型重算记录。"
        "Model 仅展示源模型文本经 Llama 重分词后的实时 Llama 重算。正常 Llama 输入为 BOS 加 31 个内容 tokens，屏蔽 EOS、贪心生成 96 tokens；"
        "跨 tokenizer 的 Model 源输入和桥接后长度可变，不声称所有 Model 序列均满足 P=32/总长128。", "",
        "数值模式以最大块相对 L2 分数严格超过阈值，或明确的输入/磁盘完整性错误判为检出。"
        "严格哈希模式另要求计算输出位级一致。注入失败不入四格；执行错误或不支持结果单列。"
        "可判定覆盖率使用成功注入阳性、正常阴性和注入状态未知请求为分母；未知状态也包含在 ERROR 中，不能重复相加。", "",
        _table(["证据来源", "校准阈值 τ"], [
            [ORIGIN_LABELS[origin], f"{value:.9g}"]
            for origin, value in audit["observed"]["thresholds_recomputed"].items()
        ]), "",
        "## 2. 全量检查结果", "",
        "以下均为摘要中的实际观测计数；D/R 是同一请求的配对证据，不合并为独立样本。TPR/FPR 区间为请求级 95% Wilson 区间。", "",
    ]
    for mode, title in (("numeric", "数值主模式"), ("strict_hash", "严格哈希模式")):
        rows = []
        for key in keys:
            metric = groups[key][mode]
            counts = metric["counts"]
            rows.append([
                _label(key), *(counts.get(name, 0) for name in ("TP", "FN", "FP", "TN")),
                _pct(metric["decidable_coverage"]),
                f"{_pct(metric['tpr'])} {_interval(metric['tpr_wilson_95'])}",
                f"{_pct(metric['fpr'])} {_interval(metric['fpr_wilson_95'])}",
            ])
        lines += [f"### {title}", "", _table(
            ["证据组", "TP", "FN", "FP", "TN", "可判定覆盖率", "TPR [95% CI]", "FPR [95% CI]"], rows), ""]
    lines += ["### 错误、注入成功与部署召回", "",
              "E/U 为 ERROR/UNSUPPORTED；未知注入状态是 E 的子集。部署召回为 TP 除以全部成功注入阳性，包含未能判定的请求。", "",
              "下表 F1 由四格计数按 2TP/(2TP+FP+FN) 计算，不改写冻结摘要。全漏检时 F1 为 0；"
              "分母为零时显示“—”。precision 仍沿用摘要值，无预测阳性时显示“—”。", "",
              _table(["证据组", "数值 E/U", "严格 E/U", "注入失败/未知", "注入成功率", "数值部署召回", "数值 precision / F1（四格计算）"], [
                  [_label(key),
                   f"{groups[key]['numeric']['counts'].get('ERROR', 0)}/{groups[key]['numeric']['counts'].get('UNSUPPORTED', 0)}",
                   f"{groups[key]['strict_hash']['counts'].get('ERROR', 0)}/{groups[key]['strict_hash']['counts'].get('UNSUPPORTED', 0)}",
                   f"{groups[key]['numeric']['counts'].get('INJECTION_FAILED', 0)}/{groups[key]['numeric'].get('injection_status_unknown', 0)}",
                   _pct(groups[key]['numeric']['injection_success_rate']),
                   _pct(groups[key]['numeric']['deployment_recall']),
                   f"{_pct(groups[key]['numeric']['precision'])} / {_pct(_f1_from_counts(groups[key]['numeric']['counts']))}"]
                  for key in keys
              ]), "", "## 3. 冻结块结果上的抽样", "",
              "每次无放回抽取 max(1,ceil(pB)) 个块。D 有 1,728 个 step/block 单元，R 与 Model 有 18 个块；"
              "因此 18 块下 1% 和 5% 都抽一个块。重复抽样复用全量检查结果，不是新的独立请求或实际 TEE 性能测量。", ""]
    for field, title in (
        ("tpr", "采样 TPR（模拟观测）"),
        ("fpr", "采样 FPR（模拟观测）"),
        ("mean_exact_positive_hit_probability", "阳性理论命中概率的请求均值"),
    ):
        lines += [f"### {title}", "", _table(
            ["证据组", *(_pct(rate) for rate in config["sample_rates"])],
            [[_label(key), *(_pct(groups[key]["sampling"][str(rate)][field])
                            for rate in config["sample_rates"])] for key in keys]), ""]
    lines += ["理论值按 1−C(B−k,s)/C(B,s) 计算，k 是固定检查结果中的拒绝块数；"
              "它不等于相对正常执行数值发生变化的块数。该值描述给定证据的抽样命中，不替代真实生成过程完整性的保证。", "",
              "<details>", "<summary>采样原始计数、覆盖率和错误</summary>", "",
              _table(["证据组", "采样率", "TP", "FN", "FP", "TN", "错误阳性/阴性", "注入失败/未知", "可判定覆盖率"], [
                  [_label(key), _pct(rate),
                   *(groups[key]["sampling"][str(rate)]["counts"].get(name, 0) for name in ("TP", "FN", "FP", "TN")),
                   f"{groups[key]['sampling'][str(rate)]['counts'].get('ERROR_POSITIVE', 0)}/{groups[key]['sampling'][str(rate)]['counts'].get('ERROR_NEGATIVE', 0)}",
                   f"{groups[key]['sampling'][str(rate)]['counts'].get('INJECTION_FAILED', 0)}/{groups[key]['sampling'][str(rate)]['counts'].get('INJECTION_STATUS_UNKNOWN', 0)}",
                   _pct(groups[key]["sampling"][str(rate)]["decidable_coverage"])]
                  for key in keys for rate in config["sample_rates"]
              ]), "", "这些计数以“请求 × 抽样重复”为单位；不据此扩大量化 TPR/FPR 置信区间的独立样本量。", "", "</details>", "",
              "## 4. 解释边界", "",
              "- 原生检查覆盖 embedding、16 层和 norm，不检查 lm_head、选词规则、独立客户端原始输入或设备身份。Model 的通过表示所提交 Llama forward 通过，不能由此推断源生成模型也是 Llama。",
              "- KV 是 use_cache=False 的完整前缀 context 替换模拟，不包含物理 DynamicCache 生命周期。Serving 是一次切换后持续的 FP32 舍入代理，未实际验证设备迁移或 thinking budget。",
              "- 本实验在 GPU 上执行参考重算，没有 TEE 远程认证或被攻陷宿主机隔离测试。单 seed、小模型、短输入和固定扰动强度限制结果的外推。",
              "- Model 保留同一请求文本后独立分词，源序列和桥接序列长度可变；这属于蓝图固定 token 长度的明确例外。数值阈值只由独立正常校准得到，不保证其他设备或分布上的零误报。",
              "- 完成审计确认计划输出与统计的一致性，不是独立安全证明。本报告不把预期检测能力写作实际结果，也不把诊断 gate 指标替代正式实验。", "",
              "## 5. 输入与复现标识", "",
              f"配置 SHA-256：`{manifest['config_sha256']}`。",
              f"语料 SHA-256：`{manifest['corpus_sha256']}`。",
              f"派生数据 SHA-256：`{manifest['dataset']['sha256']}`。", "",
              _table(["输入文件", "报告读取内容的 SHA-256"], [
                  [str(path), _digest(path)]
                  for path in (root / "manifest.json", root / "summary.json", audit_path)
              ]), "",
              "模型权重、tokenizer、12 个冻结核心文件的哈希在 manifest 中；参考快照哈希在 reference/ready.json。"
              "逐请求分数、判定与抽样种子保存在运行目录。完整方法见实验方案.md。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed run directory (read-only)")
    parser.add_argument("--audit", type=Path, help="Audit JSON; defaults to ROOT/completion_audit.json")
    parser.add_argument("--output", type=Path, help="New Markdown file outside ROOT; defaults to stdout")
    parser.add_argument("--diagnostic", action="store_true", help="Visibly labeled nonformal gate report")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    audit_path = (args.audit or root / "completion_audit.json").expanduser().resolve()
    try:
        report = render_report(root, _read_json(root / "manifest.json"),
                               _read_json(root / "summary.json"), _read_json(audit_path),
                               audit_path, diagnostic=args.diagnostic)
        if args.output is None:
            print(report, end="")
        else:
            destination = args.output.expanduser().resolve()
            if destination == root or root in destination.parents:
                raise ValueError("报告输出必须位于实验运行目录之外")
            if not destination.parent.is_dir():
                raise ValueError("报告输出的父目录不存在")
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(report)
            print(destination)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"拒绝生成报告：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

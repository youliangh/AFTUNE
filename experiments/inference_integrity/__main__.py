"""Run the inference-integrity evaluation with explicit resumable stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runner import ALL_CONDITIONS, calibrate, configure_determinism, prepare_reference, run_cases, summarize


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / ".cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3"
DEFAULT_CONFIG = Path(__file__).with_name("formal-seed0-cal32-v3.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Frozen JSON configuration; defaults to the single-seed, 32-control protocol")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Isolated run directory")
    subcommands = parser.add_subparsers(dest="stage", required=True)
    subcommands.add_parser("fetch", help="Fetch/check pinned local MentalChat16K data")
    subcommands.add_parser("prepare", help="Freeze corpus and declared-model reference")
    calibration = subcommands.add_parser("calibrate", help="Run disjoint clean calibration")
    calibration.add_argument("--limit", type=int, help="Partial diagnostic calibration; no threshold is frozen")
    formal = subcommands.add_parser("run", help="Run scheduled evaluation cases")
    formal.add_argument("--condition", action="append", choices=ALL_CONDITIONS)
    formal.add_argument("--seed", type=int, action="append")
    formal.add_argument("--limit", type=int, help="Run first K requests per selected seed, for diagnostics")
    formal.add_argument("--retry-errors", action="store_true", help="Archive failed cases and rerun them")
    report = subcommands.add_parser("summarize", help="Create numeric/hash confusion matrices")
    report.add_argument("--partial", action="store_true", help="Allow an explicitly incomplete diagnostic summary")
    complete = subcommands.add_parser("all", help="Prepare, calibrate, run all conditions, summarize")
    complete.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = args.output.expanduser().resolve()
    if args.stage != "fetch":
        configure_determinism(0)
        import torch
        torch.set_num_threads(config.cpu_threads)
    if args.stage == "fetch":
        from .fetch_data import fetch_dataset

        print(fetch_dataset())
    elif args.stage == "prepare":
        prepare_reference(config, root)
        print(root / "manifest.json")
    elif args.stage == "calibrate":
        thresholds = calibrate(config, root, limit=args.limit)
        print(json.dumps({"thresholds": thresholds, "partial": not bool(thresholds)}, indent=2))
    elif args.stage == "run":
        run_cases(
            config, root,
            conditions=tuple(args.condition) if args.condition else None,
            seeds=tuple(args.seed) if args.seed else None,
            limit=args.limit,
            retry_errors=args.retry_errors,
        )
        print(root / "progress.jsonl")
    elif args.stage == "summarize":
        result = summarize(config, root, allow_partial=args.partial)
        print(json.dumps({"complete": result["complete"], "missing_count": result["missing_count"]}, indent=2))
    elif args.stage == "all":
        prepare_reference(config, root)
        calibrate(config, root)
        run_cases(config, root, retry_errors=args.retry_errors)
        result = summarize(config, root)
        print(json.dumps({"complete": result["complete"], "summary": str(root / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

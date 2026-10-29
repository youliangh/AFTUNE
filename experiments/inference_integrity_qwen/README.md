# Qwen3.5-0.8B inference integrity (v4)

This isolated package implements [实验方案.md](实验方案.md). The declared and trusted model is Qwen3.5-0.8B; the Model Mismatch source is Llama-3.2-1B-Instruct. The v3 package and results remain unchanged.

The formal run completed on 2026-10-28 at 02:33:50 UTC: 32 calibration requests, 416 formal source tasks, 640 unique evidence records and 1,536 condition references. All six sets of 32 injections succeeded; execution/verification errors, failed injections and unknown injection status were zero. Full sampling reconstruction and retained-tensor audit passed. See [实验结果.md](实验结果.md) for both decision modes, actual detection rates and the paired v3 comparison.

The [delivery bundle](../artifacts/README.md) contains both completed experiments, compressed score/truth evidence, per-file checksums and a model-free offline verifier that reconstructs all sampling.

The output directory is `../.cache/experiment-runs/aftune-inference-integrity-qwen-seed0-cal32-v4` relative to `AFTUNE`. It contains `summary.json`, `completion_audit.json`, `paired_comparison.json` (all 288 paired texts and token IDs), and `post_audit_review.json`. The report supplement and its fingerprint are archived separately under `reporting/`; frozen runtime code and thresholds were not changed during reporting.

Run from `AFTUNE` with the pinned `.venv`. Generation and recording use CUDA 0; independent reference replay uses CPU, 8 threads, FP32. Full attention uses SDPA and linear attention uses the frozen pure Torch fallback. No chat template or artificial BOS is used for the 32-token Qwen prompt.

```bash
.venv/bin/python -m unittest discover -s experiments/inference_integrity_qwen/tests -v
.venv/bin/python -m experiments.inference_integrity_qwen.preflight --output ../.cache/experiment-runs/aftune-inference-integrity-qwen-seed0-cal32-v4-preflight-2
.venv/bin/python -m experiments.inference_integrity_qwen.supervise --output ../.cache/experiment-runs/aftune-inference-integrity-qwen-seed0-cal32-v4 --preflight ../.cache/experiment-runs/aftune-inference-integrity-qwen-seed0-cal32-v4-preflight-2 --interval 60
```

The supervisor requires the actual-checkpoint preflight to pass with exactly the same configuration and runtime/audit code hashes. `--preflight PATH` selects another diagnostic directory after an archived failed attempt. Freeze all code before preflight; a changed fingerprint requires a new diagnostic/run directory. The formal supervisor prepares, calibrates all 32 requests, runs 224 shared controls then six groups of 32 positives, summarizes, reconstructs all sampling, audits and renders a Chinese result report.

The first diagnostic directory is preserved with `failure_diagnosis.json`: Transformers retained the nested text configuration's BF16 default despite the original top-level dtype argument. Loading now explicitly fixes the text config and parameters to FP32, and recording rejects non-FP32 activations. The second diagnostic uses this correction; failed diagnostic scores never enter formal calibration.

The 24 decoder layers plus embedding and norm form 13 two-layer blocks. Sparse base outputs are 0,4,8,12; full verification reveals the other nine original live outputs after commitment. D contains 1,248 checks and samples 100 per repeat; Model contains 13 and samples 2. Each unique evidence record has 1,000 repeats. The 416 source generations produce 640 unique sequence-evidence results and 1,536 condition references. Shared controls reuse generation, evidence, verdict and sampling, not just input indices.

Rows, labels, decode/switch steps, input swap positions and requested KV slot are copied from the completed v3 run. Only whitelisted scheduling fields affect the new experiment. Qwen inputs and request text are rebuilt from the same raw data rows. New layer and fault seeds depend on the v4 configuration. KV layers are 3,7,11,15,19,23; Computation covers all 24 layers; Serving r6 covers every layer after its single switch. Model uses Llama source text, retokenized and actually recomputed by Qwen, with actual lengths preserved.

Only tokenizer EOS and unassigned vocab IDs are masked for ordinary greedy generation. Qwen tokenizer EOS is 248046; its config EOS/PAD 248044 remains eligible for ordinary greedy. A replacement token also excludes every special ID. These policies and tokenizer/backend identities are recorded in the manifest.

Resource checks require at least 768 MiB free GPU memory and 5 GiB free disk. GPU unavailability/OOM or three consecutive identical execution errors stop further work. Resume the same supervisor command; `--retry-errors` archives failed source versions before retrying. Interrupted proofs are archived, preserving a saved substitute source. All views resolve to the current shared version. Indices 0,64,128,192 retain complete tensors; other records are purged only after full per-source verification. Calibration, donor, retries and Model proof costs are distinguished from formal source-generation counts.

`runner_state.json`, `progress.jsonl`, `monitor.latest.json` and `monitor.jsonl` describe progress and resource snapshots. A read-only outcome monitor is available:

```bash
.venv/bin/python -m experiments.inference_integrity_qwen.live_outcome_monitor --root ../.cache/experiment-runs/aftune-inference-integrity-qwen-seed0-cal32-v4 --watch PID --interval 60
```

Final `summary.json` includes both decision modes, error/failed/unknown injection counts, Wilson intervals, coverage, deployment recall, source mappings and costs counted once per task. `completion_audit.json` verifies calibration, source identities, paired schedule, every block coordinate, retained evidence, sparse preparation, exact sampling reconstruction and summaries, including JSON-persisted Wilson intervals. The final report compares v3 and v4 while retaining tokenization, bridge-length, eligible-layer and sampling-cost differences. Numerical rejection does not establish source-model identity, client-input binding or token-selection correctness; no TEE attestation or new detector is added.

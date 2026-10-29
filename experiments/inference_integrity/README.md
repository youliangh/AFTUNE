# Inference integrity evaluation (v3)

This run evaluates six controlled violations with GPU generation and CPU verification. It uses the pinned local Llama-3.2-1B-Instruct and Qwen3.5-0.8B checkpoints, seed 0, FP32, SDPA, one BOS plus 31 prompt tokens, and 96 generated tokens. Full protocol: [实验方案.md](实验方案.md).

The formal v3 run is complete: 416 source tasks, 640 unique evidence records and 1,536 view references; all 192 injections succeeded and full sampling reconstruction passed. See [实验结果.md](实验结果.md) for the final report and the [delivery bundle](../artifacts/README.md) for both model directions, exported score evidence and offline verification. The earlier [阶段性实验报告.md](阶段性实验报告.md) is a historical snapshot of a superseded protocol.

## Frozen workload and evidence

- `B_L=2`, `I_A=4`, native inference `B_S=1`: the 18 tracked modules form nine layer blocks, including embedding and final norm. `lm_head`, token-selection correctness, original client identity, physical cache history, and routing identity are not newly checked.
- All conditions share 224 clean Llama generations. Each clean generation has D evidence and a text-bridged Model recomputation; the five non-Model conditions reference the same D result, including the same sampling seed and indices. Model uses the same source response with its own bridge and proof stage.
- Each condition runs its own 32 positives at the same frozen input indices. Six conditions: `kv_cache`, `serving_config` (r6 only), `model_mismatch`, `token_selection`, `computation`, and `input_mismatch`.
- D (`generation_live`) contains 96 full-prefix forwards with 9 layer blocks each. Model (`model_recompute_live`) contains 9 blocks over a GPU Llama recomputation of retokenized response text. The Model verifier receives only that origin, not D or the oracle. There is no R origin.
- 416 target generation tasks, 640 unique sequence-evidence results, and 1,536 condition-view references. Calibration, donor computation, retries, and Model proof recomputation are additional work. References and paired origins are not independent samples.
- The same 32 disjoint clean calibration inputs determine separate D and Model numerical thresholds. Old GPU-verification thresholds and old run artifacts are not reused.

## Sparse activations

The ordinary Recorder saves only block 0, 4, and 8 activation outputs, while hashing every block boundary. The adapter temporarily retains the missing original live outputs in memory, including actual injected values. After saving and freezing the sparse base files, activation preparation reveals those original tensors and checks their already committed output hashes. It never substitutes clean CPU replay values or changes the original hash.

`activation_preparation.json` preserves the sparse base file hashes, response hashes, and preparation/storage costs. This experiment performs a full audit, so all missing boundaries are revealed before CPU verification. Final proof size therefore includes the revealed tensors; it is not a measurement of sparse-only storage savings.

## Start and monitor

Run from `AFTUNE` with the pinned `.venv`. The local frozen dataset is checked against its manifest; `fetch` explicitly obtains it if needed. The v3 default configuration is `formal-seed0-cal32-v3.json`. The default output is the workspace `.cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3` directory.

Run the isolated full-length preflight first:

```bash
.venv/bin/python -m experiments.inference_integrity.preflight --config experiments/inference_integrity/formal-seed0-cal32-v3.json --output ../.cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3-preflight
```

Then start the complete supervised run:

```bash
.venv/bin/python -m experiments.inference_integrity.supervise --output ../.cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3 --interval 60
```

The supervisor holds an exclusive run lock, prepares immutable inputs and reference, calibrates all 32 clean requests, runs the shared pool and positives, summarizes, audits all source evidence and reconstructed sampling results, and writes a Chinese result report next to the run directory. It saves `runner_state.json`, `progress.jsonl`, `monitor.latest.json`, and one `monitor.jsonl` snapshot every 60 seconds. An error stops the process with a traceback and preserved evidence. Resume the same command; add `--retry-errors` to archive failed source tasks and replace their shared result versions consistently.

Read-only monitoring:

```bash
.venv/bin/python -m experiments.inference_integrity.monitor --output ../.cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3 --pid PID
.venv/bin/python -m experiments.inference_integrity.live_outcome_monitor --root ../.cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3 --watch PID --interval 60
```

GPU stages and GPU telemetry need operating-system GPU access. Missing GPU, GPU OOM, less than 768 MiB free GPU memory, less than 5 GiB free disk, or three consecutive identical execution errors stop the run for diagnosis. Keep the frozen experiment code and inputs unchanged during a run. The older stage/completion audit and renderer target legacy layouts; v3 supervision uses `v3_audit` instead.

Individual stages remain available via `python -m experiments.inference_integrity`: `prepare`, `calibrate`, `run`, and `summarize`. `calibrate --limit K` is diagnostic and cannot freeze official thresholds. `run --limit K` processes only the first K frozen input indices. `summarize --partial` explicitly labels an incomplete diagnostic summary.

## Decisions and outputs

The primary score is the maximum relative L2 over all recorded blocks, using an origin-specific calibration maximum. Explicit input/disk integrity failures or scores strictly above the frozen threshold are `DETECTED`; complete checks below it are `PASSED`; missing evidence, invalid numbers, and execution failures are `ERROR`. Failed or unknown injections are separate categories.

Strict computed-output hashes are reported separately. Normal GPU-to-CPU FP32 differences can cause hash rejection, so strict-mode rejection alone does not establish a violation.

The run first checks every block, then simulates only 8% sampling without replacement, 1,000 repeats per unique evidence record. Each repeat selects `max(1, ceil(0.08 * B))` blocks: 70 of 864 for D and 1 of 9 for Model. A saved source-evidence key and post-commit seed reconstruct every draw. Shared controls reuse the same draws across conditions. Repeats do not increase the independent-request sample size.

`summary.json` contains per-condition numeric/hash confusion matrices, rates, Wilson intervals, coverage, sampling results, source references, and costs counted once per source task. `completion_audit.json` checks the frozen inputs, code/model/reference hashes, calibration, all block coordinates, event truth, shared mappings, decisions and sampling reconstruction. The report is generated only after that audit passes.

This mechanism study does not establish checkpoint provenance or TEE isolation/attestation. KV uses full-prefix context substitution and Serving uses a rounding proxy. A passing Model recomputation can remain a false negative for source-generation integrity while its submitted Llama trace is computationally valid.

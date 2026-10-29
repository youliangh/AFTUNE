# 推理完整性实验交付包

本目录打包已经完成的 **Llama v3** 和 **Qwen v4** 实验。每轮为 32 个校准请求、416 个正式源生成任务、640 份去重证据、六个 256 样本视图与 1,536 次引用。每轮六类实验共享该轮的 224 个阴性；两轮复用原始数据行、阳性索引与指定注入时序，重新生成响应、证据与阈值。

打包后已在 Python 3.12 标准库环境中完成两份包的全量离线复核，并通过现有 33 项实验测试。记录见 [validation.json](validation.json) 与 [unit-tests.log](unit-tests.log)。冻结代码按原字节提交，保留其既有末尾空行。

## 文档与实现入口

| 内容 | Llama v3 | Qwen v4 |
|---|---|---|
| 实验方案 | [实验方案](../inference_integrity/实验方案.md) | [实验方案](../inference_integrity_qwen/实验方案.md) |
| 实现与运行细节 | [README](../inference_integrity/README.md) | [README](../inference_integrity_qwen/README.md) |
| 最终报告 | [实验结果](../inference_integrity/实验结果.md) | [实验结果](../inference_integrity_qwen/实验结果.md) |
| 正式配置 | [v3 JSON](../inference_integrity/formal-seed0-cal32-v3.json) | [v4 JSON](../inference_integrity_qwen/formal-qwen-seed0-cal32-v4.json) |
| 原始完成审计代码 | [v3_audit.py](../inference_integrity/v3_audit.py) | [audit.py](../inference_integrity_qwen/audit.py) |

生成、注入、Recorder 适配、CPU 验证、预检、监控和统计实现随对应实验目录一并提交。根目录的 `verify_inference.py` 包含本轮使用的可信参考缓存与数值/哈希双模式支持；`pyproject.toml`、`uv.lock`、`.python-version` 固定 Python 与依赖。测试随两个实验目录和根目录 `tests/` 保存。

[初始蓝图](../../experiment-blueprint.md)与[早期阶段快照](../inference_integrity/阶段性实验报告.md)保留为历史资料。它们包含已被正式 v3/v4 方案替代的 GPU 验证、R 证据和多档 Serving 设置，不应与两轮最终结果合并。

## 结果摘要

| 条件 | Llama v3 数值 TP/32 | Qwen v4 数值 TP/32 |
|---|---:|---:|
| KV Cache | 32 | 32 |
| Serving r6 | 5 | 1 |
| Model Mismatch | 1 | 22 |
| Token Selection | 4 | 2 |
| Computation | 32 | 32 |
| Input Mismatch | 8 | 0 |

共享阴性的数值误报：v3 D 为 14/224、Model 为 3/224；v4 D 为 0/224、Model 为 2/224。两轮严格哈希模式的阴性误报均为 224/224。两轮的错误、注入失败与未知注入均为 0。完整四格、Wilson 区间、覆盖率、deployment recall、抽样结果与解释边界见报告及各自 `summary.json`。

v3 声明 Llama、替代源 Qwen；v4 声明 Qwen、替代源 Llama。v3 的输入是 BOS＋31 个内容 token，v4 是无 BOS 的 32 个内容 token；288 行中仅 12 行的实际请求文本相同。v3/v4 的 D 每轮分别抽 70/864、100/1248，Model 分别抽 1/9、2/13。KV 可选层总体、桥接长度和计算成本亦不同；这些差异及逐行请求文本/IDs 保存在 Qwen 包的 `paired_comparison.json`，不能把结果差异仅归因于模型大小。共享引用和每证据 1,000 次抽样重复不扩大独立样本量。

## 压缩证据与离线复核

- `llama-v3-evidence.tar.gz`、`qwen-v4-evidence.tar.gz`：保持原始运行目录的相对路径，保存全部校准块分数、640 份数值/严格哈希块结果、416 份源结果及注入真值、生成 IDs、响应文本、共享映射、阈值、配置/环境/代码指纹、汇总、完成审计、报告和可用的预检记录。Qwen 包另外保存配对时序、两轮对照与补充审阅记录。
- `bundle-manifest.json`：每份压缩包和每个包内文件的 SHA-256 与字节数。原始 JSON 按字节复制；原始机器绝对路径保留为运行溯源记录，查阅入口以本 README 的相对链接为准。
- `commitment.export.json`：保留冻结时间、激活准备成本、承诺文件数和原始 `commitment.json` 的 SHA-256；体积较大的逐张量文件哈希映射未导出，导出文件不冒充原始承诺。

从仓库根目录执行，使用 Python 3.12 标准库即可复核统计，无需模型、GPU 或安装 Torch：

```bash
python3 experiments/artifacts/verify_bundle.py
```

默认检查压缩包和所有成员哈希、冻结的代码指纹、32 条校准阈值、全部块坐标与判定、六个视图的双模式指标/Wilson 区间、共享引用、去重成本，并逐份重建每次 1,000 轮抽样。两轮合计核对 1,280 份证据及 1,280,000 轮相关抽样。仅快速核对保存的抽样汇总时使用 `--quick`；此选项不重新生成抽样索引。

v3 没有单独保存 `challenge_created_at` 字段；复核器检查实际存在的阈值冻结、承诺冻结和完成时间，不补造时间戳。v4 另检查承诺冻结→challenge 创建→结果完成的顺序。

模型权重、完整数据集、参考模型快照、原始激活张量和已归档中断尝试不进入 Git。上述命令复核导出的分数与统计，不重新执行 GPU 推理、CPU 模型重放或逐张量承诺验证；这些原始检查的完成记录随包保存。原始运行目录仍保留索引 0/64/128/192 的完整张量。重新运行模型实验需按对应 README 准备匹配哈希的 checkpoint、固定数据版本和运行环境，并完成实际长度预检；配置中的绝对路径也参与冻结指纹。

## 从本地原始运行重新导出

```bash
python3 experiments/artifacts/build_bundle.py \
  --runs ../.cache/experiment-runs \
  --output experiments/artifacts
python3 experiments/artifacts/verify_bundle.py
```

导出器读取已经通过完整审计的源目录，不改写其分数、阈值、报告或判定。压缩成员顺序、时间戳、权限和 gzip 时间戳固定；相同源字节得到相同压缩包。

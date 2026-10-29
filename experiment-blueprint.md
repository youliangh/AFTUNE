# 实验设计

对于每种inference integrity violation type都做一组实验。每组实验有M条序列，其中均匀随机有N条序列有inference integrity violation，每条序列prefill P个tokens，decode直到max_length个 tokens时停止，每条序列只有一处地方有violation。

## 参数设置
- M=256
- N=32
- P=32
- max_length=128
- 独立干净校准集=32条序列，与正式实验序列不重叠
- 正式实验每组仅使用一个随机种子（seed=0），不做随机种子重复实验
- 生成与审计均使用GPU

## Inference Integrity Violation Type

### KV Cache Mismatch:
decode的某一step中，用另一条干净请求实际计算得到的K/V替换一个历史位置，影响当前最后一个query；记录与审计仍使用完整前缀形状。

### Serving Configuration:
运行环境发生变化，例如route到不同设备上，或者在不符合用户配置的环境下推理，例如用户给定的thinking budget为0，但推理过程中仍然存在思维链条
本实验用FP32中间结果舍入模拟一次route切换；三档舍入强度在相同序列使用同一个切换step，只改变舍入位数。

### Model Mismatch:
替代模型生成后，通过文本重新分词把序列传给声明模型；声明模型在最终重算时现场生成证明。verifier只看到这次声明模型重算的现场信息，Model Mismatch按一个证据条件计数。

| 实验类型 | 生成模型 | 最终重算模型 |
|---|---|---|
| 非攻击、KV、Computation、Input | 声明模型 | 声明模型 |
| Model：声明 Llama | Qwen | Llama |

### Token Selection Mismatch：
decode的某一step中，强行更改某个token

### Computation Mismatch：
某一处计算时，kernel出错导致了计算结果错误

### Input Mismatch：
prompt中信息发生删除或重排序

## 检测指标

- True/False Postives/Negaives, 收集原始数据用于计算高阶统计数据

**False Positives**: 该序列中无violation，但是工具误判该序列有violation
**True Positives**: 该序列有violation，该工具也判断出该序列确实有violation
**False Negatives**: 该序列有violation，但工具判断出该序列无violation
**True Negatives**: 该序列无violation，该工具也判断出该序列无violation

## 设置
- Qwen3.5-0.8B，本地权重: ../models/Qwen3.5-0.8B

### 测试模型
- Llama-3.2-1B-Instruct，本地权重: ../models/Llama-3.2-1B-Instruct

## 交付内容

- 实验具体实施细节
- 自动化实验运行脚本

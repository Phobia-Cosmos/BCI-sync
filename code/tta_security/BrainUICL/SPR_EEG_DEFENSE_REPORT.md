# SPR-EEG Continual Learning Defense Report

Date: 2026-07-13

This report adapts Self-Purified Replay (SPR, ICCV 2021) to the
BrainUICL ISRUC individual continual-learning protocol and evaluates both its
useful operating range and its failure modes.

## 1. Experimental Scope

- Dataset: ISRUC subgroup I, 98 available subjects.
- Input: sequences of 20 sleep epochs, each with 2 EOG + 6 EEG channels.
- Task: five-class sleep staging.
- Split seed: 4321.
- Probe stream: the first 10 BrainUICL new individuals.
- Training budget: 3 CPC epochs and 3 joint incremental epochs per individual.
- Pretrained model and source replay data are identical across variants.
- Metrics: final old/generalization ACC, MF1, average accuracy (AAA), average
  MF1 (AAF1), forgetting rate (FR), new-individual plasticity, and replay-label
  error measured with held-out ground truth.

Ground-truth labels are only used to report replay-label error. The defense
does not read them.

## 2. Extracted SPR Method

The original image SPR method contains two networks and two memories:

1. Delayed buffer `D`: temporarily holds the current data stream.
2. Purified buffer `P`: stores samples judged likely to have clean labels.
3. Expert network: trained self-supervised on `D`, then used to compute
   class-conditional feature centrality.
4. Base network: trained with self-supervised replay on `D union P`.
5. Self-Centered Filter: constructs one feature-similarity graph per observed
   class, estimates stochastic eigenvector centrality, fits a two-component
   Beta mixture, and interprets the high-centrality posterior as cleanliness.
6. Downstream inference: supervised training only uses purified memory.

## 3. EEG Mapping

| SPR component | BrainUICL / EEG implementation |
| --- | --- |
| Incoming task-free stream | Sequential unseen ISRUC individuals |
| Delayed buffer | All sequences from the current individual |
| Observed noisy label | BrainUICL teacher pseudo-label for each sleep epoch |
| Expert self-supervision | BrainUICL CPC adapted only on the current individual |
| Base Self-Replay | Optional CPC on current data plus sampled replay data |
| Class graph vertex | One 30-second EEG epoch embedding |
| Class graph grouping | Predicted sleep-stage pseudo-label |
| Edge weight | Non-negative cosine similarity between expert embeddings |
| Stochastic ensemble | Five sampled similarity graphs per sleep stage |
| Clean posterior | Two-component Beta-mixture posterior over centrality |
| Purified memory unit | A 20-epoch EEG sequence |

Sequence acceptance first requires BrainUICL's original confidence rule
(`15/20` epochs with confidence at least `0.9`). The epoch clean posteriors are
then aggregated into a sequence score. A ranked minimum-acceptance fallback
keeps the highest-centrality 75% of candidates when an absolute threshold
would remove too much individual or class coverage.

## 4. Implementation

- `model/spr_eeg.py`
  - stochastic graph construction;
  - power-iteration eigenvector centrality;
  - guarded two-component Beta-mixture EM;
  - epoch-to-sequence purification.
- `experiments/rttdp_brainuicl_full.py`
  - `--defense-mode spr`;
  - optional EEG Self-Replay;
  - SPR buffer filtering and purity diagnostics;
  - reproducible symmetric buffer-label noise;
  - clean, noisy, and adaptive-attack variants with reset random seeds.
- `tests/test_spr_eeg.py`
  - verifies that pseudo-label/feature-cluster mismatches receive lower clean
    probabilities;
  - verifies input-shape validation.

## 5. Main Results

### 5.1 Clean stream and 40% random buffer-label noise

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| SPR ranked clean | 0.7005 | 0.6666 | 0.7059 | 0.6841 | 0.0028 |
| BrainUICL + 40% noise | 0.7005 | 0.6734 | 0.7091 | 0.6886 | 0.0028 |
| Full SPR, strict filter | 0.6861 | 0.6515 | 0.7029 | 0.6817 | 0.0233 |
| Full SPR, relaxed filter | 0.6775 | 0.6416 | 0.6923 | 0.6661 | 0.0355 |
| SPR ranked filter-only | **0.7059** | **0.6805** | 0.6980 | 0.6761 | 0.0049 |

The ranked filter-only variant improves final noisy-stream ACC by 0.54
percentage points and MF1 by 0.71 points over noisy BrainUICL. It also has no
measurable clean-stream penalty in this probe.

However, the final gain is small and AAA/AAF1 are lower. The defense changes
the intermediate trajectory and does not dominate BrainUICL at every step.

### 5.2 Purification diagnostics

| Variant | Mean error before | Mean error after | Accepted / candidates |
| --- | ---: | ---: | ---: |
| Strict SPR | 0.5569 | **0.4339** | 93 / 191 |
| Ranked SPR | 0.5304 | 0.5196 | 172 / 227 |

Strict filtering removes substantially more noisy labels, but loses too much
EEG individual/class coverage and hurts classification. Ranked filtering gives
up most of the purity gain to preserve diversity, producing better final
accuracy. This is the central purity-diversity tradeoff on EEG.

### 5.3 Adaptive proxy-meta poisoning

| Variant | ACC | MF1 | AAA | AAF1 | FR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainUICL clean | 0.6943 | 0.6601 | 0.7089 | 0.6876 | 0.0117 |
| BrainUICL proxy-meta | 0.6195 | 0.5684 | 0.5774 | 0.5146 | 0.1181 |
| SPR ranked proxy-meta | **0.5495** | **0.4874** | 0.5749 | 0.5130 | 0.2178 |

SPR does not defend this attack. Mean pseudo-label error changes from 0.7432
before filtering to 0.7464 after filtering. The attacker moves many samples
into a coherent, high-confidence wrong cluster, violating SPR's assumption
that clean samples form the largest central feature cluster inside each label.
Filtering then retains the attack cluster and removes useful diversity.

The stronger direct `model_nhe` diagnostic similarly collapses SPR ranked to
ACC 0.2270 and FR 0.6768. A replay-purification defense is not expected to stop
an attacker that directly changes model updates.

## 6. Interpretation

The extracted method is useful as a narrow label-noise defense:

- It identifies isolated pseudo-label/feature mismatches.
- It can measurably increase replay purity.
- With diversity-preserving ranked selection, it provides a small final gain
  under random buffer-label noise without harming clean final accuracy.

It is not a general poisoning defense:

- Full Self-Replay directly transferred from image SPR causes EEG feature
  drift under this short BrainUICL budget.
- Absolute purity thresholds remove too many subject-specific sequences.
- Centrality cannot identify a coherent adversarial cluster whose labels,
  confidence, and features have all moved together.

The practical configuration from this probe is therefore `SPR ranked
filter-only`, not the literal full image SPR recipe. A stronger EEG defense
would need temporal consistency, source-anchor distances, class/subject quotas,
and an explicit detector for coherent distribution shifts.

## 7. Reproduction Commands

Environment:

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python
```

BrainUICL clean/noisy plus SPR noisy:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --no-save-checkpoints
```

Ranked SPR filter-only:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --run-defense-only \
  --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

Proxy-meta comparison:

```bash
PYTHONUNBUFFERED=1 /home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321 \
  --max-subjects 10 --ssl-epoch 3 --incremental-epoch 3 --cross-epoch 2 \
  --batch 16 --num-worker 0 --attack-mode proxy_meta_conflict \
  --proxy-meta-poison-scope individual --proxy-meta-steps 5 \
  --proxy-meta-eps-scale 0.50 --proxy-meta-param-scope classifier \
  --proxy-meta-conflict-weight 5.0 --proxy-meta-confidence-weight 0.1 \
  --proxy-meta-grad-norm-weight 0.0 --proxy-meta-raw-weight 0.001 \
  --proxy-meta-l2-weight 0.0005 --pgd-random-start \
  --defense-mode spr --spr-disable-self-replay --spr-min-accept-ratio 0.75 \
  --no-save-checkpoints
```

## 8. Result Locations

- `experiments/rttdp_brainuicl_runs/probe10_spr_buffer_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_noise40_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_filter_ranked_clean_e3_seed4321`
- `experiments/rttdp_brainuicl_runs/probe10_spr_proxy_meta_ranked_e3_seed4321`

## 9. 原始 SPR 与 EEG 迁移 FAQ

### 9.1 SPR 原文是否使用 CPC

没有。原始 SPR 使用 SimCLR 风格的 NT-Xent 对比损失。每个输入生成两个
增强视图，同一样本的两个视图作为正样本，batch 内其他视图作为负样本。
SPR 的 expert 在 Delayed Buffer 上训练，base 在 Delayed Buffer 与 Purified
Buffer 的并集上训练。

对正样本视图 `(i,j)`，NT-Xent 为：

```text
L(i,j) = -log exp(sim(z_i,z_j)/tau)
              / sum_{k != i} exp(sim(z_i,z_k)/tau)
```

`z` 是 projection head 输出的 L2-normalized embedding，`sim` 为余弦
相似度，`tau` 是 temperature；SPR 配置使用 `tau=0.5`。一个 batch 的
`B` 个输入生成 `2B` 个视图，对每个视图分别计算损失。

EEG 版本使用 CPC，是因为 BrainUICL 已经提供了适合 EEG 时间序列的 CPC
自监督目标。这是模态迁移，不是 SPR 原文配置。CPC 负责学习特征，本身不
执行样本过滤。

### 9.2 SPR 原文的 buffer 大小

| Dataset | Delayed Buffer | Purified Buffer |
| --- | ---: | ---: |
| MNIST | 300 | 300 |
| CIFAR-10 | 500 | 500 |
| CIFAR-100 | 1250 | 5000 |
| WebVision | 1000 | 1000 |

原 SPR 的 Purified Buffer 是固定容量。满容量后，class-aware reservoir
先确定需要淘汰的类别，再优先删除该类别中 clean probability 最低的样本。

当前 EEG 对比实验为了保持 BrainUICL 的 memory protocol，没有额外施加
固定容量：初始 source buffer 为 1030 条 sequence；40% 噪声 BrainUICL
最终为 1231 条，strict SPR 为 1123 条，ranked SPR 为 1202 条。这是与
原 SPR 的明确差异。

### 9.3 EEG buffer 存储单位

BrainUICL 和当前 SPR-EEG buffer 都存储 sequence，而不是完整 subject，也
不是独立的 30 秒 epoch：

```text
one stored sequence = [20 epochs, 8 channels, 3000 samples]
stored label          = [20 epoch labels]
```

当前 subject 的全部 sequence 构成逻辑上的 Delayed Buffer。过滤图的顶点
是 sequence 内的 30 秒 epoch，但最终进入长期 replay 的单位仍是完整
20-epoch sequence。一个 subject 的 sequence 只会有一部分进入 replay。

### 9.4 Self-Centered Filter 基于哪些数据

原 SPR 只对当前 Delayed Buffer 建图和过滤，不会在每个步骤重新过滤整个
历史 Purified Buffer。流程为：

1. 所有新流样本先进入 Delayed Buffer。
2. expert 使用 Delayed Buffer 做自监督训练。
3. 按 Delayed Buffer 中的观测标签分组。
4. 每个类别分别建立相似图，计算 eigenvector centrality。
5. Beta mixture 将中心性转换为 clean posterior。
6. 按 clean posterior 随机选择进入 Purified Buffer 的样本。
7. base 使用 Delayed Buffer 与 Purified Buffer 做 Self-Replay。
8. 清空 Delayed Buffer，处理下一段数据流。

原文没有进入 Delayed Buffer 的置信度要求。Delayed Buffer 的作用正是先
隔离尚未判断是否干净的数据，积累足够的同类邻居后再决定是否进入长期
memory。

### 9.5 原文是否使用分类器置信度过滤

原文没有 `softmax >= 0.9` 或 `15/20 epochs` 规则。原 SPR 有带标签的数据
流，直接使用可能带噪的观测标签构图。它使用中心性 Beta mixture posterior
作为 clean probability，并通过 `clean_probability > Uniform(0,1)` 随机
接纳样本。

当前 EEG 版本的 `0.9 + 15/20` 是 BrainUICL 伪标签协议的前置门限：

```text
BrainUICL confidence gate -> SPR centrality filter -> replay buffer
```

### 9.6 为什么构建随机相似图

随机图不是与特征无关的随机连接。非负余弦相似度被当作边存在概率：

```text
edge(i,j) = 1 if cosine_similarity(i,j) > Uniform(0,1)
```

高相似样本更容易连接。原文采样五张图，分别计算中心性和 Beta mixture
posterior 后取平均，降低单次相似度误差、偶然边以及少量错标样本获得虚假
高中心性的影响。

### 9.7 EEG 中一张 graph 的范围

不是每条 sequence 的 20 个 epoch 单独构成一张图。当前实现先收集一个
subject 所有候选 sequence 中的高置信 epoch，再按预测睡眠阶段分为最多
五组，每个睡眠阶段建立一张图。

若一个 subject 有 `N` 条 sequence，则最多有 `20N` 个 epoch 顶点，分别
进入五个 class graph。中心性在当前 subject（逻辑 Delayed Buffer）范围内
计算，之后再把 epoch clean posterior 聚合成 sequence score。

### 9.8 old/new individual 是否都评估

是。每适配一个新 subject 后，都在固定 19 个 old/generalization subjects
上计算 ACC、MF1、AAA、AAF1 和 FR。对新 subject 则分别评估：

- initial：原始预训练模型；
- before：当前 continual model 在适配前；
- after：适配当前 subject 后。

40% buffer-label noise 的 10-subject 实验中：

| Method | Initial ACC | Before ACC | After ACC | After MF1 |
| --- | ---: | ---: | ---: | ---: |
| BrainUICL noisy | 0.5920 | 0.5885 | 0.6124 | 0.5302 |
| SPR ranked | 0.5920 | 0.5705 | 0.5898 | 0.5167 |

因此 ranked SPR 改善了最终 old ACC/MF1，但降低了新个体 plasticity。

### 9.9 strict SPR 与 ranked filter-only

| Setting | Strict SPR | Ranked filter-only |
| --- | --- | --- |
| Base Self-Replay | current + replay CPC | disabled |
| Absolute clean threshold | enabled | enabled |
| Minimum clean epochs | 12 | ranked fallback |
| Minimum acceptance | none in completed strict run | top 75% candidates |
| Accepted/candidates | 93/191 | 172/227 |
| Error before/after | 0.5569/0.4339 | 0.5304/0.5196 |
| Final ACC | 0.6861 | 0.7059 |

ranked filter-only 是 EEG 工程变体，不是原文 SPR。它同时改变了 Self-Replay
和过滤回退规则，因此不能视为单变量消融。

### 9.10 EEG 标签噪声如何生成

40% symmetric buffer-label noise 不修改 EEG/EOG 信号。student 为每个
sleep epoch 生成伪标签后，每个标签独立以 40% 概率替换成另外四类中的
随机类别，并保证替换后类别不同。噪声只在写入 replay 前注入，影响后续
subject 的 replay；当前 subject 已完成的即时更新不受该噪声影响。

BrainUICL baseline 直接保存这些噪声标签；SPR 在噪声注入后、写入 replay
前执行中心性过滤。随机过程由 seed、subject step 和 sequence index 固定。

## 10. 纯 SPR-EEG 与当前混合方案的边界

### 10.1 当前伪标签由谁生成

当前混合实现中，CPC 不生成监督标签：

1. CPC-adapted guiding/teacher model 在 joint update 中生成训练伪目标。
2. 完成当前 subject 适配后，student 生成最终 buffer 伪标签和置信度。
3. CPC-adapted expert 只提供 Self-Centered graph 的 EEG embedding。

原始 SPR 的 expert 同样不生成标签；原文直接使用数据流携带的观测标签。

### 10.2 为什么不能默认伪标签完全正确

未见个体存在明显 EEG domain shift，高 softmax confidence 不等于标签正确。
在没有人工注入噪声的 clean probe 中，通过 BrainUICL confidence gate 的
候选 sequence 仍观察到约 10% 到 25% 的 epoch 伪标签错误。若默认伪标签
完全正确，就无法评估 replay error accumulation，也是引入 SPR 的原因。

可以把伪标签视为“正常但带未知噪声的观测标签”，这与 SPR 的问题设定
一致；不应把它们视为 ground truth。

### 10.3 CPC 会不会导致保留数据太少

CPC 只在所有输入 `x` 上训练 representation，不会删除数据。造成数据过少
的是 BrainUICL confidence gate 和 strict SPR sequence threshold 的串联。

更接近原 SPR 的 EEG 方案应让所有新样本先进入 Delayed Buffer，CPC 在全部
输入上训练 expert，再由中心性决定进入 Purified Buffer 的概率。若必须使用
伪标签，可以降低或取消硬 confidence gate，改用 confidence soft weighting、
每类 top-k 或 ranked minimum acceptance，避免在构图前丢失太多样本。

### 10.4 subject 候选 sequence 如何生成

当前混合流程为：

1. 加载当前 subject 的全部 36 到 52 条 sequence。
2. 模型输出 `[N, 5, 20]` logits。
3. 对每个 epoch 计算 softmax confidence 和 pseudo-label。
4. 至少 15/20 个 epoch 的 confidence 不低于 0.9，sequence 才成为候选。
5. 噪声实验在这里对 epoch pseudo-label 注入对称噪声。
6. expert 为候选 epoch 生成 embedding，按 pseudo-label 构建 class graph。
7. epoch clean posterior 聚合成 sequence score。
8. strict threshold 或 ranked fallback 决定是否写入 replay。

### 10.5 不参考 BrainUICL 的纯 SPR-EEG

当前报告中的 0.7059 ACC 不是纯 SPR-EEG 结果。它仍复用了 BrainUICL 的：

- ISRUC source/new/old individual split；
- 预训练 EEG backbone；
- CPC guiding model；
- 0.9 与 15/20 confidence gate；
- pseudo-label joint update；
- old/new evaluation protocol。

在第 12 节正式实验完成前没有可报告的纯 SPR-EEG 数值；现在应以第 12 节
独立 runner 的结果为准，不能用前述混合结果代替。

纯 SPR-EEG 必须先确定监督条件：

- 若在线流提供 sleep-stage 标签：直接把可能带噪的观测标签作为 SPR 的
  `y`，不使用 guiding model 或 confidence gate；CPC/NT-Xent 只学习特征。
- 若在线流完全无标签：SPR 本身无法按类别构建 Self-Centered graph，仍需
  pretrained classifier、聚类或其他 pseudo-labeler，不能声称完全不依赖
  标签生成机制。

纯 SPR-EEG 已在第 12 节通过独立 runner 实现并运行。当前 BrainUICL+
SPR-ranked 的结果仍不能代替该数值。

## 11. 纯 SPR-EEG 实验协议（已锁定）

### 11.1 保持不变的 BrainUICL 协议

新实验固定使用 `seed=4321`，直接复用 `split_subjects()` 产生的四组 subject：

- `source/train`：仅用于已有 ISRUC 预训练模型和 source memory；
- `val`：不进入在线持续学习；
- `old/generalization`：每次适配后都重新评价；
- `new`：按原顺序逐个 subject 到达。

评价协议也保持不变。old subject 记录 ACC、MF1、AAA、AAF1、FR；每个 new
subject 分别记录 initial（原预训练模型）、before（上一时刻模型）和 after
（处理当前 subject 后的模型）的 ACC/MF1。这样纯 SPR 与 BrainUICL 的结果
具有相同 split 和评价口径。

### 11.2 被替换为原文 SPR 的部分

在线方法不再使用 BrainUICL guiding model、teacher pseudo-label、`0.9`
confidence threshold、`15/20` gate 或 pseudo-label joint update。当前 new
subject 的人工 sleep-stage 标签作为 SPR 的观测标签；噪声实验只在训练侧按
0%、20%、40% 或 60% 注入 symmetric label noise，评价始终使用原始标签。

每个 subject 是一个逻辑 Delayed Buffer，执行顺序与 SPR 一致：

1. 独立 expert 在当前 Delayed Buffer 上用 SimCLR NT-Xent 训练；
2. continual base 在当前 Delayed Buffer 和 Purified Buffer replay 上做
   NT-Xent Self-Replay；
3. expert embedding 按带噪观测标签分组，建立 `E_max=5` 个随机余弦相似图；
4. eigenvector centrality 和两分量 Beta mixture 得到 epoch clean posterior；
5. 按 `clean_probability > Uniform(0,1)` 接纳 epoch；
6. 从 base representation 复制 evaluation model，只用 Purified Buffer 的
   retained epoch mask 做监督 fine-tuning；
7. 执行相同的 old/new 评价。

这里的 NT-Xent temperature 固定为原文的 `0.5`。EEG 的两种 view 使用轻微
jitter、幅值缩放、时间遮挡和 channel dropout；CPC 不参与纯 SPR 实验。

### 11.3 EEG Purified Buffer 预算

Purified Buffer 固定为 5000 个 30 秒 epoch reference，其中 3000 个是
sequence-aware 抽样的 source epoch，2000 个是 new-subject 动态分区。动态
分区使用 class-aware replacement，并优先淘汰过量类别中 clean posterior
最低的 epoch。每条记录为：

```text
(sequence_path, epoch_index, observed_label, clean_probability)
```

模型加载时仍读取完整 `[20 epochs, 8 channels, 3000 samples]` sequence 作为
Transformer 上下文，但监督 loss 只作用于 retained epoch mask。这个设计使
过滤粒度与原 SPR 的 per-sample 语义一致，同时不破坏 BrainUICL backbone 的
20-epoch 输入约束。source 与 dynamic 容量分开，是为了避免 clean probability
为 1 的 source epoch 在 memory 满后永久挤掉所有 new epoch。

### 11.4 实现与结果边界

独立 runner 为 `experiments/spr_eeg_pure.py`。它可以复用 ISRUC loader、预训练
backbone、subject split 和评价函数，但不会调用 BrainUICL 的 continual
training 路径。纯 SPR 数值必须来自该 runner 的完整输出；第 8 节的 0.7059
仍是 BrainUICL+SPR-ranked 混合实验，不能改名为纯 SPR。

## 12. 纯 SPR-EEG 正式实验结果

### 12.1 运行配置

- new subject：固定顺序前 10 个，`64, 89, 1, 27, 60, 5, 52, 42, 80, 26`；
- old/generalization subject：固定 19 个，与 BrainUICL 完全相同；
- 训练轮次：expert NT-Xent 10、base NT-Xent 10、evaluation fine-tune 10；
- noise：clean（0%）和 symmetric 40%；
- memory：5000 epoch，其中 source 3000、dynamic 2000；
- temperature：0.5，Self-Centered graph ensemble：5；
- confidence gate、guiding model、pseudo-label joint update：均关闭。

正式输出在本地（由 `.gitignore` 排除）：

```text
experiments/rttdp_brainuicl_runs/pure_spr_10sub_e10_seed4321/
experiments/rttdp_brainuicl_runs/brainuicl_10sub_e10_noise40_seed4321/
```

两组纯 SPR 运行均完成 10/10 subjects；每组 stability 曲线包含 initial 加 10
次适配共 11 个点，最终 buffer 均严格满足 5000-epoch 容量。

### 12.2 与同轮次 BrainUICL 协议对照

| Method | Noise | Old ACC | Old MF1 | AAA | AAF1 | FR | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL, 10 epochs | 0% | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 | 0.5946 | 0.5235 |
| Pure SPR-EEG, 10 epochs | 0% | **0.7123** | **0.6881** | **0.7153** | **0.6954** | **0.0140** | **0.6424** | **0.5746** |
| BrainUICL, 10 epochs | 40% | 0.6678 | 0.6238 | 0.7072 | 0.6833 | 0.0493 | 0.5942 | 0.5234 |
| Pure SPR-EEG, 10 epochs | 40% | **0.7324** | **0.7088** | **0.7155** | **0.6914** | **0.0426** | **0.6293** | **0.5608** |

在相同 subject 顺序和 10-epoch 预算下，纯 SPR-EEG 的 clean 最终 old ACC/MF1
比 BrainUICL 高 4.47/6.35 个百分点，40% noise 下高 6.46/8.50 个百分点。
clean 的 new after ACC/MF1 高 4.78/5.11 个百分点，40% noise 下高
3.52/3.74 个百分点。

这不是完全相同监督条件下的消融。BrainUICL 在线阶段使用 guiding/teacher
伪标签，且 40% noise 只污染写入 replay 的伪标签；纯 SPR 遵循原文的有标签
noisy stream 假设，当前 subject 的人工标签是观测标签，40% noise 在进入
Delayed Buffer 时注入。表格用于回答“保留 split 和评价协议、替换 CL 方法”
后的结果，不能解释成 SPR 在无标签适配条件下优于 BrainUICL。

### 12.3 过滤和 memory 诊断

| Diagnostic | Clean | 40% noise |
| --- | ---: | ---: |
| Delayed epochs | 8880 | 8880 |
| Injected noisy epochs | 0 | 3609 |
| Accepted epochs | 5624 | 4897 |
| Acceptance rate | 63.33% | 55.15% |
| Accepted noisy epochs | 0 | 约 1149 |
| Injected noise removed | - | **68.16%** |
| Final total buffer purity | 100% | 94.22% |
| Final dynamic partition purity | 100% | 85.55% |

40% 运行的实际随机噪声率为 `3609/8880=40.64%`。Self-Centered Filter 接纳
的 4897 个 epoch 中约 1149 个仍为错标，因此即时 accepted-set purity 约
76.5%；后续 class-aware replacement 使最终 2000 个 dynamic epoch purity
达到 85.55%。94.22% 是把 3000 个已知干净 source epoch 一起计算的总 purity，
不能用它夸大过滤器单独的效果。

### 12.4 与 SPR 原论文比较

原论文 Table 1 在 40% symmetric noise 下报告最终 accuracy：MNIST 86.7%、
CIFAR-10 43.0%；Table 3 报告注入噪声被过滤的比例：MNIST 96.5%、CIFAR-10
70.5%。本次 EEG 的 injected-noise removal 为 68.2%，接近论文 CIFAR-10 的
70.5%，但低于 MNIST 的 96.5%。

EEG 的 73.24% final old ACC 不能与论文的 86.7% 或 43.0% 直接比较，因为
数据模态、类别、任务构造、backbone、source pretraining 和评价集合都不同。
可比较的是方法行为：在 40.64% 输入噪声下，过滤器显著降低长期 memory 的
错标率；代价是只接纳 55.15% 的输入 epoch，且 new after ACC 从 clean 的
64.24% 降到 62.93%，仍存在 purity 与 plasticity 的权衡。

### 12.5 正式复现命令

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_pure.py \
  --output-root experiments/rttdp_brainuicl_runs/pure_spr_10sub_e10_seed4321 \
  --max-subjects 10 --noise-rates 0.0 0.4 \
  --expert-epochs 10 --base-epochs 10 --ft-epochs 10
```

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/brainuicl_10sub_e10_noise40_seed4321 \
  --max-subjects 10 --ssl-epoch 10 --incremental-epoch 10 --cross-epoch 2 \
  --batch 16 --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --no-save-checkpoints
```

## 13. BrainUICL、原始 SPR 与无标签 EEG FAQ

### 13.1 `3 CPC + 3 joint epochs` 的含义

报告开头的 3+3 只属于早期 10-subject 混合 probe 的快速预算，不是
BrainUICL 原始代码默认值。BrainUICL `main.py` 默认：

```text
ssl_epoch         = 10
incremental_epoch = 10
cross_epoch       = 2
batch             = 16
```

正式 10-subject 和 49-subject 对照都使用 10 CPC epochs 和 10 joint epochs；
纯 SPR-EEG 使用 expert/base/evaluation fine-tune 各 10 epochs。

### 13.2 SPR expert、base 和 inference model

| Network | Lifetime | Training data | Function |
| --- | --- | --- | --- |
| Expert | 每个 D 重新建立 | 当前 Delayed Buffer | 学习局部自监督特征，供 SCF 建图，不生成标签 |
| Base | 整条流持续存在 | 当前 D + 历史 P | 通过 Self-Replay 学长期 representation |
| Inference | 每个评价点从 base 复制 | Purified Buffer | 监督微调分类头和 representation，用于预测 |

expert 和 base 可以在第一块 D 上看到相同数据，但它们的参数、生命周期和
用途不同。expert 是一次性的局部过滤器；base 是跨块累积知识的 continual
model。

### 13.3 原 SPR 为什么使用 SimCLR NT-Xent

监督交叉熵直接读取可能错误的标签，NT-Xent 只读取输入 `x`。同一样本的两种
随机增强构成 positive pair，batch 中其他样本构成 negatives。训练使模型对
合理增强保持不变，同时区分不同实例。这样 noisy label 不会直接污染 expert
和 base 的 representation，且更一致的特征有利于发现类内异常样本。

原 SPR 的图像数据包括 MNIST、CIFAR-10、CIFAR-100 和 WebVision。一个训练
样本是一张图像和一个可能错误的观测标签。B 个样本产生 2B 个 view，不是一个
batch 只有一个样本。

### 13.4 原 SPR 与 EEG 的 batch/sample 语义

| Stage | 原 SPR sample | EEG sample/context |
| --- | --- | --- |
| Expert/Base NT-Xent | 一张图片 | 一条 20-epoch EEG sequence |
| SCF graph vertex | 一张图片 | 一个 30 秒 EEG epoch |
| EEG sequence shape | - | `[20, 8, 3000]` |

当前 EEG runner 的 expert batch 默认 8 条 sequence。base 通常把最多 8 条当前
sequence 与 8 条 replay sequence 合并，因此 NT-Xent batch 最多约 16 条
sequence，并产生 32 个增强 view。

### 13.5 BrainUICL 如何产生和筛选伪标签

模型对每条 sequence 输出 `[5 classes, 20 epochs]`，所以每条 sequence 产生
20 个 sleep-stage 伪标签。存在两个阶段：

1. joint update：CPC-adapted teacher 为当前 epoch 生成伪标签；只有
   confidence > 0.9 的 epoch 参与当前伪标签交叉熵。
2. buffer merge：适配后的 student 再生成 20 个伪标签；若至少 15/20 个
   epoch 的 confidence 不低于 0.9，则整条 sequence 与全部 20 个标签进入
   replay。

confidence 不足的当前 epoch 不参与 `loss_new`，但仍存在于输入 tensor；它们
没有来自当前伪标签交叉熵的直接梯度。若所属 sequence 通过 15/20 gate，低
置信 epoch 的最终 student 标签仍会随整条 sequence 写入 replay。

### 13.6 BrainUICL 与 EEG-SPR 的 memory 粒度

BrainUICL 接受一条 sequence 后，保存完整 sequence 和全部 20 个伪标签。
当前纯 SPR-EEG 保存 epoch reference：

```text
(sequence_path, epoch_index, observed_label, clean_probability)
```

所以同一 sequence 可能只保留部分 epoch。模型训练仍加载完整 sequence 作为
Transformer 上下文，但监督 loss 只计算 retained epoch mask。早期混合
SPR-ranked 仍然保存完整 sequence，不能与当前 epoch-level P 混为一谈。

### 13.7 Delayed Buffer 和五个 class graph

原 SPR 的 D 是固定容量输入块，不等于 task。迁移到当前 EEG 协议时，一个
subject 的所有 sequence 构成一个逻辑 D。若 subject 有 45 条 sequence，则
最多有 `45*20=900` 个 epoch 顶点。它们按观测或预测标签分成：

```text
W -> graph 0
N1 -> graph 1
N2 -> graph 2
N3 -> graph 3
REM -> graph 4
```

每个 epoch 只进入一张图，不是每条 sequence 单独建图。每张图在当前 subject
范围内，根据非负余弦相似度随机采边，采样 `E_max=5` 张图并平均 centrality
posterior。

### 13.8 无标签 new subject 的关键边界

原 SPR 需要输入 `(x, y_observed)`，因为 SCF 必须按 `y_observed` 分组。完全
无标签时，原算法不能直接运行。当前纯 SPR-EEG 的 clean/40% 实验使用 ISRUC
ground truth（或其人工加噪版本）作为观测标签，因此属于 oracle noisy-stream
实验，不是可部署的无监督 UICL。

真实无标签 EEG 应采用：

```text
guiding model 无监督适配
-> 为全部 epoch 生成伪标签
-> 不使用 0.9/15-of-20 硬删除
-> 所有 epoch 进入 D
-> expert/base NT-Xent
-> 按伪标签建五类 SCF graph
-> 选中的 epoch 和原伪标签进入 P
```

confidence 可以保留为诊断或 soft weight，但不必在建图前删除数据。这是
BrainUICL pseudo-labeler + SPR purifier，不是完全原始的有标签 SPR。

### 13.9 Beta Mixture 和原文接纳规则

SCF 将 centrality 归一化到 `[0,1]`，再用两个 Beta 分布拟合低中心性 noisy
分量和高中心性 clean 分量，通过 EM 得到 `P(clean|centrality)`。它不是标签
生成器，也不能识别形成大而紧密错误簇的 coherent pseudo-label noise。

原 SPR 没有固定 `clean_p > 0.5` 阈值，而是：

```text
accepted = clean_probability > Uniform(0,1)
```

P 满后，从过量类别中优先淘汰 clean probability 较低的样本。strict threshold
和 ranked fallback 都是此前 EEG 工程变体。

### 13.10 pseudo-label joint update 和评价指标

BrainUICL joint update 近似为：

```text
loss = alpha * 当前 subject 高置信伪标签损失
     + (1-alpha) * replay 标签损失
```

并每隔 `cross_epoch=2` 对 replay representation 做 KL 对齐。原 SPR 和当前纯
SPR-EEG 都没有 cross-epoch alignment。

| Metric | Interpretation |
| --- | --- |
| ACC | 全部 epoch 的总体正确率，容易被数量大的 N2 主导 |
| MF1 | 五类 F1 等权平均，更能暴露 N1 等少数类问题 |
| AAA | old ACC 整条适配轨迹的均值 |
| AAF1 | old MF1 整条适配轨迹的均值 |
| FR | `abs(initial-final)/initial`，需与曲线一起解读 |
| Initial/Before/After | source model/当前适配前/当前适配后的 new-subject 指标 |

原 SPR 虽然有标签，仍需持续学习，因为数据按非 IID 流依次到达，不能保存全部
历史数据，没有 test-time task ID，且错误标签会加剧 catastrophic forgetting。

## 14. Source Replay、数据增强与逐步处理对照

### 14.1 当前 EEG 是否使用 BrainUICL source replay

是，但不同实现的粒度不同：

| Method | Source memory |
| --- | --- |
| BrainUICL | source/train 中约 1030 条完整 sequence 持续参与 replay |
| 当前纯 SPR-EEG | 3000 个受保护 source epoch + 2000 个 dynamic epoch |
| 原 SPR | 无 source partition，D/P 初始为空 |

source/dynamic 分区是为了保持 BrainUICL split 和 old stability，是 EEG-specific
设计。原 SPR 从随机初始化 base/expert 和空 D/P 开始，第一块 D 到达后才训练。

### 14.2 Cross-epoch alignment 是否仍存在

不存在。`cross_epoch` 只在 BrainUICL joint update 中使用；当前纯 SPR-EEG、
原 SPR expert/base 和 evaluation fine-tune 均不执行该特征 KL 对齐。

### 14.3 Purified Buffer 标签来源

原 SPR 不生成或修正标签。输入 `(x,y_observed)` 经 SCF 接受后，原
`y_observed` 原样进入 P。当前 oracle EEG clean 使用 ground truth；40% noise
先把约 40% ground truth 随机替换为其他类。真实无标签方案中，P 应保存
guiding model 产生的伪标签。

### 14.4 同一样本的两种增强与 NT-Xent

原图像对同一图片独立执行两次随机裁剪、翻转、颜色变化等增强。EEG 对同一
sequence 独立执行幅值缩放、jitter、时间 masking 和 channel dropout。增强
操作本身不被优化，优化的是 encoder 和 projection head 参数，使两个 view 的
embedding 接近，并与其他样本的 embedding 分离。

以 batch 中三个样本为例，`x1a/x1b` 是 positive pair；`x2a/x2b/x3a/x3b`
是 `x1a` 的 negatives。原 SPR 虽然逐个接收流样本，但会先积累 D，再用包含
很多样本的 batch 训练，而不是一次只训练一个样本。

### 14.5 无标签 EEG 中伪标签与 NT-Xent 的顺序

建议顺序为：guiding model 先做 CPC 无监督适配并固定伪标签；随后 SPR expert
和 base 执行 NT-Xent。NT-Xent 不读取伪标签，先后在数学上独立，但 SCF 建图
前必须已经有固定伪标签。SPR expert 没有分类头，不能因为训练了 NT-Xent 就
自动产生伪标签。

### 14.6 原 SPR replay 如何取样

原 SPR 的 P 单位是一张完整图片。base 每个 iteration 从 D 取当前 batch，并
从 P 随机采样 replay batch；不是每次把整个 P 放进一个 batch。P 满时通常令
D/P 各占 base batch 的一半。监督 fine-tune 使用 batch size 16，多轮遍历 P。

### 14.7 原 SPR 与 EEG 三种状态逐步对照

| Step | 原始 SPR | 当前 oracle SPR-EEG | 建议无标签 SPR-EEG |
| --- | --- | --- | --- |
| 初始模型 | 随机初始化 | BrainUICL source-pretrained | BrainUICL source-pretrained |
| 初始 P | 空 | 3000 source epochs | 3000 source epochs |
| 流输入 | 单张图片 | 一个 subject 的 sequences | 一个 subject 的 sequences |
| 标签 | 输入 noisy label | ground truth/人工噪声 | guiding pseudo-label |
| D 触发 | 固定 300-1250 samples | subject boundary | subject boundary |
| Expert | D 上 NT-Xent | 当前 subject NT-Xent | 当前 subject NT-Xent |
| Base | D+P Self-Replay | current+P Self-Replay | current+P Self-Replay |
| 图顶点 | 图片 | 30 秒 epoch | 30 秒 epoch |
| 图分组 | 输入标签 | ground truth/noisy label | pseudo-label |
| P 单位 | 图片 | epoch reference | epoch reference |
| P 标签 | 输入标签 | ground truth/noisy label | pseudo-label |
| 监督模型 | base copy + P fine-tune | retained-mask fine-tune | retained-mask fine-tune |

### 14.8 N1、N2、F1 和 evaluation timing

N1 是从清醒进入睡眠的浅睡阶段，样本少且难分类；N2 是稳定浅睡阶段，通常
样本最多。F1 是 precision 与 recall 的调和平均：

```text
F1 = 2 * precision * recall / (precision + recall)
```

Macro-F1 对 W/N1/N2/N3/REM 分别计算 F1 后等权平均，避免只依靠 N2 获得较高
ACC。

当前 EEG 协议在每个 new subject 处理完 D、更新 P 后，都从 base 复制一个
evaluation model，在整个 P 上做 retained-mask supervised fine-tune，然后评价
当前 new subject 和固定 old subjects。下一个 subject 到来时 base 继续存在，
而 evaluation model 主要提供预测和 `before` 指标。

## 15. Expert/Base 重复训练、Self-Replay 与论文评价 FAQ

### 15.1 为什么 expert 和 base 第一块都在同一 D 上做 NT-Xent

第一块 D 时 P 为空，因此两张网络确实都只看到同一批数据，但这不是对同一
模型重复优化：

```text
expert(D): 学局部特征，立即用于当前块 SCF，之后重置
base(D):   初始化长期 representation，之后继续在每个 D+P 上累积
```

如果只训练 expert，过滤后没有一个持续模型记住所学 representation；如果只
训练 base，又缺少与长期模型解耦的局部 expert 来判断当前 D 中的异常点。到
第二块以后差异更明显：expert 仍只看当前 D，而 base 同时看当前 D 和历史 P。

### 15.2 做 NT-Xent 后实际发生什么

随机增强策略不会被学习。反向传播更新的是 feature extractor 和 projection
head，使同一样本两种增强的 embedding 更接近、不同样本更远。结果不是生成
新图片或新数据集，而是改变网络参数和特征空间。

官方 expert 用于 SCF 的输出是 normalization 后的 projection embedding；
ResNet SimCLR 配置维度为 256。当前 EEG 实现为保留 20-epoch 时序语义，SCF
使用 transformer encoder 的 512 维 epoch embedding，而 sequence-level
NT-Xent 通过 mean pooling 后再进入 projection head。

### 15.3 EEG 中 SCF 后是否把数据和伪标签放入 P

是。在真实无标签版本中，SCF 只决定接纳，不重新生成标签：

```text
(sequence_path, epoch_index, guiding pseudo-label, clean_p)
```

被接受的 epoch reference 和原 guiding pseudo-label 进入 dynamic P。完整
sequence 仍保留在磁盘并在训练时作为上下文加载。

### 15.4 Self-Replay 发生在哪个阶段

Self-Replay 就是 base 的 NT-Xent 训练阶段，发生在每次 D 触发后的在线适配
过程中。官方代码顺序是：

```text
D full
-> train expert on D
-> train base on D + existing P  (Self-Replay)
-> SCF(D)
-> accepted samples enter P
-> reset D
```

因此在 EEG 中它发生在每个新个体的适配阶段。当前 subject 新过滤出的样本在
本轮 base Self-Replay 完成后才进入 P，从下一轮 subject 起成为 replay。

### 15.5 EEG 的 D 是按容量还是按 subject 触发

当前 runner 不使用跨 subject 的固定 500/1000-epoch 触发，而使用 subject
boundary：一个 subject 的全部约 36-52 条 sequence（约 720-1040 epochs）组成
一个逻辑 D，加载完该 subject 后触发一次 expert/base/SCF。原因是要保留
BrainUICL 的逐个体适配和每个体 after-evaluation 协议。

更贴近原文的替代方案可以设置固定 `D=1000 epochs`，但 D 可能横跨两个
subjects，且 subject 内可能触发多次，会改变现有 old/new 评价语义。因此当前
实现属于 subject-chunk SPR，而非严格 sample-count-trigger SPR。

### 15.6 P、base 和新数据集的关系

P 就是历史 replay memory。base NT-Xent 不产生新数据集，也不修改标签；它
只在当前 D 和从 P 采样的历史数据上更新长期 representation。P 只有经过 SCF
接纳和 reservoir replacement 时才改变。

base 必须再次做 NT-Xent，因为 expert 是局部且会重置，不能承担长期知识；
base 需要把当前 distribution 融入历史 representation，并通过 P 避免只拟合
当前块而遗忘过去。

### 15.7 SPR expert 是否生成伪标签

不生成。NT-Xent 只能产生 embedding，不产生 W/N1/N2/N3/REM 类别。原 SPR
直接使用输入流已有标签；无标签 EEG 必须由 guiding model 或其他 classifier
生成伪标签。expert 只用这些标签做 class graph 分组，标签不进入 NT-Xent。

### 15.8 Supervised fine-tune 是什么

原 SPR 从 base 复制 feature extractor，增加分类头，只用 P 中 `(x,y)` 做监督
交叉熵训练。D 中未过滤的数据不会直接用于该分类 fine-tune。当前 EEG 也只用
P，但加载完整 sequence，并通过 retained mask 仅对保留 epoch 计算 loss。

在 EEG 协议中，每处理完一个 new subject 都执行一次 fine-tune 和 old/new
评价；原论文通常在任务 checkpoint 执行 test phase。

### 15.9 原论文如何划分和验证新旧知识

原论文不是“没有划分数据”，而是没有 BrainUICL 式 source/new/old subject
划分。它使用标准训练集构造 noisy non-IID stream，并保留干净测试集：

- MNIST、CIFAR-10：五个顺序 task，每个 task 为一对类别；
- CIFAR-100：二十个 task，每个 task 五类；
- WebVision：选取十四类，组成七个顺序 task；
- shared output head，训练和测试时都不提供 task ID；
- 报告所有 task 流结束后的 clean-test overall accuracy；
- 在 task progression 中持续评价整体准确率，并观察第一任务 T1 的遗忘；
- 主要结果取五个随机 seed 的均值。

这里 earlier tasks 对应 old knowledge，当前或后续 task 对应 new knowledge；
它没有单独的 old-generalization subject 集。EEG 迁移才把这种 task progression
重新定义为按 subject 到达，并额外设置固定 old subjects 评价泛化稳定性。

## 16. 扩展到完整 49 个 new subjects

### 16.1 扩展设置

为检查 10-subject probe 是否过于乐观，实验进一步扩展到 split 中全部 49 个
new subjects。source/train、val、19 个 old/generalization subjects、seed、模型、
10-epoch 预算和评价协议均不变，只把 `--max-subjects` 从 10 改为 0。clean 与
40% symmetric noise 均重新从同一预训练 checkpoint 独立运行，同时补跑完整
49-subject BrainUICL 对照。

本地输出目录（继续由 `.gitignore` 排除）为：

```text
experiments/rttdp_brainuicl_runs/pure_spr_full49_e10_seed4321/
experiments/rttdp_brainuicl_runs/brainuicl_full49_e10_noise40_seed4321/
```

四组运行均完成 49/49 subjects；每条 stability 曲线有 initial 加 49 次适配共
50 个点。纯 SPR 在整个流中始终保持 3000 source + 2000 dynamic epoch，没有
超过 5000-epoch memory 上限。

### 16.2 完整流结果

| Method | Noise | Old ACC | Old MF1 | AAA | AAF1 | FR | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL, 49 subjects | 0% | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.0649 | 0.6182 | 0.5548 |
| Pure SPR-EEG, 49 subjects | 0% | **0.7126** | **0.6923** | **0.7147** | **0.6933** | **0.0145** | **0.6864** | **0.6167** |
| BrainUICL, 49 subjects | 40% | 0.6650 | 0.6249 | 0.6876 | 0.6614 | 0.0534 | 0.6104 | 0.5517 |
| Pure SPR-EEG, 49 subjects | 40% | **0.7100** | **0.6840** | **0.7254** | **0.7027** | **0.0107** | **0.6702** | **0.6028** |

完整流上，纯 SPR clean 的最终 old ACC/MF1 比 BrainUICL 高 5.57/6.92 个
百分点，40% noise 下高 4.50/5.91 个百分点。New after ACC/MF1 的平均提升
分别为 clean 6.82/6.18 点、40% noise 5.97/5.11 点。

纯 SPR 自身从 clean 到 40% noise 时，最终 old ACC 只下降 0.26 点，MF1
下降 0.84 点；new after ACC/MF1 分别下降 1.63/1.39 点。40% noise 的 AAA
高于 clean，但最终点略低，说明这是中间 stability trajectory 的差异，不能
解释为“加入标签噪声会提高性能”。

### 16.3 49-subject 过滤诊断

| Diagnostic | Clean | 40% noise |
| --- | ---: | ---: |
| Delayed epochs | 42960 | 42960 |
| Injected noisy epochs | 0 | 17195（40.03%） |
| Accepted epochs | 26954 | 23451 |
| Aggregate acceptance rate | 62.74% | 54.59% |
| Accepted noisy epochs | 0 | 5226 |
| Injected noise removed | - | **69.61%** |
| Accepted-set purity | 100% | 77.72% |
| Final dynamic buffer purity | 100% | **85.45%** |
| Final total buffer purity | 100% | 94.18% |

与 10-subject 结果相比，噪声过滤率从 68.16% 变为 69.61%，最终 dynamic
purity 从 85.55% 变为 85.45%。这说明固定容量 buffer 经约五倍长度的数据流
和持续 replacement 后，纯度没有明显衰减。40% noise 最终 old ACC 从
10-subject 的 0.7324 降到 49-subject 的 0.7100，说明前 10 个 subject 的最终
点偏乐观；扩大实验是必要的。

### 16.4 按 subject 配对检验

对 49 个 subject 的 new-after 指标计算 paired difference；置信区间使用固定
seed 的 20000 次 subject bootstrap，p 值使用双侧 Wilcoxon signed-rank test：

| Noise | Metric | SPR - BrainUICL | 95% bootstrap CI | Wins | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: |
| 0% | ACC | +6.82 points | [3.91, 10.02] | 35/49 | 5.87e-5 |
| 0% | MF1 | +6.18 points | [3.86, 8.66] | 37/49 | 1.09e-5 |
| 40% | ACC | +5.97 points | [2.57, 9.52] | 33/49 | 0.0021 |
| 40% | MF1 | +5.11 points | [2.49, 7.85] | 37/49 | 0.0002 |

这些配对结果比单独比较最后一个 old-ACC 点更稳定，但仍然只有一个
`seed=4321` split，且两种方法的监督条件和 memory unit 不同。因此它们支持
“在当前固定协议下扩大到 49 subjects 后优势仍存在”，不构成跨 split、跨 seed
或无标签场景下的普遍统计结论。

### 16.5 扩展实验复现命令

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/spr_eeg_pure.py \
  --output-root experiments/rttdp_brainuicl_runs/pure_spr_full49_e10_seed4321 \
  --max-subjects 0 --noise-rates 0.0 0.4 \
  --expert-epochs 10 --base-epochs 10 --ft-epochs 10
```

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/rttdp_brainuicl_full.py \
  --output-root experiments/rttdp_brainuicl_runs/brainuicl_full49_e10_noise40_seed4321 \
  --max-subjects 0 --ssl-epoch 10 --incremental-epoch 10 --cross-epoch 2 \
  --batch 16 --attack-mode buffer_label_noise --buffer-label-noise-rate 0.40 \
  --no-save-checkpoints
```

## 17. 随机初始化与空 Purified Buffer 版本

### 17.1 为什么需要单独实验

此前纯 SPR-EEG 复用了 BrainUICL source-pretrained model，并预装 3000 个受保护
source epochs。原 SPR 则从随机 base/expert 和空 D/P 开始，因此预训练模型会
显著改变初始 representation、SCF 特征质量和 old stability。为分离这一影响，
新增独立目录：

```text
experiments/spr_eeg_random_init/
```

该版本不读取 checkpoint，P 从空开始，source/new epoch 都必须经过同一
expert/base/SCF 和 fixed-capacity replacement，不存在受保护 source partition。

### 17.2 更合适的数据划分

每个参与适配的 subject 在 sequence 层面固定拆为 80% adaptation 和 20%
held-out evaluation，避免同一 sequence 同时用于更新 P 和报告 new-subject
性能。实现两种协议：

1. Task progression：全部 subjects 随机排序为顺序任务，持续评价当前 subject
   held-out sequences 和最早 anchor subjects，直接测 plasticity/forgetting。
2. Fixed subject split：保留 BrainUICL train/val/old/new 划分，但 train subjects
   不做 supervised pretraining，而是作为普通 SPR chunks 逐个 warmup；old
   subjects 永不训练，每个 new subject 适配后评价固定 old 与其 held-out 数据。

原 SPR 需要观测标签，所以该随机初始化实现使用 ISRUC observed labels，可选
对称加噪。随机分类器无法产生有意义的五类伪标签，完全无标签随机起步仍需要
额外 clustering、少量标注 bootstrap 或 pretrained guiding model。

### 17.3 10-subject clean 验证

配置为 expert/base/fine-tune 各 10 epochs，seed 4321，P capacity 5000。

| Protocol/Stage | ACC | MF1 |
| --- | ---: | ---: |
| Progression random initial | 0.1270 | 0.0419 |
| Progression before adaptation | 0.5406 | 0.4220 |
| Progression after adaptation | 0.6679 | 0.5363 |
| Fixed split random old | 0.1117 | 0.0402 |
| Fixed split after 24 source SPR chunks | 0.6545 | 0.6239 |
| Fixed split after 10 new subjects | 0.6868 | 0.6653 |

fixed-split 中 new held-out ACC 从适配前平均 0.5832 提升到适配后 0.6467，MF1
从 0.4583 提升到 0.5423。P 从空开始，在第 14 个 source chunk 达到 5000
epoch，之后全部通过无 source 保护的 class-aware replacement 更新。

同预算 source-pretrained SPR 的 10-subject old ACC/MF1 为 0.7123/0.6881，
随机初始化加 24-subject SPR warmup 后为 0.6868/0.6653，说明预训练仍有约
2.55/2.29 个百分点优势。但两者 new 指标不能直接比较：随机版本使用严格
held-out sequence，早期 oracle runner 在同一 subject 数据上适配和评价。

完整命令和结果说明见 `experiments/spr_eeg_random_init/README.md` 与
`experiments/spr_eeg_random_init/RESULTS.md`。

### 17.4 全部个体 progression 结果

完整实验在 progression 协议中处理全部 98 个 subjects，每个 subject 均采用
80% adaptation/20% held-out sequence，clean 和 40% noise 均完成 98/98。

| Noise | Before ACC | After ACC | Before MF1 | After MF1 | Final P purity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.6438 | 0.6909 | 0.5256 | 0.5593 | 100% |
| 40% | 0.6110 | 0.6295 | 0.4832 | 0.4921 | 61.34% |

clean 下当前个体适配平均提升 ACC 4.71 点、MF1 3.37 点；40% noise 下仍提升
1.85/0.89 点，说明存在 plasticity，但噪声明显压缩适配收益。

clean 流结束后，最早三个 anchor subjects 的 ACC 为 0.6000、0.7500、0.7056，
均值 0.6852；它们首次学习后的均值为 0.6146，但历史 peak 均值为 0.7949。
因此相对首次学习没有整体遗忘，反而受后续共享表示帮助；相对各自最佳时刻仍
遗忘 10.97 点。40% noise 下 anchor 最终均值 0.6507，相对 peak 遗忘 13.56 点。

这表明随机初始化 SPR 具有持续适配和正向 transfer，但固定容量 P replacement
仍会丢失部分个体最优知识，标签噪声会进一步加重 peak-to-final forgetting。

### 17.5 完整 fixed-split 与 BrainUICL 对比

fixed-split 完整处理 24 个 source SPR warmup chunks 和全部 49 个 new subjects，
19 个 old subjects 始终只用于评价。

| Method | Noise | Final old ACC | Old MF1 | AAA | AAF1 | Last-10 ACC | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | 0% | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.6969 | 0.6182 | 0.5548 |
| Random-init SPR | 0% | 0.6663 | 0.6291 | 0.7061 | 0.6837 | 0.6971 | 0.6804 | 0.5441 |
| BrainUICL | 40% | 0.6650 | 0.6249 | 0.6876 | 0.6614 | 0.6891 | 0.6104 | 0.5517 |
| Random-init SPR | 40% | 0.7201 | 0.6754 | 0.6539 | 0.6208 | 0.6622 | 0.6176 | 0.4623 |

clean 最终点中 Random-init SPR 比 BrainUICL 高 0.95 ACC 和 0.60 MF1 点，AAA/
AAF1 高 1.27/1.52 点；但最后 10 步 old ACC 只高 0.02 点，可视为长期稳定性
基本相同。Random-init SPR 的 new held-out adaptation 平均带来 +4.09 ACC 和
+4.01 MF1 点，而 BrainUICL 的全 subject 评价中 after-before 为 -1.22 ACC 和
约 0 MF1。

40% noise 的 Random-init SPR 最终 old ACC 看似高 5.51 点，但这个结论被最后
一个 subject 放大：其 AAA 比 BrainUICL 低 3.37 点，最后 10 步均值低 2.69 点。
New after ACC 只高 0.71 点，MF1 反而低 8.94 点，说明随机初始化 noisy stream
产生严重类别不平衡，不能仅用最终 ACC 宣称防护更强。

两者监督条件也不同：Random-init SPR 使用 observed labels 并在 held-out
sequences 上评价；BrainUICL 使用 teacher pseudo-label，原实验在当前 subject
全部 sequences 上评价。因此这里只能比较协议表现，不能视为完全公平的算法
显著性对照。

### 17.6 全流过滤结论

40% fixed stream 共 50,760 个 epoch，其中 20,282 个被加噪。SCF 接受 28,536
个 epoch，过滤掉 9,049 个 noisy epochs，即 injected-noise removal 44.62%。
最终 P purity 仅 61.60%，只略高于输入约 60% 的 clean 比例。

此前 source-pretrained SPR 在 40% noise 下 dynamic P purity 为 85.45%。因此
预训练的主要价值不只是提高初始分类 ACC，更重要的是提供有语义的 expert
特征，使 SCF 能形成可靠 class graph。随机 expert 在仅 10 epochs 的预算下
过滤能力明显不足，这是随机初始化版本在 noisy continual learning 中的核心
瓶颈。

## 18. EEG 迁移实验端到端流程

### 18.1 输入、数据划分与模型状态

ISRUC 每个 30 秒 epoch 包含 2 个 EOG、6 个 EEG channels 和 3000 个采样点；
连续 20 个 epoch 组成一条约 10 分钟 sequence，输入 shape 为 `[20,8,3000]`，
输出为 W/N1/N2/N3/REM 二十个 epoch 标签。

fixed-split 使用 24 source、6 validation、19 old/generalization 和 49 sequential
new subjects。随机初始化 progression 还提供全部 98 subjects 的 80% adaptation/
20% held-out sequence 协议。NT-Xent 的训练单位是完整 sequence；SCF 顶点和
P 保存单位是 30 秒 epoch reference。

### 18.2 每个 new subject 的处理链

```text
new-subject EEG sequences
-> observed label 或 guiding pseudo-label
-> 当前 subject 构成逻辑 Delayed Buffer D
-> expert 在 D 上做 EEG NT-Xent
-> base 在 D + 历史 P 上做 NT-Xent Self-Replay
-> expert 提取每个 epoch embedding
-> 按 W/N1/N2/N3/REM 分成五个 class graphs
-> stochastic cosine graph ensemble (E_max=5)
-> eigenvector centrality
-> Beta Mixture clean posterior
-> clean_p > Uniform(0,1) 随机接纳
-> class-aware fixed-capacity P replacement
-> 从 base 复制 inference model
-> 只用 P retained epoch mask 做监督 fine-tune
-> current new / fixed old / earliest anchors 评价
-> 下一个 subject
```

当前 subject 通常有 36-52 条 sequence，即约 720-1040 个 epoch。当前实现按
subject boundary 触发一次 expert/base/SCF，以保持 BrainUICL 逐个体评价语义，
不是严格跨 subject 累积固定 D 容量。

### 18.3 Expert、Base、P 和 evaluation model

expert 是每个 D 的局部过滤 representation，处理下一 D 时重建；base 是跨整条
流持续存在的长期 representation。Base NT-Xent 不生成标签、样本或新数据集，
只更新 feature extractor、Transformer 和 projection head 参数，使同一 EEG
sequence 的两个增强 view 接近、不同 sequence 分离，并通过 P replay 缓解遗忘。

P 保存 `(sequence_path, epoch_index, label, clean_p)`。训练时仍读取完整
20-epoch sequence 作为 Transformer 上下文，但未保留 epoch 使用
`ignore_index=-100`，不参与监督交叉熵。P 满后只替换 epoch reference，不删除
磁盘原始 sequence。

### 18.4 三种实验边界

| Variant | Initialization | New-subject label | Initial P |
| --- | --- | --- | --- |
| BrainUICL | source-pretrained | guiding pseudo-label | 完整 sequence replay |
| Oracle SPR-EEG | source-pretrained/random | observed/noisy label | source split 或 empty |
| Unlabeled SPR-EEG | source-pretrained | guiding pseudo-label | sampled labeled source P |

oracle 实验用于验证原 SPR 的 noisy labeled stream 假设；真实无标签部署必须由
guiding model 产生伪标签。new-subject ground truth 只能用于 ACC/MF1、伪标签
error 和 P purity 诊断，不得参与 NT-Xent、SCF 分组、接纳或 fine-tune 标签。

### 18.5 评价

每个 new subject 记录 source/random initial、适配前 before 和适配后 after 的
ACC/MF1；每一步在固定 old subjects 上计算 ACC、MF1、AAA、AAF1 和 FR。
progression 协议额外记录最早 anchors 的 first、peak、final，区分正向 transfer
与 peak-to-final forgetting。ACC 衡量总体正确率，MF1 对五类等权，更能暴露
N1 等少数类退化。

## 19. 无标签 guiding-pseudo-label SPR-EEG

### 19.1 实现边界

独立实现位于 `experiments/spr_eeg_unlabeled/`。它保留 source-pretrained model
和 3000 个有标签 source epochs，但 new-subject ground truth 不参与训练决策。
每个 subject 的训练流程为：

```text
previous inference model
-> 当前 subject 上 CPC guiding adaptation
-> 为全部 epoch 生成 pseudo-label 和 confidence
-> 不使用 0.9/15-of-20 gate，全部 epoch 进入 D
-> source-initialized expert NT-Xent
-> continual base D+P NT-Xent Self-Replay
-> expert epoch embedding + pseudo-label SCF
-> accepted epoch reference + 原 pseudo-label 进入 P
-> base copy + retained-mask P fine-tune
-> old/new evaluation
```

confidence 只记录诊断，不进入候选或接纳条件。ground truth 在 SCF 完成后只用于
计算 pseudo-label error 和 P purity；测试验证改变 diagnostic labels 不会改变
accepted epoch。

### 19.2 Base NT-Xent 的输出

Base NT-Xent 不生成伪标签、EEG 样本或新数据集。它更新长期存在的 feature
extractor、Transformer encoder 和 projection head 参数，输出可用于后续任务的
representation。两个增强 view 的同一 sequence 被拉近，不同 sequence 被推远；
P replay 让 representation 同时保留历史 subject 信息。分类标签来自 guiding
model，分类能力来自复制 base 后在 P 上做的监督 fine-tune。

### 19.3 10-subject 正式结果

配置为 guiding CPC、expert、base 和 fine-tune 各 10 epochs，seed 4321。

| Method | Old ACC | Old MF1 | AAA | AAF1 | FR | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL, 10 epochs | 0.6676 | 0.6247 | 0.7076 | 0.6837 | 0.0496 | 0.5946 | 0.5235 |
| Unlabeled SPR-EEG | **0.6898** | **0.6615** | 0.6996 | 0.6724 | **0.0180** | **0.6195** | **0.5471** |
| Oracle-label SPR-EEG | 0.7123 | 0.6881 | 0.7153 | 0.6954 | 0.0140 | 0.6424 | 0.5746 |

无标签 SPR 比 BrainUICL 的最终 old ACC/MF1 高 2.22/3.68 点，new after ACC/MF1
高 2.48/2.36 点，但 AAA/AAF1 低 0.80/1.12 点，说明最终稳定性改善但中间轨迹
不占优。相比 oracle-label SPR，伪标签使 old ACC/MF1 低 2.25/2.67 点，new
after ACC/MF1 低 2.29/2.75 点。

### 19.4 无 confidence gate 的过滤效果

| Diagnostic | Value |
| --- | ---: |
| 全部 candidate epochs | 8880 |
| confidence < 0.9 但仍进入 SCF | 2511 |
| accepted epochs | 5451 |
| acceptance rate | 61.39% |
| SCF 前 pseudo-label error | 33.07% |
| accepted-set error | 27.92% |
| final dynamic P purity | 76.20% |
| final total P purity | 90.48% |

SCF 在不做 confidence 硬删除时仍降低了伪标签错误，但无法修复大规模 coherent
wrong cluster。例如 subject 26 的伪标签错误率为 62.02%，过滤后仍有 59.93%。
这说明下一步应引入 source-anchor distance、temporal consistency 或 class quota，
而不是重新启用会大量删除数据的硬 confidence gate。

### 19.5 完整 49-subject 无标签结果

训练 batch 保持 8，只把不影响训练的 evaluation batch 从 16 提高到 32。RTX
4070 SUPER 上完整 49 subjects 用时约 13.8 分钟，全部步骤和 50 个 old
stability checkpoints 均完成。

| Method | Old ACC | Old MF1 | AAA | AAF1 | Last-10 ACC | New after ACC | New after MF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.6969 | 0.6182 | 0.5548 |
| Unlabeled SPR-EEG | **0.7109** | **0.6815** | **0.7037** | **0.6766** | **0.7066** | **0.6694** | **0.5951** |
| Oracle-label SPR-EEG | 0.7126 | 0.6923 | 0.7147 | 0.6933 | 0.7210 | 0.6864 | 0.6167 |

无标签 SPR 相比 BrainUICL 的 final old ACC/MF1 高 5.40/5.84 点，AAA/AAF1
高 1.03/0.81 点，最后 10 步 old ACC 高 0.97 点；new after ACC/MF1 高
5.12/4.03 点。按 subject 配对，new-after ACC 差值 95% bootstrap CI 为
2.56-7.95 点、Wilcoxon `p=0.0013`；MF1 CI 为 1.96-6.17 点、`p=0.0016`，
两个指标均在 34/49 个 subjects 上获胜。

相比 oracle-label SPR，无标签版本 final old ACC 只低 0.17 点，但 old MF1、
new ACC、new MF1 分别低 1.08、1.70、2.16 点；AAA/AAF1 低 1.10/1.68 点，
说明伪标签主要损失少数类和中间 trajectory，而不是最终总体 ACC。

### 19.6 完整流伪标签与 P 诊断

| Diagnostic | Value |
| --- | ---: |
| 全部 candidate epochs | 42960 |
| confidence < 0.9 但仍进入 SCF | 12243 |
| accepted epochs | 26934 |
| acceptance rate | 62.70% |
| weighted pseudo-label error before | 30.09% |
| weighted accepted-set error | 26.35% |
| error 得到改善的 subjects | 45/49 |
| final dynamic P purity | 77.95% |
| final total P purity | 91.18% |

SCF 在无 confidence gate 下把加权伪标签错误率降低 3.74 点。subjects 26、98、
2、15 的 accepted-set error 反而上升，说明 coherent wrong cluster 仍是主要失败
模式。总体上，guiding pseudo-label + SCF 在完整流上同时改善了 BrainUICL 的
old stability 和 new plasticity，但距离 oracle 的 MF1 与 trajectory 仍有差距。

### 19.7 Random-init 与 extra-noise 完整矩阵

上一节的无标签主实验使用自然伪标签错误，没有额外人工噪声。为补齐矩阵，
另运行 49-subject `extra_pseudo_noise=0.40`：guiding 先产生伪标签，再以 40%
概率替换为不同类别。它叠加在自然伪标签错误上，因此总错误率不是 40%。

| Variant | Label source | Old ACC | Old MF1 | AAA | New after ACC | New after MF1 | Dynamic P purity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BrainUICL | pseudo + confidence gate | 0.6569 | 0.6231 | 0.6934 | 0.6182 | 0.5548 | sequence buffer |
| Oracle SPR | observed clean | 0.7126 | 0.6923 | 0.7147 | 0.6864 | 0.6167 | 100% |
| Unlabeled SPR | natural pseudo | 0.7109 | 0.6815 | 0.7037 | 0.6694 | 0.5951 | 77.95% |
| Unlabeled SPR + noise | pseudo + extra 40% | 0.6953 | 0.6694 | 0.7018 | 0.6405 | 0.5767 | 63.90% |
| Random-init SPR clean | observed clean | 0.6663 | 0.6291 | 0.7061 | 0.6804 | 0.5441 | 100% |
| Random-init SPR 40% | observed + 40% | 0.7201 | 0.6754 | 0.6539 | 0.6176 | 0.4623 | 61.60% |

extra-noise 流共 42,960 epochs，实际新增 17,251 次标签替换。有效伪标签错误率
为 55.49%，SCF 后 accepted-set error 为 41.05%；最终 dynamic P purity 63.90%。
相对自然伪标签版本，old ACC/MF1 降 1.56/1.21 点，new after ACC/MF1 降
2.89/1.83 点。说明 SCF 能过滤大量独立随机错标，但 P purity 和 plasticity
仍随有效噪声显著恶化。

Random-init 两行使用 observed labels，且 new 指标来自 20% held-out sequences；
其余三种 SPR/BrainUICL 使用原 full-subject transductive 协议。该统一表用于避免
遗漏实验，不应把不同监督条件的行解释为严格公平排名。

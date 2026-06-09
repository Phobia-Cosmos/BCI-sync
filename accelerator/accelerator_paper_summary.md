# DNN/LLM/BCI Accelerator Papers Summary

整理目录: `/home/undefined/Desktop/bci/papers/accelerator`

## 1. 已下载的有效 PDF

| 论文 | 主线 | 核心思想 | 实验/落地方式 | 和 BCI/模型适配的关系 |
|---|---|---|---|---|
| 2016ISCA-EIE Efficient Inference Engine on Compressed Deep Neural Network | 稀疏 DNN 推理 | 压缩权重 + 稀疏激活, 面向 sparse matrix-vector | Verilog RTL + cycle-accurate simulator + 45nm synthesis/layout estimate, 不是 PC 插卡 | 适合借鉴到小模型稀疏推理, 不覆盖训练/适配反传 |
| 2016ISCA-Cnvlutin Ineffectual-Neuron-Free Deep Neural Network Computing | CNN 稀疏激活 | 跳过无效 neuron/activation 计算 | 架构建模/仿真 + synthesis estimate | 可迁移到 EEGNet/CNN decoder 的零值/低幅值跳过 |
| 2017ISCA-SCNN An Accelerator for Compressed-sparse CNNs | 稀疏 CNN | 压缩权重和激活, sparse Cartesian product dataflow | HLS/SystemC/cycle-level simulator + 16nm synthesis estimate | 适合 CNN decoder 推理, 对在线适配只覆盖前向 |
| 2018ISCA-Bit Fusion | 可变精度 DNN | bit-level 可组合 PE, 按层/按操作动态调整 bitwidth | Verilog RTL + cycle-accurate simulator + 45nm synthesis | EEG/BCI 小模型可用低 bit 推理和 adapter 更新 |
| 2020MICRO-Gobo | Transformer/NLP 量化 | outlier-aware quantization, 降低 attention-based NLP 推理延迟/能耗 | 硬件感知量化 + accelerator modeling | 主要面向 Transformer, 可借鉴到 EEG foundation model 的量化 |
| 2021arXiv-Dual-side Sparse Tensor Core | 稀疏 Tensor Core | 权重和激活双边稀疏 Tensor Core | GPU Tensor Core 架构建模/仿真 | 适合大矩阵 GEMM, 对小 EEG 模型收益取决于 batch/矩阵规模 |
| 2021HPCA-SpAtten | Attention 加速 | token/head pruning + progressive quantization | accelerator architecture + simulator/synthesis estimate | 可迁移到 EEG/BCI Transformer 的注意力剪枝 |
| 2021ISCA-ELSA | 轻量 self-attention | hardware-software co-design, 降低 self-attention 复杂度 | accelerator modeling + synthesis estimate | 适合资源受限 Transformer decoder/foundation model 部署 |
| 2021MICRO-Sanger | 稀疏 attention | reconfigurable sparse attention dataflow | accelerator simulation + synthesis estimate | 对 EEG Transformer 的稀疏 attention 有直接参考价值 |
| 2022MICRO-Ant | 自适应数值类型 | adaptive numerical type + dataflow co-design | Verilog RTL + 28nm synthesis + cycle-accurate simulator | 适合端侧低 bit 推理, 也可用于 adapter-only 低 bit 更新 |
| 2023HPCA-Sibia | signed bit-slice | signed bit-slice architecture for dense DNN | RTL Verilog + RTL simulation + synthesis | 可用于 dense GEMM/conv, 适合低功耗 decoder |
| 2023ICML-SmoothQuant | LLM 量化 | 将 activation outlier 平滑迁移到 weight, 便于 INT8 | 软件/系统评估, 不是硬件加速器论文 | 更像部署前量化方法, 可服务 EEG foundation model |
| 2023ISCA-Olive | LLM 量化加速 | outlier-victim pair quantization, 硬件友好地处理 outlier | ISCA 架构论文, hardware-aware evaluation | 可迁移到带 outlier 的 EEG foundation model/Transformer 推理 |
| 2024ICML-KIVI | KV cache 量化 | KV cache asymmetric 2-bit quantization | 主要是软件/系统评估, 不是独立硬件加速器 | 仅当 BCI 用长序列 Transformer/LLM-like decoder 时相关 |
| 2024MLSys-AWQ | on-device LLM 量化 | activation-aware weight quantization | 量化算法 + TinyChat/端侧部署评估, 不是新硬件 | 适合端侧 foundation model 压缩 |
| 2025HPCA-NeuVSA | neural vector search | 统一 neural vector search accelerator | HPCA 架构论文, 面向 ANN/vector search workload | 不属于标准 DNN 推理, 但可借鉴到 EEG/fMRI embedding 检索和跨被试相似性搜索 |
| 2025HPCA-Panacea | DNN bit-slice + asymmetric quantization | accuracy-preserving asymmetric quantization + bit-slice sparsity | 28nm ASIC-level implementation/synthesis estimate | 与低 bit DNN decoder/adapter 更新最相关 |
| 2025ISCA-Oaken | LLM serving/KV cache | online-offline hybrid KV cache quantization + Q/DQ engine + MMU | SystemVerilog RTL for engines + TSMC 28nm synthesis + hardware simulator | 对实时长序列模型的 memory bandwidth 优化有参考价值 |
| 2025ISCA-Transitive Array | GEMM 加速 | result reuse for GEMM | accelerator paper, simulation/synthesis-style evaluation | 可迁移到重复矩阵块明显的 feature/decoder 计算 |
| 2025TC-DSTC | GPU Tensor Core 稀疏 | dual-side sparsity Tensor Core for GPU | GPU/Tensor Core 架构建模/仿真, 不是独立芯片 | 更适合 GPU/NPU 架构改造, 不适合直接落到 MCU |
| 2026HPCA-Uni-STC | sparse Tensor Core | unified sparse tensor core supporting multiple sparse ops | C++ simulator + synthesis/artifact evaluation | 稀疏矩阵/图/Transformer 通用, 可借鉴到统一 EEG workload 加速 |
| 2021ICCD-MasterMind | BCI SoC | RISC-V + 多个 Sinkhorn/SVD accelerator, 加速 HiWA 脑信号对齐 | FPGA prototype + ASIC feasibility analysis | 直接说明 BCI 模型适配/对齐算法可以硬件化 |
| 2023ISCA-SCALO | implantable BCI distributed system | accelerator-rich distributed BCI, compute/network/storage 协同 | system-level architecture + physical synthesis/network/storage model | 直接说明 BCI 端侧/植入式实时处理需要系统级加速 |
| 2025-A high performance heterogeneous hardware architecture for BCI | EEG/SSVEP BCI | ARM + FPGA, EEGNet 量化/层融合/硬件 PE | Xilinx Zynq FPGA real system | 非侵入式 BCI 端侧 CNN decoder 硬件落地案例 |
| 2025HPCA-Prosperity | SNN accelerator | product sparsity for spiking GeMM | RTL/synthesis + simulator, 开源 | SNN 在低功耗 BCI 中有潜力, 但不是传统 EEGNet 路线 |
| 2026DAC-FILCO | DNN flexible accelerator | real-time reconfigurable composition for diverse DNN workloads | DAC 架构论文, 面向 reconfigurable accelerator | 对“一个 BCI 设备多任务多模型”很相关 |

## 2. 下载失败或未找到公开 PDF

| 论文 | 状态 | 说明 |
|---|---|---|
| 2016MICRO-Cambricon-X An Accelerator for Sparse Neural Networks | 下载失败 | 作者 PDF 源返回 502/SSL EOF; IEEE staging 直链也失败。目录中保留 `.download-failed` 标记文件 |
| 2024SC-MixQ Taming Dynamic Outliers in Mixed-Precision Quantization by Online Prediction | 未下载 | 找到作者 HTML 页面和 ACM/SC 条目, 未找到稳定公开 PDF |
| 2024HPCA-LUTein Dense-Sparse Bit-Slice Architecture with Radix-4 LUT-Based Slice-Tensor Processing Units | 未下载 | 找到题名/会议页面, 未找到稳定公开 PDF |
| 2025ISCA-AiF Accelerating On-Device LLM Inference Using In-Flash Processing | 未下载 | 找到作者/ACM 条目, 未找到公开 PDF; ACM 访问需要权限 |
| 2025TCAS-I-OFQ-LLM Outlier-Flexing Quantization for Efficient Low-Bit LLM Acceleration | 未下载 | 找到题名/出版条目, 未找到稳定公开 PDF |

## 3. Model adaptation 过程中主要计算类型

模型适配不是单纯 inference。它通常包含这些计算:

| 阶段 | 典型计算 | 对硬件的压力 |
|---|---|---|
| target data forward | conv/GEMM/attention/normalization/activation | 和推理类似, 可复用 DNN accelerator |
| pseudo-label/confidence | softmax/argmax/top-k/entropy/filtering | 控制流多, 但计算量较小 |
| distribution alignment | MMD/CORAL/Wasserstein/feature covariance/mean-variance | 统计量、矩阵乘、距离计算, 对 SRAM 和归约友好 |
| supervised or self-training loss | cross entropy/entropy minimization/KL/contrastive loss | 小规模标量/向量计算 |
| partial backward | head/adapter/LoRA/BN 参数的梯度 | 和推理不同, 需要 activation buffer、transpose GEMM、reduction |
| optimizer update | SGD/Adam/AdamW/momentum/weight decay | memory-bound, 对状态读写敏感 |
| normalization/stat update | BN running mean/var, layer norm stats, calibration stats | 低算力但强流式, 适合硬件归约 |

BCI/SFDA/cross-subject adaptation 中最常见的是: source model forward target EEG/fMRI, 生成伪标签或目标域特征统计, 做 feature alignment, 然后只更新 classifier head、BN/statistics、adapter 或低秩参数。full fine-tuning 成本高, 也更容易泄露隐私和引入不稳定延迟。

## 4. 模型适配是否有优化空间

有, 而且和传统 DNN 推理加速的优化点不完全一样。

| 优化方向 | 含义 | 适合硬件化吗 |
|---|---|---|
| freeze backbone | 主干只前向, 只更新 head/adapter | 很适合; 复用推理加速器 + 小型更新单元 |
| BN/stat-only adaptation | 只更新均值/方差/校准统计 | 很适合; 流式归约 + 少量寄存器状态 |
| LoRA/adapter update | 低秩小矩阵更新 | 适合; 小 GEMM + optimizer state |
| sparse gradient/update | 只更新高置信/高影响参数 | 适合, 但需要索引和调度 |
| quantized adaptation | 低 bit forward/backward/update | 有潜力, 但训练数值稳定性比推理更难 |
| sample selection | 低置信样本跳过或延迟更新 | 可节省算力, 但可能引入 timing side channel |
| early-stop/periodic adaptation | 不每个 window 都更新 | 适合实时系统, 但需要延迟/鲁棒性约束 |
| feature-cache adaptation | 保存中间特征而非 raw EEG | 可省带宽, 但缓存特征仍可能泄露隐私 |

因此, 面向 BCI 的新硬件点不应该只做“再造一个 EEGNet 推理 accelerator”。更有价值的是做一个 secure adaptation engine: 主干固定低功耗推理, 旁边有小型统计/伪标签/adapter-update 单元, 并且限制更新时间、访存模式和可观察输出, 避免隐私泄露和 timing leakage。

## 5. DNN accelerator 论文的实验通常怎么做

这些论文大多数不是“做一块 PCIe 卡插到 PC 上跑”。常见评估路径如下:

| 类型 | 代表论文 | 说明 |
|---|---|---|
| RTL + synthesis + cycle simulator | EIE, BitFusion, Ant, Sibia, Panacea, Oaken, Uni-STC | 设计 RTL 或关键模块, 用 Synopsys/Cadence/Yosys 等综合到 45/28/16/7nm, 再用 simulator 估算吞吐/能耗/面积 |
| HLS/FPGA prototype | MasterMind, ARM+FPGA BCI EEGNet paper | 在 FPGA/Zynq 上真实跑, 更接近可用硬件, 但频率和能效不等于 ASIC |
| 架构模拟 GPU/Tensor Core 修改 | DSTC, Dual-side STC, Uni-STC | 假设修改 GPU tensor core, 用模拟器/模型评估, 不是当前 PC GPU 能直接使用 |
| 算法 + hardware-aware software evaluation | SmoothQuant, AWQ, KIVI | 主要证明量化/压缩算法可在 GPU/edge GPU/CPU 上提速, 不是新硬件架构 |
| 系统级 BCI 架构 | SCALO | 关注植入式/分布式 BCI 的 compute-network-storage-power trade-off, 不是单个 PE 微结构 |

如果你的目标是发体系结构/硬件方向, 仅在 PC/GPU 上跑模型通常不够。更像顶会论文的路径是: workload profiling -> 找到算子/访存/稀疏/实时瓶颈 -> 提出 dataflow/PE/memory hierarchy/调度机制 -> RTL 或 HLS 实现 -> simulator/synthesis/FPGA prototype -> 和 CPU/GPU/NPU/已有 accelerator 对比。

## 6. 对 EEG/BCI 的直接启发

1. 非侵入式 EEG 的实时链路通常是 streaming window: preprocessing -> feature extraction -> decoder -> feedback。推理计算通常不大, 真正难点是低功耗、稳定延迟、跨被试泛化、在线校准和隐私。
2. 如果只做固定 EEGNet 推理加速, 创新空间偏工程; 如果做 cross-subject/source-free/foundation-model adaptation 的低功耗安全加速, 更接近当前研究空白。
3. 模型适配阶段可能泄露隐私: source model 参数/BN stats/adapter gradients/pseudo-label confidence/更新时间都可能暴露目标被试特征。
4. 安全硬件切入点可以是: fixed-latency adaptation, oblivious sample selection, encrypted feature buffer, bounded-update adapter, poisoning/backdoor detector for calibration windows, privacy-aware statistics update。
5. 对 BCI 最可迁移的 accelerator 思路是低 bit quantization、小矩阵/小 batch GEMM、流式统计归约、adapter-only update、稀疏/跳过计算和实时 reconfigurable dataflow。

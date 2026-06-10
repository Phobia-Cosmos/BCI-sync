# ML/DL 硬件加速器与资源受限实时场景调研

更新时间：2026-06-08

## 1. 总体结论

近几年 ML/DL 硬件加速器的顶会主线明显分化：

1. 体系结构顶会（ISCA/MICRO/HPCA/ASPLOS）最热的是 LLM inference、KV cache、长上下文、RAG/vector search、GEMM、PIM/CIM、GPU-NPU 协同。
2. 低功耗、资源受限、实时场景更多出现在 ISSCC/VLSI/DAC/DATE/ISCAS、TinyML、边缘 AI、IoT 和部分期刊中，严格 ISCA/MICRO/HPCA 主会数量相对少。
3. 2026 年并不是没有相关论文，而是目前能确认的 2026 论文更多集中在 DAC/ISCAS/期刊/arXiv 和工业边缘 AI，体系结构顶会到 2026 年中仍以 LLM/边缘生成式 AI/安全计算为主。
4. 硬件加速器对安全和隐私的考虑仍然不是主流。已有工作主要关注 side-channel-resistant accelerator、secure in-memory computing、FHE/ZKP accelerator、secure embedding、GPU interconnect side-channel mitigation；很少直接把“训练数据隐私泄漏 / membership inference / 医疗传感数据隐私”作为 accelerator 的核心指标。

## 2. 代表性 ML/DL/LLM 加速器论文

| 年份 | 论文/系统 | Venue | 应用场景 | 核心内容 |
|---|---|---|---|---|
| 2024 | MECLA: Memory-Compute-Efficient LLM Accelerator with Scaling Sub-matrix Partition | ISCA | LLM inference | 用 scaling sub-matrix partition 降低 LLM 推理的 memory/compute 开销。 |
| 2024 | ALISA: Accelerating LLM Inference via Sparsity-Aware KV Caching | ISCA | LLM 长上下文 | 利用 sparsity-aware KV cache 优化长上下文推理。 |
| 2024 | Trapezoid: A Versatile Accelerator for Dense and Sparse Matrix Multiplications | ISCA | GEMM / sparse GEMM | 统一支持稠密和稀疏矩阵乘。 |
| 2024 | NeuraChip | ISCA | GNN | hash-based decoupled spatial accelerator，加速图神经网络。 |
| 2024 | Cambricon-LLM | MICRO | 大模型片上推理 | chiplet-based hybrid architecture，面向 70B 级别 LLM on-device inference。 |
| 2024 | VGA: Hardware Accelerator for Scalable Long Sequence Model Inference | MICRO | 长序列模型 | 加速 long sequence model inference。 |
| 2025 | Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization | ISCA | LLM serving | 在线/离线混合 KV cache 量化，减少 HBM/内存压力。 |
| 2025 | AiF: Accelerating On-Device LLM Inference Using In-Flash Processing | ISCA | 端侧 LLM | 在 flash 内处理，减少权重搬运和内存墙。 |
| 2025 | Hybe: GPU-NPU Hybrid System for Efficient LLM Inference with Million-Token Context Window | ISCA | 百万 token 长上下文 | GPU-NPU 混合系统，处理超长上下文推理。 |
| 2025 | Transitive Array: An Efficient GEMM Accelerator with Result Reuse | ISCA | GEMM | 利用 result reuse 提高 GEMM 能效。 |
| 2025 | REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing | ISCA | RAG / vector retrieval | 在存储侧做 retrieval，加速 RAG 系统。 |
| 2025 | Prosperity: Accelerating Spiking Neural Networks via Product Sparsity | HPCA | SNN / neuromorphic | 利用 product sparsity 加速 SNN，适合低功耗实时感知。 |
| 2025 | NeuVSA: A Unified and Efficient Accelerator for Neural Vector Search | HPCA | 向量检索 | 面向 ANN/vector DB 的专用加速器。 |
| 2025 | Panacea: Novel DNN Accelerator using Asymmetric Quantization and Bit-Slice Sparsity | HPCA | DNN inference | asymmetric quantization + bit-slice sparsity。 |
| 2025 | CogSys: Efficient and Scalable Neurosymbolic Cognition System | HPCA | neuro-symbolic AI | 同时支持 neural kernels 和 symbolic kernels，面向实时推理。 |
| 2025 | Fast On-device LLM Inference with NPUs | ASPLOS | 端侧 LLM | 利用 NPU 做端侧 LLM 加速。 |
| 2025 | PICACHU: Plug-In CGRA Handling Upcoming Nonlinear Operations in LLMs | ASPLOS | LLM nonlinear ops | 用 CGRA 处理 LayerNorm、GELU、Softmax 等非线性瓶颈。 |
| 2025 | Analytical Performance/Power Model + Fine-Grained DVFS for AI Accelerators | ASPLOS | AI accelerator 能效管理 | 用细粒度 DVFS 和功耗模型提升 accelerator 能效。 |

## 3. 资源受限、低功耗、实时场景论文

| 年份 | 论文/系统 | Venue/状态 | 场景 | 核心内容 |
|---|---|---|---|---|
| 2022/2023 | TinyVers | arXiv / SoC line | extreme edge | RISC-V + ML accelerator + eMRAM，面向 μW-mW 级别边缘推理。 |
| 2024 | RAMAN | IEEE IoT-J / ISCAS demo | TinyML / audio-visual | reconfigurable sparse TinyML accelerator。 |
| 2024 | On-device Learning of EEGNet-based Network for Wearable Motor Imagery BCI | ISWC/UbiComp Adjunct | 可穿戴 EEG | 在低功耗 RISC-V/GAP9 上运行/更新 EEGNet。 |
| 2025 | Heterogeneous NPUs Leveraging Analog In-Memory Computing for Edge AI | VLSI | edge AI | 数字 + 模拟 CIM 混合 NPU。 |
| 2025 | Adelia: A 4nm LLM Accelerator | VLSI | 生成式 AI inference | stream-lined dataflow + dual-mode parallelization。 |
| 2025 | ISSCC edge AI / tiny NPU papers | ISSCC | edge generative AI / always-on AI | 3nm tiny NPU、低延迟 transformer accelerator 等。 |
| 2025 | Prosperity | HPCA | SNN / low-power realtime | event-driven sparsity，适合传感器实时推理。 |
| 2025 | AiF | ISCA | on-device LLM | in-flash processing，适合存储/内存受限端侧系统。 |
| 2026 | HCTA: A heterogeneous conv-transformer networks accelerator overcoming nonlinear operator bottlenecks | Neurocomputing | edge vision / CTNN | 加速 Softmax、LayerNorm、GELU 等非线性算子，提升 Conv-Transformer 网络端侧推理。 |
| 2026 | Win-CIM: A Winograd-optimized full-digital CIM accelerator for CNNs | Integration | CNN edge inference | Winograd + digital CIM + RISC-V 扩展，面向端侧 CNN。 |
| 2026 | SHF: A DNNs accelerator with software/hardware fusion mechanism | Integration | edge AI inference | 软件/硬件协同任务划分，GPP + accelerator 混合执行。 |
| 2026 | FILCO: Flexible Composing Architecture with Real-Time Reconfigurability for DNN Acceleration | DAC 2026 | heterogeneous DNN accelerator | 面向多样 DNN workload 的实时可重构 accelerator。 |
| 2026 | FlexCNN / SeVeDo / 16nm DNN accelerator / NMSAcc / spectral CNN accelerator | ISCAS 2026 | CNN / transformer / edge vision | ISCAS 2026 中多篇边缘 DNN、CNN、Transformer、NMS、语义分割加速论文。 |
| 2026 | PowerFlow-DNN | arXiv | end-to-end edge AI inference | compiler-directed fine-grained power orchestration，针对 DNN accelerator 做功耗编排。 |
| 2026 | RISC-V TinyML accelerator for depthwise separable convolutions | arXiv | TinyML / mobile CNN | 面向 MobileNet 类 depthwise separable convolution 的 RISC-V 边缘加速。 |

## 4. LLM 加速器和传统 DL 加速器的区别

| 维度 | 传统 DL/DNN 加速器 | LLM 加速器 |
|---|---|---|
| 典型模型 | CNN、RNN、EEGNet、GNN、小型 Transformer、SNN | Decoder-only Transformer、MoE、长上下文模型、多模态 LLM |
| 典型任务 | 图像分类、检测、语义分割、语音、传感器分类、医疗信号 | 文本生成、对话、RAG、代码生成、多模态推理 |
| 核心 kernel | Conv、GEMM、pooling、activation、FFT/DSP、small attention | GEMM、attention、KV cache、LayerNorm、GELU、Softmax、sampling |
| 主要瓶颈 | MAC 利用率、SRAM 复用、片外内存访问、低功耗 | KV cache、HBM 带宽、长上下文、token-by-token decode、batching、调度 |
| 数据流 | 通常固定输入大小，pipeline 可预测 | 自回归生成，动态序列长度，runtime 行为更复杂 |
| 精度策略 | INT8/INT4、binary/ternary、mixed precision | FP8/INT4/INT3、weight-only quant、KV quantization、低秩 adapter |
| 目标平台 | MCU、FPGA、NPU、ASIC、sensor SoC、edge gateway | GPU/NPU、HBM 系统、chiplet、flash/PIM/CIM、server/edge hybrid |
| 实时性 | deadline 明确，例如 10ms/100ms/1s sensor window | token latency、throughput、TTFT、long-context latency |
| 资源受限重点 | mW 级功耗、SRAM 小、实时响应、always-on | 内存容量、KV cache、带宽、功耗/成本、serving throughput |

对 EEG/BCI 这类资源受限实时场景，传统 DL 加速器、TinyML、SNN、DSP/vector/CIM 更直接相关；LLM 加速器的 KV cache、长上下文、RAG 思想可以借鉴，但硬件尺度通常过大。

## 5. 是否有加速器考虑安全和隐私泄漏

有，但不是主流。现有安全型加速器主要分为五类：

| 类型 | 代表论文/系统 | Venue/年份 | 保护对象 | 说明 |
|---|---|---|---|---|
| Secure DNN accelerator | GuardNN | DAC 2022 | 模型/输入隐私 | secure accelerator architecture for privacy-preserving deep learning。 |
| Secure embedding / memory trace protection | SecEmb | HPCA 2025 | ML memory access/control-flow leakage | 保护 embedding layer / memory access pattern，降低模型推理侧信道。 |
| GPU timing side-channel mitigation | Ghost Arbitration | MICRO 2024 | GPU interconnect timing leakage | 防 GPU on-chip interconnect contention side channel。 |
| Secure IMC | IBM Secure Digital IMC Macro | CICC 2024 / JSSC 2025 | input/model security | 保护 in-memory compute 免受 side-channel 和 bus probing。 |
| FHE/ZKP accelerator | HYDRA / zkPHIRE | HPCA 2025/2026 | encrypted/private computation | 针对 FHE/ZKP 加速，支持隐私保护计算，但成本高。 |
| Side-channel-resistant edge AI core | PermuteV | arXiv 2025/2026 | edge AI inference side-channel | 用 RISC-V core 乱序/置换等方式降低边缘 AI 推理侧信道。 |
| LLM cache privacy | SafeKV | arXiv 2025 | LLM KV-cache timing leakage | 选择性 KV cache sharing，缓解 prompt/privacy timing side channel。 |

关键判断：

1. 大多数 AI 加速器只优化性能/能耗/面积，不主动考虑输入隐私、训练数据隐私或 side-channel。
2. 已有安全加速器更多保护物理侧信道、bus probing、memory access pattern、FHE/ZKP 计算，不一定直接解决 membership inference 或模型记忆。
3. LLM 加速器的安全问题正在增加，尤其是 KV cache sharing、timing side channel、multi-tenant serving。
4. Edge AI 加速器如果处理医疗/EEG/BCI 数据，隐私问题更强，但目前专门面向 EEG/BCI 的 secure accelerator 基本空白。

## 6. 对 EEG/BCI 资源受限实时设备的启发

EEG/BCI 场景的算子和大模型不同，更适合以下加速路径：

| EEG/BCI 处理阶段 | 可借鉴加速器方向 | 安全扩展 |
|---|---|---|
| 滤波、重采样、notch、bandpass | DSP / RISC-V vector / tiny accelerator | 预处理配置签名、参数完整性检查 |
| PSD、DE、FFT、CSP、Riemannian covariance | vector/SIMD、FFT accelerator、CIM | 特征 SRAM 隔离、访问控制 |
| EEGNet/CNN/GNN/Transformer decoder | TinyML NPU、SNN accelerator、CIM/PIM | 固定延迟推理、置信度门控、model signature |
| online adaptation / SFDA adapter | low-rank update accelerator、small NPU | adapter 隔离、隐私审计、更新日志 |
| 无线传输前处理 | near-sensor compression / feature extraction | metadata/timing padding、固定窗口输出 |
| real-time feedback | deadline-aware scheduler / DVFS | deadline monitor、安全动作门控 |

如果后续做 BCI secure accelerator，比较合理的定位不是复刻 LLM accelerator，而是：

Secure real-time EEG/BCI preprocessing + feature extraction + lightweight decoder accelerator。

核心创新点可以是：

1. 低功耗实时推理。
2. 预处理/特征提取配置完整性。
3. EEG feature/model/adapter 的隐私隔离。
4. 固定速率/固定延迟处理，降低 timing leakage。
5. 后门/异常输入检测。
6. 面向 closed-loop BCI 的安全门控。

## 7. 参考链接

- ISCA 2024 DBLP: https://dblp.org/db/conf/isca/isca2024.html
- MICRO 2024 DBLP: https://dblp.org/db/conf/micro/micro2024.html
- ISCA 2025 DBLP: https://dblp.uni-trier.de/db/conf/isca/isca2025.html
- HPCA 2025 DBLP: https://dblp.org/db/conf/hpca/hpca2025
- ASPLOS 2025 DBLP: https://dblp.org/db/conf/asplos/asplos2025-1.html
- HCTA 2026: https://www.sciencedirect.com/science/article/pii/S0925231226010179
- Win-CIM 2026: https://www.sciencedirect.com/science/article/pii/S1879239126002031
- SHF 2026: https://www.sciencedirect.com/science/article/pii/S1879239126001748
- DAC 2026 FILCO: https://www.researchgate.net/publication/403682446_FILCO_Flexible_Composing_Architecture_with_Real-Time_Reconfigurability_for_DNN_Acceleration
- ISCAS 2026 program: https://2026.ieee-iscas.org/assets/ISCAS_2026_Program.pdf
- IBM secure digital IMC: https://research.ibm.com/publications/digital-in-memory-compute-for-machine-learning-applications-with-input-and-model-security
- Secure IMC CICC 2024: https://research.ibm.com/publications/a-secure-digital-in-memory-compute-imc-macro-with-protections-for-side-channel-and-bus-probing-attacks
- Ghost Arbitration MICRO 2024: https://pure.kaist.ac.kr/en/publications/ghost-arbitration-mitigating-interconnect-side-channel-timing-att
- zkPHIRE HPCA 2026: https://2026.hpca-conf.org/details/hpca-2026-main-conference/105/zkPHIRE-A-Programmable-Accelerator-for-ZKPs-over-HIgh-degRee-Expressive-Gates
- GuardNN DAC 2022: https://www.csl.cornell.edu/~zhiruz/pdfs/guardnn-dac2022.pdf

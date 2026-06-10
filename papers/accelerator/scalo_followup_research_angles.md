# SCALO Follow-up Papers and Research Angles

## Download status

| Paper | Local status | Note |
|---|---|---|
| HALO: Hardware-Software Co-Design for Brain-Computer Interfaces | Downloaded | ISCA 2020, low-power implantable BCI architecture |
| Noema: Hardware-Efficient Template Matching for Neural Population Pattern Detection | Downloaded | MICRO 2021, neural template matching accelerator |
| uBrain: A Unary Brain Computer Interface | Downloaded | ISCA 2022, EEG/DNN BCI accelerator |
| SCALO: An Accelerator-Rich Distributed System for Scalable Brain-Computer Interfacing | Already present | ISCA 2023, distributed multi-implant BCI system |
| Marple: Scalable Spike Sorting for Untethered Brain-Machine Interfacing | Downloaded | ASPLOS 2024, spike sorting pipeline for untethered BMI |
| Foresee: A Modular and Open Framework to Explore Integrated Processing on Brain-Computer Interfaces | Downloaded | EMBC 2025, BCI processor exploration framework |
| Rearchitecting a Neuromorphic Processor for Spike-Driven Brain-Computer Interfacing | Not publicly downloaded | MICRO 2024, closed IEEE access/request-only public page |
| Neuralite: Enabling Wireless High-Resolution Brain-Computer Interfaces | Not publicly downloaded | MobiCom 2024, closed ACM access/request-only public page |
| InfiniMind: A Learning-Optimized Large-Scale Brain-Computer Interface | Not publicly downloaded | ISCA 2025, closed ACM access/request-only public page |
| TierX: A Simulation Framework for Multi-tier BCI System Design Evaluation and Exploration | Not publicly downloaded | ASPLOS 2026, closed ACM access/request-only public page |

## Best research angles

### 1. Secure on-device adaptation engine

BCI devices often need cross-subject calibration, online adaptation, and continual learning, but adaptation can leak private neural state through update timing, selected samples, gradients, statistics, and memory access patterns. A strong direction is a fixed-latency secure adaptation engine: frozen backbone inference plus small statistics/adapter/head update units, bounded updates, constant-time scheduling, and privacy-aware feature buffering.

### 2. GPU-free wearable BCI accelerator

The SCALO/HALO/uBrain/Marple line is not desktop-GPU-centric. A realistic wearable or implantable BCI target is an ARM/RISC-V controller plus SRAM, DSP/NPU/FPGA/ASIC accelerators, and strict latency/power constraints. The research question is how to support real-time EEG/spike processing without offloading raw neural data to a server or relying on a high-power GPU.

### 3. Streaming neural preprocessing and compression

Marple and Noema suggest that front-end processing can dominate scalability. For wearable BCI, useful kernels include spike sorting, template matching, event/window detection, artifact rejection, filtering, feature extraction, and streaming compression. The hardware opportunity is to reduce raw neural data movement before it reaches heavier decoders.

### 4. Secure distributed BCI interconnect and storage

SCALO and Neuralite point to multi-node/wireless BCI systems. Security-sensitive directions include encrypted low-power links, authenticated packet scheduling, traffic-shaping to hide neural-state-dependent communication, secure NVM buffering, replay protection, and resilience to dropped or adversarial packets.

### 5. Design-space exploration framework

Foresee and TierX suggest that BCI accelerator research needs a reusable workload and architecture exploration environment. A practical first step is to build a BCI workload profiler covering latency, power, memory traffic, channel count, window size, adaptation frequency, and leakage-sensitive events.

### 6. Quantized and sparse BCI model execution

uBrain and the broader accelerator papers in this directory support low-bit, unary/stochastic, sparse, and event-driven computation. This can be combined with EEGNet/TCN/Transformer/SNN decoders and adapter-only updates to meet wearable energy budgets.

## Recommended first project shape

Start with a non-invasive EEG workload because datasets and models are easier to reproduce. Use EEGNet or a small EEG Transformer as the decoder, add online BN/statistics or adapter-only adaptation, then design a GPU-free secure streaming accelerator that measures accuracy, latency, energy, memory traffic, and leakage proxies. Compare against CPU/GPU software only as baselines, not as the deployment target.


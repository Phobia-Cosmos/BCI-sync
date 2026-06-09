# BCI 实时数据处理论文索引

说明：我优先下载了官方、作者主页、arXiv、开放期刊或会议官网可访问的 PDF。BCI 实时系统架构的经典工作很多发表在 IEEE TBME、JNE、Frontiers、Neuroinformatics、Graz BCI 等权威 venue，而不一定都是传统 CS 顶会；未下载非授权或付费墙 PDF。

## 建议阅读顺序

1. 先看 `01_BCI2000_general_purpose_BCI_system_2004.pdf`：经典实时 BCI 架构，Source / Signal Processing / Application / Operator 四模块。
2. 再看 `02_OpenViBE_open_source_BCI_platform_2010.pdf`：图式 pipeline、在线处理、可视化和反馈设计。
3. 再看 `12_Timeflux_open_source_near_real_time_signal_streams_GrazBCI_2019.pdf`、`04_NeuXus_real_time_biosignal_pipeline_2022.pdf`、`13_Human_in_the_loop_BCI_research_framework_2023.pdf`：现代 Python/LSL/WebSocket 风格工程化架构。
4. 如果关注低延迟闭环和更复杂模型，看 `11_BRAND_closed_loop_real_time_neural_decoding_platform_2024.pdf`。
5. 如果关注情绪/被动 BCI，看 `06_Detecting_affective_covert_user_states_passive_BCI_2009.pdf` 和 `07_Real_time_EEG_emotion_monitoring_stable_features_2016.pdf`。
6. 如果关注康复/无线/最新在线原型，看 `08_Real_time_wireless_imagined_speech_EEG_decoding_2025.pdf`、`09_Real_Time_EEG_BCI_Stroke_Rehab_Latent_Features_2025.pdf`、`14_Adaptive_closed_loop_EEG_BCI_neurorehabilitation_review_2024.pdf`。

## 下载文件

| 文件 | 主题 | 重点看什么 | 来源 |
|---|---|---|---|
| `01_BCI2000_general_purpose_BCI_system_2004.pdf` | BCI2000 通用实时 BCI 系统 | Source 负责采集/存储，Signal Processing 做特征和翻译，Application 做反馈/控制，Operator 做配置和监控 | https://www.neurotechcenter.org/sites/default/files/misc/BCI2000%20-%20A%20General-Purpose%20Brain-Computer%20Interface%20%28BCI%29%20System.pdf |
| `02_OpenViBE_open_source_BCI_platform_2010.pdf` | OpenViBE 开源 BCI 平台 | Acquisition server、scenario graph、boxes、在线/离线复用、实时可视化和反馈 | https://inria.hal.science/hal-00477153/document |
| `04_NeuXus_real_time_biosignal_pipeline_2022.pdf` | NeuXus 实时 biosignal pipeline | Python 节点式 pipeline、LSL 输入/输出、DataFrame 分块、实时分类和 neurofeedback 部署 | https://arxiv.org/pdf/2012.12794 |
| `05_Real_Time_EEG_Classification_via_Coresets_2020.pdf` | Coreset 在线 EEG 分类 | 如何避免“离线训练慢、在线只能推理”的问题，用压缩样本支持实时/并行学习 | https://arxiv.org/pdf/1901.00512 |
| `06_Detecting_affective_covert_user_states_passive_BCI_2009.pdf` | 被动 BCI / covert affective state | 被动 BCI 如何把用户状态作为隐式输入，目标论文在会议集内部 | https://www.honda-ri.de/pubs/pdf/1396.pdf |
| `07_Real_time_EEG_emotion_monitoring_stable_features_2016.pdf` | 实时 EEG 情绪识别稳定特征 | 情绪识别特征稳定性、实时监测、如何避免只看单点预测 | https://personal.ntu.edu.sg/elpwang/PDF_web/06980754.pdf |
| `08_Real_time_wireless_imagined_speech_EEG_decoding_2025.pdf` | 无线 imagined speech 实时 BCI | 低通道/无线/实时解码原型，适合看端到端产品化思路 | https://arxiv.org/pdf/2511.07936 |
| `09_Real_Time_EEG_BCI_Stroke_Rehab_Latent_Features_2025.pdf` | 中风手部康复实时 BCI | EEG 特征到嵌入式外骨骼控制，适合看“识别结果如何驱动设备” | https://arxiv.org/pdf/2510.15890 |
| `10_RT_NET_real_time_hdEEG_source_reconstruction_2020.pdf` | hdEEG 实时源重建 | 在线处理、源定位、低延迟 neurofeedback/BCI 支撑 | https://link.springer.com/content/pdf/10.1007/s12021-020-09479-3.pdf |
| `11_BRAND_closed_loop_real_time_neural_decoding_platform_2024.pdf` | BRAND 闭环实时神经解码平台 | 异步节点图、Redis in-memory database、stream 传输、日志记录、毫秒级延迟 | https://bbhaduri.github.io/assets/pdf/brand_paper.pdf |
| `12_Timeflux_open_source_near_real_time_signal_streams_GrazBCI_2019.pdf` | Timeflux 近实时信号流框架 | YAML/pipeline、节点、LSL、epoching/windowing、recording、插件化 | https://timeflux.io/assets/pdf/Timeflux_GBCIC2019.pdf |
| `13_Human_in_the_loop_BCI_research_framework_2023.pdf` | Human-in-the-loop 在线 BCI 框架 | LSL + WebSocket、实时刺激控制、在线分类、云端训练/重训练、保存未处理 EEG | https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1129362/pdf |
| `14_Adaptive_closed_loop_EEG_BCI_neurorehabilitation_review_2024.pdf` | 闭环 EEG BCI 康复综述 | 闭环 BCI 的采集、解码、反馈、适应性更新和康复应用全景 | https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1431815/pdf |

## 共性处理模式

- 实时系统通常不把所有 raw EEG 无限期存储；研究/校准阶段更倾向保存 raw，产品阶段更倾向保存特征、预测、置信度、事件和信号质量。
- 数据流一般是采集端、时间戳同步、环形缓冲、滤波/去伪迹、窗口化、特征提取、分类/回归、平滑决策、反馈/控制。
- 现代系统越来越多使用节点图/流式架构，例如 BCI2000 模块、OpenViBE boxes、Timeflux nodes、NeuXus nodes、BRAND nodes。
- 实时通信常见 LSL、WebSocket、Redis streams/in-memory DB、框架内部 buffer；长期研究数据常落成 XDF/EDF/自定义二进制/数据库日志。
- 关键工程指标是端到端延迟、时间戳同步、丢包处理、信号质量、事件 marker、模型版本、可复现实验配置。

## 未直接下载但值得看

- Lab Streaming Layer 2025 论文：PDF 直链返回 403/验证页，因此只保留链接。它对实时多模态同步、stream 和 XDF 记录很重要。链接：https://doi.org/10.1162/IMAG.a.136

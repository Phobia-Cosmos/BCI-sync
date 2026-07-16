# 外部代码仓库索引

以下外部项目保留各自的 `.git` 和远端配置；整理时未重克隆或覆盖工作树。

| 本地目录 | 远端 | 基准提交 |
|---|---|---|
| `bci_toolkits/Deep-BCI` | `https://github.com/DeepBCI/Deep-BCI.git` | `225b7b4` |
| `brain_decoding/2022NeurIPS-TSMNet-SPD-DSBN-EEG-UDA` | `https://github.com/rkobler/TSMNet.git` | `90293b9` |
| `brain_decoding/2024CVPR-MindBridge` | `https://github.com/littlepure2333/mindbridge.git` | `2a3943f` |
| `sfda/2020ICML-SHOT` | `https://github.com/tim-learn/SHOT.git` | `f7d555a` |
| `sfda/2023Arxiv-SFDA-SSVEP-BCI` | `https://github.com/osmanberke/SFDA-SSVEP-BCI.git` | `abf7316` |
| `sfda/2024CVPR-DIFO-Plus` | `https://github.com/tntek/DIFO-Plus.git` | `639d9f6` |
| `sfda/2024CVPR-DIFO-ProDe-Source-Free-Domain-Adaptation` | `https://github.com/tntek/source-free-domain-adaptation.git` | `5f7fe57` |
| `sfda/2025AAAI-SF-UIDA` | `https://github.com/xiaobaben/SF-UIDA.git` | `4172de4` |
| `sfda/2025JBHI-PDCC-Cross-Subject-EEG-Classification` | `https://github.com/SunseaIU/PDCC.git` | `b61dd0e` |

PDCC 的 `get_proxy_domain.py` 含本地未提交的路径处理修复，重新克隆前需先保存该改动。

`tta_security/` 下的 BrainUICL、RTTDP、SPR 和 PuriDivER 由 BCI-sync 根仓库直接跟踪。

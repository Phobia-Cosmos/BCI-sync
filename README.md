# BCI research workspace

本目录已按内容类型整理：

- `papers/`：46 篇唯一正式论文，按 BCI、持续学习和测试时自适应主题分类。
- `code/`：项目源码、实验代码和工作区工具；外部仓库信息见 `code/REPOSITORIES.md`。
- `documents/`：本地分析稿、调研笔记、实验报告和配图。

## 数据策略

本地 `data/` 不属于 BCI-sync 远端仓库，已按“远端不包含则本地不保留”的规则删除。公开数据集来源和处理流程仍记录在：

- `documents/research_notes/bci_sfda_inventory_and_plan.md`
- `documents/research_notes/data_task_processing_comparison.md`

代码仓库内名为 `data/` 或 `datasets/` 的目录如果由上游 Git 跟踪，则属于数据加载代码、类别表或路径清单，继续随源码保留。

## 说明

外部项目保留各自的嵌套 `.git`，因为本仓库没有配置 Git submodule，不能仅凭 BCI-sync 远端恢复这些项目。PDCC 的 `get_proxy_domain.py` 仍保留本地未提交修复。

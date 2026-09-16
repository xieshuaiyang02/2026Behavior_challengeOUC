# BEHAVIOR Challenge 2026 — Code Snapshot

本仓库汇总 BEHAVIOR Challenge 2026 相关的 BEHAVIOR-1K 与 openpi 工作代码，包括本地修改及新增的 OUC 训练、推理、阶段监督和评测实现。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| [`BEHAVIOR-1K/`](BEHAVIOR-1K/) | OmniGibson、BDDL 任务定义、评测与资产处理源码 |
| [`openpi/`](openpi/) | 策略模型、训练、推理和 OUC 扩展源码 |

主要自定义入口：

- 训练：[`openpi/scripts/b1k/train_b1k_ouc.py`](openpi/scripts/b1k/train_b1k_ouc.py)
- 策略服务：[`openpi/scripts/b1k/serve_b1k_ouc.py`](openpi/scripts/b1k/serve_b1k_ouc.py)
- 阶段标注：[`openpi/scripts/behavior/build_stage_annotations_ouc.py`](openpi/scripts/behavior/build_stage_annotations_ouc.py)
- OUC 训练配置：[`openpi/src/openpi/training/config_ouc.py`](openpi/src/openpi/training/config_ouc.py)
- OUC 数据加载：[`openpi/src/openpi/training/data_loader_ouc.py`](openpi/src/openpi/training/data_loader_ouc.py)
- OUC 评测：[`BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval_ouc.py`](BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval_ouc.py)

## 使用说明

安装与基础使用请参考 [BEHAVIOR-1K README](BEHAVIOR-1K/README.md) 和 [openpi README](openpi/README.md)。运行前需要自行准备相应的仿真环境、数据集和模型权重，并按实际环境设置代码及配置中的路径。

该快照包含导出时的工作文件与本地修改。上游版本和新增文件记录在 [`SOURCE_SNAPSHOT.json`](SOURCE_SNAPSHOT.json)。

## 上传范围

保留源码、依赖声明、配置、BDDL 任务定义和语义规则，以及上游测试所需的少量固定数据。

已排除模型检查点、训练输出、W&B 记录、下载的数据集、虚拟环境、缓存、日志、备份文件和本地编辑器设置。Notebook 保留代码与说明，运行输出已清空。源码中的 `checkpoints.py`、`checkpoints_ouc.py` 等文件仍然保留。

为控制仓库体积，省略了上游文档中的大批图片、动画，以及资产处理工具附带的四个可执行程序。因此部分本地文档图片链接无法显示；完整素材与工具可从原始项目获取。

## 上游与许可证

- [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
- [wensi-ai/openpi](https://github.com/wensi-ai/openpi)

各子项目保留原有许可证文件，包括 [OmniGibson](BEHAVIOR-1K/OmniGibson/LICENSE)、[BDDL](BEHAVIOR-1K/bddl3/LICENSE)、[openpi](openpi/LICENSE) 及 [Gemma](openpi/LICENSE_GEMMA.txt)。使用相应代码和资源时，请查看其许可证。

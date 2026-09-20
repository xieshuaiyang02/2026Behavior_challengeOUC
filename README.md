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

## 数据预处理：生成 OUC 阶段标注资源

入口为 [`openpi/scripts/behavior/build_stage_annotations_ouc.py`](openpi/scripts/behavior/build_stage_annotations_ouc.py)。它读取**已有、已验证的 episode manifest**，按任务编号筛选记录，再构建阶段词表、分段标注和相关元数据。当前入口不包含从原始 demos 首次生成 manifest 的步骤；只有数据集目录、没有已验证清单时，尚不能执行此流程。

### 准备输入

1. 进入已安装本项目依赖的 Python 环境，确保 `numpy` 和 `polars` 可用。
2. 准备本地 demos 数据集目录，以及独立保存的 `manifest_ouc.json`。清单需要包含非空的 `episodes` 列表、所选任务的 episode 记录、数据划分、帧数与采样率信息，以及内嵌的 `annotation` 或有效的 `annotation_path`。构建阶段词表还需要所选数据中包含训练集技能标注。
3. 检查清单中引用的标注文件和可选对象元数据文件。相对路径以**输入 manifest 所在目录**为基准；换机器时需要更新失效的绝对路径。脚本保留原始 episode 标识，不会自动推断标注文件名或重新编号。
4. 选择一个新目录或空目录作为输出，建议放在 Git 仓库外。输入 manifest 应放在另一个目录，避免与输出目录冲突。

### 执行预处理

替换下面的路径变量后，在 Bash 中执行。`OPENPI_ROOT` 指包含 `scripts/` 和 `src/` 的 openpi 目录；在本仓库中它是 `openpi/` 子目录。示例任务编号可按需修改，且所有选定任务都必须出现在输入清单中。

```bash
OPENPI_ROOT="/path/to/openpi"
DATASET_ROOT="/path/to/2026-challenge-demos"
MANIFEST_PATH="/path/to/manifests/manifest_ouc.json"
OUTPUT_DIR="/path/to/generated/stage_assets_ouc"
TASK_IDS=(0 4 14 30 35 38 42 64)

python "$OPENPI_ROOT/scripts/behavior/build_stage_annotations_ouc.py" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST_PATH" \
  --task-ids "${TASK_IDS[@]}" \
  --output-dir "$OUTPUT_DIR"
```

正式生成前，可在命令末尾增加 `--check-only` 进行预检。该选项会在临时目录完成构建，不写入目标输出目录，但可能创建其父目录；检查通过后去掉该选项，重新执行正式构建。

请显式传入 `--manifest`。省略时，脚本默认读取 `<DATASET_ROOT 的父目录>/stage_assets_ouc/manifest_ouc.json`；若同时把这个非空的 `stage_assets_ouc` 目录作为输出，脚本会拒绝覆盖。需要重新构建时，应选择新的输出目录。

### 生成结果与 manifest 的来源

成功后，`OUTPUT_DIR` 中会生成：

| 文件 | 用途 |
| --- | --- |
| `stage_segments.parquet`、`raw_stage_segments.parquet` | 处理后的阶段分段及原始阶段分段 |
| `stage_vocab.json`、`task_stage_vocab.json`、`stage_id_to_metadata.json` | 阶段词表、任务词表与阶段元数据映射 |
| `episode_stage_metadata.json`、`stage_config.json` | Episode 元数据与阶段资源配置 |
| `stage_conflicts.json`、`stage_invalid_intervals.json` | 冲突与无效区间记录 |
| `manifest_ouc.json` | 本次筛选并用于构建的 manifest 副本 |

输出的 `manifest_ouc.json` 来自 `prepare_manifest()` 对输入清单的处理，并在构建成功后由 `write_json(output / "manifest_ouc.json", manifest)` 写入。它不是从原始 demos 首次发现并生成的清单。输出清单中的外部标注路径会被解析为绝对路径，迁移到另一台机器后应重新检查这些路径。

## 上传范围

保留源码、依赖声明、配置、BDDL 任务定义和语义规则，以及上游测试所需的少量固定数据。

已排除模型检查点、训练输出、W&B 记录、下载的数据集、虚拟环境、缓存、日志、备份文件和本地编辑器设置。Notebook 保留代码与说明，运行输出已清空。源码中的 `checkpoints.py`、`checkpoints_ouc.py` 等文件仍然保留。

为控制仓库体积，省略了上游文档中的大批图片、动画，以及资产处理工具附带的四个可执行程序。因此部分本地文档图片链接无法显示；完整素材与工具可从原始项目获取。

## 上游与许可证

- [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
- [wensi-ai/openpi](https://github.com/wensi-ai/openpi)

各子项目保留原有许可证文件，包括 [OmniGibson](BEHAVIOR-1K/OmniGibson/LICENSE)、[BDDL](BEHAVIOR-1K/bddl3/LICENSE)、[openpi](openpi/LICENSE) 及 [Gemma](openpi/LICENSE_GEMMA.txt)。使用相应代码和资源时，请查看其许可证。

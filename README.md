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
- Manifest 生成：[`openpi/scripts/behavior/build_manifest_ouc.py`](openpi/scripts/behavior/build_manifest_ouc.py)
- 阶段标注：[`openpi/scripts/behavior/build_stage_annotations_ouc.py`](openpi/scripts/behavior/build_stage_annotations_ouc.py)
- OUC 训练配置：[`openpi/src/openpi/training/config_ouc.py`](openpi/src/openpi/training/config_ouc.py)
- OUC 数据加载：[`openpi/src/openpi/training/data_loader_ouc.py`](openpi/src/openpi/training/data_loader_ouc.py)
- OUC 评测：[`BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval_ouc.py`](BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval_ouc.py)

## 使用说明

安装与基础使用请参考 [BEHAVIOR-1K README](BEHAVIOR-1K/README.md) 和 [openpi README](openpi/README.md)。运行前需要自行准备相应的仿真环境、数据集和模型权重，并按实际环境设置代码及配置中的路径。

该快照包含导出时的工作文件与本地修改。上游版本和新增文件记录在 [`SOURCE_SNAPSHOT.json`](SOURCE_SNAPSHOT.json)。

## 数据预处理：生成 OUC 阶段标注资源

预处理分为两步，按顺序执行：

1. [`build_manifest_ouc.py`](openpi/scripts/behavior/build_manifest_ouc.py) 从官方 episode 元数据生成首份 `manifest_ouc.json`，核对所选任务和本地标注文件。
2. [`build_stage_annotations_ouc.py`](openpi/scripts/behavior/build_stage_annotations_ouc.py) 读取第一步的 manifest，构建阶段词表、分段标注和相关元数据。

两个脚本都使用本地数据，保留官方 episode 标识和标注文件映射。Manifest 生成脚本不会下载数据、推断标注文件名或生成阶段资源。

### 准备输入

进入已安装本项目依赖的 Python 环境。读取 Parquet episode 元数据需要 `pyarrow`；第二步构建阶段资源需要 `numpy` 和 `polars`。

预处理至少需要以下本地文件；episode 元数据优先读取 Parquet，仅在没有 Parquet 文件时回退到 JSONL：

```text
DATASET_ROOT/
├── meta/
│   ├── info.json                    # 包含有效的 fps
│   └── episodes/**/*.parquet        # 或 meta/episodes.jsonl
└── annotations/
    └── task-XXXX/*.json             # 文件路径以官方元数据为准
```

Episode 元数据需要包含 `episode_index`、`task_index`、`tasks`、`length`、`annotation_path`；当前脚本要求每条记录的 `tasks` 中恰好有一个任务名称。所有请求的任务都必须出现在元数据中，其标注文件必须已下载到本地。第二步还需要所选数据包含可用于构建词表的训练集技能标注。

### 第一步：生成 manifest

替换下面的路径变量后，在 Bash 中执行。`OPENPI_ROOT` 指包含 `scripts/` 和 `src/` 的 openpi 目录；在本仓库中它是 `openpi/` 子目录。示例任务编号可按需修改。Manifest 和阶段资源使用不同的输出目录，建议都放在 Git 仓库外。

```bash
OPENPI_ROOT="/path/to/openpi"
DATASET_ROOT="/path/to/2026-challenge-demos"
MANIFEST_DIR="/path/to/generated/ouc_manifests"
MANIFEST_PATH="$MANIFEST_DIR/manifest_ouc.json"
OUTPUT_DIR="/path/to/generated/stage_assets_ouc"
TASK_IDS=(0 4 14 30 35 38 42 64)

python "$OPENPI_ROOT/scripts/behavior/build_manifest_ouc.py" \
  --dataset-root "$DATASET_ROOT" \
  --task-ids "${TASK_IDS[@]}" \
  --output "$MANIFEST_PATH" \
  --audit-output "$MANIFEST_DIR/manifest_audit_ouc.json"
```

第一步生成 `MANIFEST_PATH` 指向的清单，以及可选的任务、episode 和帧数统计文件 `manifest_audit_ouc.json`。脚本会创建输出文件的父目录；若输出 JSON 已存在，会覆盖该文件，因此请选用合适的新路径或提前保留需要的旧清单。

如果标注保存在数据集目录外，可在第一步命令中增加 `--annotation-root "/path/to/annotations"`。该目录应直接包含 `task-XXXX/` 子目录；脚本处理相对 `annotation_path` 时会移除开头的 `annotations/` 再拼接目录。元数据中的绝对路径不受这个参数影响。

**当前生成规则：** 官方 `episode_index` 保留为 `episode_id`，`dataset_index` 固定为 `0`，所选 episode 全部标为 `split="train"`；source 与 dataset 使用相同的 FPS、帧数，时间偏移为 `0`。该默认清单适用于单数据集、未重采样的训练数据，不会自动划分验证集。若使用不同的数据划分或时间轴，需要先据实调整清单，再进行第二步。

### 第二步：生成阶段标注资源

在同一 Bash 会话中沿用上面的变量，将第一步产物通过 `--manifest` 传入：

```bash
python "$OPENPI_ROOT/scripts/behavior/build_stage_annotations_ouc.py" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST_PATH" \
  --task-ids "${TASK_IDS[@]}" \
  --output-dir "$OUTPUT_DIR"
```

`OUTPUT_DIR` 必须是新目录或空目录；不要把第一步的 manifest 放进这个目录。正式生成前，可在**第二步**命令末尾增加 `--check-only` 进行预检。该选项会在临时目录完成构建，不写入目标输出目录，但可能创建其父目录；检查通过后去掉该选项，重新执行正式构建。

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

首份清单由第一步的 `build_manifest_ouc.py` 根据官方元数据生成。第二步输出目录中的同名文件，则是 `prepare_manifest()` 处理输入清单后、随阶段资源保存的副本。

生成的清单包含绝对数据与标注路径。迁移到另一台机器后，应按新目录重新执行第一步，或核对并更新已有清单中的路径，再构建阶段资源。若自行提供带相对路径的清单，第二步以输入 manifest 所在目录解析这些路径。

## 上传范围

保留源码、依赖声明、配置、BDDL 任务定义和语义规则，以及上游测试所需的少量固定数据。

已排除模型检查点、训练输出、W&B 记录、下载的数据集、虚拟环境、缓存、日志、备份文件和本地编辑器设置。Notebook 保留代码与说明，运行输出已清空。源码中的 `checkpoints.py`、`checkpoints_ouc.py` 等文件仍然保留。

为控制仓库体积，省略了上游文档中的大批图片、动画，以及资产处理工具附带的四个可执行程序。因此部分本地文档图片链接无法显示；完整素材与工具可从原始项目获取。

## 上游与许可证

- [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
- [wensi-ai/openpi](https://github.com/wensi-ai/openpi)

各子项目保留原有许可证文件，包括 [OmniGibson](BEHAVIOR-1K/OmniGibson/LICENSE)、[BDDL](BEHAVIOR-1K/bddl3/LICENSE)、[openpi](openpi/LICENSE) 及 [Gemma](openpi/LICENSE_GEMMA.txt)。使用相应代码和资源时，请查看其许可证。

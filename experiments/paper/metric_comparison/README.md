# 同一干预数据上的两套指标比较

本包先建立可核验的比较队列，再调用已有指标或导入已有指标表。准备、采集和精确对齐分为独立命令；不修改冻结实验、指标定义、论文或已有输出，也不启动模拟。

## 当前准备

本轮已生成 `outputs/paper/metric_comparison_preparation_20261122_ready`：staged 和 timing 共928个任务、6048条路径（含参考）、413088个观测状态。原始盘当前不可用，旧指标尚未计算。可直接使用该目录进入后续采集；准备产物位于忽略的 `outputs/`，代码和以下命令可重建。

从模型仓库运行，重建时输出必须是尚不存在的新目录：

```sh
python -m experiments.paper.metric_comparison prepare \
  --bundle outputs/paper/recommendation_manuscript/completed_20261119 \
  --output outputs/paper/metric_comparison_preparation_new
```

`--bundle` 可重复指定，例如后续补充 timing 数据包。比较推荐实现时，可显式追加最新 `opinion_random_factorial_complete` 数据包。同一研究只能选一代；混入旧六臂与新九臂数据包中的重复 task 会被拒绝。数据包应包含 `data/manifest.json`、`data/study/plan.json`、记录及运行绑定。

本地 timing 的已认证旧布局也可直接读取：`--bundle outputs/paper/recommendation_manuscript/final_timing_review_20261120`。其 `source_manifest.json` 对 `source/` 下的计划、记录和绑定逐项提供哈希，不需要接盘或重新导出。阶段与 timing 可在同一新准备目录中联合展开：

```sh
python -m experiments.paper.metric_comparison prepare \
  --bundle outputs/paper/recommendation_manuscript/completed_20261119 \
  --bundle outputs/paper/recommendation_manuscript/final_timing_review_20261120 \
  --output outputs/paper/metric_comparison_preparation_20261122_staged_timing
```

准备验证 manifest 中全部文件的 SHA、冻结计划指纹、逐任务指纹、逐臂二进制来源及观测步唯一性。只展开已记录的 `P`、`E`、`C_edge`、`C_pair`，不重新计算它们。输出包括：

- `points.csv.gz`：以 `profile,task_id,stage,arm,step` 严格标识观测状态，保留原始值、相对步数、计划／二进制／原始记录来源。
- `paths.csv`：每臂起止时刻、终态原始分类、参数、初始化 seed、所需原始目录与基本文件可用性。staged 参考轨迹单列为 `stage=reference,arm=reference`。
- `tasks.csv`：冻结任务指纹及完整 metadata；两个 `legacy_*_template.csv` 提供导入表头。
- `preparation.json`：来源数据包、完整性信息、输出哈希，以及等待采集的历史指标清单。

原始目录取自数据包的 `source_dataset` 和每臂 `raw_record`。缺盘时保留 `pending_legacy_metrics` 和具体缺失文件，不补零。若外接盘挂载路径发生变化，在新的准备目录中指定 `--raw-root staged=/new/path/to/staged`；其它 profile 同理。准备检查基本文件是否存在，后续原始采集还需验证可读取性、历史覆盖和状态一致性。

## 接盘后采集

```sh
python -m experiments.paper.metric_comparison collect-raw \
  --preparation outputs/paper/metric_comparison_preparation_20261122_ready \
  --output /tmp/ehk-metrics-first-paths --max-paths 2
```

去掉 `--max-paths` 才处理全部准备路径。该命令使用持久记录的 `RawSimulationRecord`，在准备表指定的同一状态调用既有 `observables.calc_distance` 计算 `I_p,I_s`、`calc_homophily` 计算 `I_h`。完整路径的 `I_w` 单独调用既有 `get_indices` 和 `get_gradation_index_hp`，沿用自适应采样与积分实现。采集可在同一目录按路径恢复，配置及现有指标实现的源码哈希被绑定；挂载位置变化可用 `--raw-root PROFILE=DATASET` 指定；改变已绑定的源码、参数或挂载覆盖时，使用新的采集输出目录。原始元数据、观测文件、轨迹清单、初始图及事件库的哈希进入逐路径凭据；已缓存指标在原始盘可用时核对这些哈希。轨迹读取器继续验证清单内各数据块的校验和。

阶段分支的准备表只列出其干预窗口，但原始分支记录必须覆盖从 0 开始的完整重放前缀。不能把窗口或16步导出网格积分当作既有 `I_w`。参考、早／中／晚分支及各自终点分别处理，不把不同终点合并为同一条路径。

## 导入已有旧指标表

也可用模板准备外部表，然后导入：

```sh
python -m experiments.paper.metric_comparison collect \
  --preparation /path/to/preparation --output /tmp/ehk-metrics-import \
  --points /path/to/legacy_points.csv.gz --paths /path/to/legacy_paths.csv
```

状态表需要五项状态键、`plan_fingerprint,binary_sha256,raw_record,I_p,I_s,I_h`；路径表需要四项路径键、同三项来源、`I_w,sampling_method,prefix_start,trajectory_end`。`prefix_start` 必须为0，终点必须与准备路径一致。本轮 `sampling_method` 固定为 `existing_get_indices_adaptive_0.01_max_interval_512`，16步网格积分或其它算法不能冒充现有 `I_w`。重复键、队列之外的键、来源不一致或非有限指标值会报错。尚未计算的状态不写入导入表，继续作为 pending 保留。已有采集目录也可用 `--collection /path/to/collection` 导入。

```sh
python -m experiments.paper.metric_comparison compare \
  --preparation /path/to/preparation --collection /path/to/collection \
  --output /tmp/ehk-metrics-comparison
```

`compare` 验证准备／采集哈希和来源后，仅精确 join，输出 `points_aligned.csv.gz`、`paths_aligned.csv.gz` 及覆盖率。缺值留空，结果保持 pending，不自动分类、不拟合新标量，也不强迫两套指标一致。

## 应当区分的解释

输出中的 `legacy` 表示原有指标家族。采集使用当前仓库的既有实现并记录源码哈希，不自动复现旧归档中不同的基线、计数对象或端点补项。

`I_h` 对有正出度用户的同信关注比例等权，`C_edge` 对有向边等权，不能互相代入。`I_p,I_s` 与新 `P` 使用不同定义；`I_w` 是完整轨迹量，`P/E` 首达分类使用16步观测与各自阈值。后续比较应检查状态差异、路径现象和配对干预结论，不能用调阈值追求标签一致。现有终态 confidence components 与 KDE 意见峰数也保持各自含义。

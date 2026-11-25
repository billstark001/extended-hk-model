# EHK 指标重算与第3.1节图源

这次改动准备原论文的完整重新分析和带种子的新运行。论文正文未改；第3.1节候选稿保存在 paper 仓库的 `artifacts/ehk_section31_draft_20261122/`。已有原始模拟继续可读，旧统计宽表不迁移、不覆盖。

## 指标和统计存储

- `I_p,I_s` 使用正文约定的 random-reference JS divergence 比值。历史 JS distance、clustered-reference 补量分别进入审计。`I_p` 衡量意见距离分布偏离初始随机参照的程度，共识也能使其上升；它不保证在 `[0,1]` 内。
- `I_h` 使用关注边两端的当前意见，先计算每个正出度节点的同信关注比例，再取节点平均。置信条件严格为 `distance < epsilon`，均匀基线为 `epsilon-epsilon**2/4`。关注帖同信率独立保存。
- `I_w` 沿时间顺序积有符号梯形，不加终点补项，不裁剪到 `[0,1]`。它是连续轨迹摘要。
- `0.6` 只作为待核对的历史分组值。分组图必须显式设置 `EHK_IW_CUTOFF`，图中称 low/high `I_w`，不会自动解释为严格 PbS/SbP 分类。
- 默认重新分析不调用 P/E probe。只有显式加入 `--include-pathway-probe` 才计算独立的 P/E 首达标签。

新统计库默认是 `outputs/experiments/paper/stats-v3.sqlite`，使用标准库 SQLite 文档存储。元数据、指标协议、原始输入身份和数组一起保存，增加统计字段无需修改宽表。NumPy 数组无 pickle，压缩后嵌入文档；每个任务事务写入。

任务的参数、真实 raw metadata、RNG 可得性、指标源码和输入文件身份共同约束缓存。旧 raw 缺种子时明确记录为不可得，不回填新任务的种子。输入改变或强制重算后，旧文档持久失效；失败任务不能继续向绘图提供旧结果。多个指标版本或 raw 来源共存时，绘图要求明确选择。

## 从现有 raw 重新分析

以下命令在 model 仓库执行。把路径替换为包含 `s_grad_*` 等场景目录的实际 raw 根目录。当前历史工作区所指的外接卷未挂载；本轮没有完成历史全量重算。

```sh
python -m experiments.paper.analysis.run gradation \
  --raw-root /path/to/existing/raw --origin original-paper \
  --database /path/to/reanalysis/stats-v3.sqlite --plan-only

python -m experiments.paper.analysis.run gradation \
  --raw-root /path/to/existing/raw --origin original-paper \
  --database /path/to/reanalysis/stats-v3.sqlite --limit 1 --concurrency 1

python -m experiments.paper.analysis.run gradation \
  --raw-root /path/to/existing/raw --origin original-paper \
  --database /path/to/reanalysis/stats-v3.sqlite --concurrency 6
```

`--limit 1` 的成功文档可在全量运行时复用。再分别用 `epsilon` 和 `replicate` 运行同样命令；三个模式可以写入同一个新统计库。已有的 `MODE INSTANCE_NAME` 调用仍读取 `sim_ws.json`。`--force` 使当前修订下的文档重新计算。失败或缺失任务列在运行状态中，成功条目可恢复，不把 partial 称为完整重算。

默认重算还在同一原始轨迹上重建历史口径，依次区分基线、帖子/边、JS 参考方向、distance/divergence、终点补项和采样覆盖的变化。重建使用本轮确定性采样，不保证与未保留采样信息的旧成品数值完全相同；缺失的帖子观察不补造。`--skip-metric-audit` 仅适合不需要该历史比较的独立分析。

## 核对分组及依赖分组的数字

```sh
python -m experiments.paper.pathway_index.audit \
  --database /path/to/reanalysis/stats-v3.sqlite \
  --origin original-paper --analysis-id ANALYSIS_ID --cutoff 0.6 \
  --output /path/to/reanalysis/audit

python -m experiments.paper.pathway_index.grouped_summary \
  --database /path/to/reanalysis/stats-v3.sqlite \
  --origin original-paper --analysis-id ANALYSIS_ID --cutoff 0.6 \
  --output /path/to/reanalysis/audit
```

`ANALYSIS_ID` 来自 gradation 分析命令的返回值。audit 输出逐运行口径变化、0.6 分组迁移表和边/帖同信轨迹。可用 `--legacy /path/to/old/stats.db` 增加旧成品的只读对照；不要求旧统计库存在。

grouped_summary 只比较匹配的 `k_h=0` 两种 M9，输出逐运行值、组均值/标准差及有明确参照分母的百分比变化。指标覆盖换边数、归一化换边时刻、`I_s` 面积和闭合三元组数。它们是分组描述，不是配对干预效应。先查看样本数、缺失量和迁移结果，再判断旧分界及原论文百分比是否仍可使用。

## 绘图

旧的七个统计绘图入口均读取新文档库。使用默认 `stats-v3.sqlite`，或将 `SIMULATION_STAT_DIR` 指向包含该文件的目录。`EHK_ANALYSIS_ID` 可在同一模式存在多个分析版本时选择；`EHK_STATS_ORIGIN` 选择 raw 来源。不同模式的 analysis ID 不必相同。

```sh
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.main_scan.plot
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.pathway_index.plot_trajectories
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 \
  python -m experiments.paper.pathway_index.plot
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 \
  python -m experiments.paper.pathway_index.plot_raw
EHK_STATS_ORIGIN=original-paper EHK_IW_CUTOFF=0.6 \
  python -m experiments.paper.mechanism_cases.plot_cluster_and_repost
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.epsilon_scan.plot
EHK_STATS_ORIGIN=original-paper python -m experiments.paper.replicate.plot
```

主要算法比较固定 `k_h=0`；原始历史长度分组图仍保留单独的历史对照。图读取拒绝失效文档，不合并不同来源或指标修订。

## 若需要重新运行模拟

```sh
python -m experiments.paper.run gradation --plan-only
python -m experiments.paper.run gradation \
  --run-dir /path/to/new/raw --freeze-only
python -m experiments.paper.run gradation \
  --run-dir /path/to/new/raw --concurrency 4
```

`epsilon`、`replicate`、`mech` 使用相同入口。全部使用新的 raw 目录，任务名称和历史轴顺序不变；新运行显式记录独立 PCG 双种子、坐标、模型/推荐参数、采集设置、float64 精度和二进制/源码身份。不要用新种子身份续写旧无种子的同名模拟。完成代码修改后再冻结正式实验。

过程控制及网络图源采用独立模块，完整命令见 [控制实验说明](../experiments/paper/pathway_controls/README.md)。三臂 full、`q=0`、`alpha=0` 检查完整初始状态和 RNG 相同，导出实际有向边、意见及原帖；不需要 P/E 或路径阈值。小规模试跑只检查工作流，代表性正文图须从完成的科学规模轨迹选择。

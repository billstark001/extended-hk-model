# EHK 推荐实现的完整轨迹比较

本包维护 `recommender_variants` 的独立设计、调度、统计及图表入口。研究比较同一初态下整条轨迹使用不同推荐实现的结果。分阶段切换与在线时机比较由 [`recommendation`](../recommendation/README.md) 维护；运行与分析的底层代码由 [`recommendation_shared`](../recommendation_shared/README.md) 复用。

## 设计与数据兼容

设计沿用原来的九个 `(α,q)` 格点：`(0.01,0.30)`、`(0.03,0.10)`、`(0.05,0.05)`、`(0.10,0.03)`、`(0.30,0.01)`、`(0.01,0.10)`、`(0.03,0.30)`、`(0.10,0.01)`、`(0.30,0.03)`，各取 `p=0,0.1,0.25`，每格 16 个独立初态，共 432 组。每组使用 500 个节点、初始期望出度 15、`ε=0.45`、十个推荐位和 `k_h=0`，各臂运行 2048 步。

在 `(α,q,p)=(0.05,0.05,0.1)` 另有四项单因素敏感性：`ε=0.35/0.55`（保留 `k_h=0`）及 `k_h=2/6`（保留 `ε=0.45`）。各有 16 个重复，沿用对应核心格的初始化种子，共 496 组。半径和历史长度不交叉扩展，核心与敏感性结果分别报告。

原六臂为 Opinion、OpinionM9、OpinionRandom、Structure、StructureM9 和 StructureRandom。无后缀版本按各自分数排序；M9 请求一个均匀探索位和九个排序位；`Random` 后缀按各自分数加权抽样，额外均匀混合权重为零。各臂共享初始化图、意见、帖及 RNG 状态，版本比较的初始化种子独立于阶段与时机实验。

OpinionRandom 的 `ζ∈{1,4} × σ∈{0,2}` 因子设计对应 `Steepness` 与 `OpRandomNoiseStd`。原 `OpinionRandom` 即 `ζ=1,σ=2`，另三臂为 `OpinionRandomZeta1Noise0`、`OpinionRandomZeta4Noise0`、`OpinionRandomZeta4Noise2`。相对于原六臂，扩展增加 `496×3=1488` 条轨迹，组内共九臂。推荐容差 `Tolerance=0.4` 与置信半径 `ε` 分开保留。

OpinionRandom 先计算 `b=max(1−|x_i−x_j|/0.4,0)`，再以 `max(b+(1−2b)η,0)^ζ` 加权抽样，其中 `η~N(0,σ²)`。`σ` 是推荐分数噪声；提高 `ζ` 也会强化噪声产生的高权重。噪声开关会改变后续随机数消耗，配对要求同一初始化及 RNG，而不要求运行中的 RNG 状态持续相同。

拆包保留已有数据目录及布局、`study="paper_recommendation"`、`profile="recommender_variants"`、计划 schema、task ID、参数坐标和种子。种子 namespace 固定为历史值 2，不依赖研究注册顺序。冻结计划中的历史 `campaign.order` 字段仅为维持既有指纹而保留；当前运行由本包独立调度。已有六臂计划仍按各自冻结协议解析，逐臂二进制来源继续保留。

## 入口与恢复

从 EHK 仓库预览计划，两种入口都只输出设计：

```sh
python -m experiments.paper.run recommender_variants --plan-only
python -m experiments.paper.recommender_variants.study --plan-only
```

使用 `sim_ws.json` 工作区配置运行，或者直接指定该研究的数据目录：

```sh
python -m experiments.paper.run recommender_variants --workspace ssd_tmp --concurrency 6
python -m experiments.paper.recommender_variants.study \
  /path/to/recommender_variants --smp-binary /path/to/smp --concurrency 6
```

工作区入口使用 `SMP_BINARY_PATH` 或 `--smp-binary` 配置模拟器；直接研究入口要求 `--smp-binary`。数据仍写入原来的 `recommender_variants` 目录。中断后以相同命令恢复，已完成且通过原始记录校验的臂复用。恢复沿用冻结计划和绑定的二进制 SHA；不要同时对同一数据集运行两个调度器。

原始持久记录保存在 `raw/<task>/full/<arm>`，包含 float64 意见轨迹和意见和、逐用户计数、换边与转发事件、模型快照、元数据及 `smp.log`。`AgentNumber`、`OpinionSum`、`RewiringEvent`、`PostEvent` 开启，`ViewPostsEvent` 和 `ViewPostsDetails` 关闭；各观测间隔保存曝光计数。

新增因子臂或替换运行二进制时，先停止调度器，再显式迁移冻结计划。共享迁移入口可以处理工作区根目录或单项研究；更换二进制时必须指定该项研究目录：

```sh
python -m experiments.paper.recommendation_shared.migration \
  /Volumes/Cache/ehk_cache/run/recommender_variants \
  --smp-binary /Volumes/Cache/ehk_cache/run/recommendation_runtime/bin/smp-opinion-zero-row-20261120
```

该历史二进制修正 OpinionRandom 分数全零时沿用上一轮缓存的行为。迁移保留旧计划、旧绑定和凭据；旧六臂的来源保留，新增三臂记录修正版本的来源，任务 ID、种子、既有结果和原始记录不变。迁移与数据包导出均使用共享包的唯一入口。

## 分析与图表

```sh
python -m experiments.paper.recommender_variants.plot /path/to/recommender_variants
python -m experiments.paper.recommender_variants.plot /path/to/recommender_variants \
  --output /path/to/analysis --tables-only
```

分析默认写入数据目录下的 `analysis/`。输出逐运行指标、轨迹、配对差、覆盖率与来源信息，并另外生成 `arm_coverage.csv`、`opinion_random_factorial.csv` 和完整格点的因子对照表。因子表报告同初态配对的四项 `ζ`／噪声简单效应及二者交互效应。

各对照只使用所需臂均完成的配对样本；整组完成数要求该冻结计划中的臂齐备。旧六臂数据包按六臂协议判断，不被当前九臂设计重新解释。核心与各敏感性条件分别写入 `core_e45_kh0/`、`epsilon_e35_kh0/` 等目录；`complete_cells/` 汇总只含重复齐备的格点。

分析读取已保存的 `P`、`E`、16 步首达分类、置信分量终态和 KDE 意见峰数，各量保持原定义。它们与历史 `I_h,I_p,I_s,I_w` 的定义分别维护。`coverage.json` 保存计划、记录 SHA、二进制绑定和分析时刻；逐臂来源同时进入覆盖率及运行表。

## 可移植数据包与因子比较图

`recommendation_shared.bundle export` 验证并复制已有摘要，调用本包分析，将表格压缩到 `data/tables/`。`--require-complete` 要求冻结协议中的全部臂完成；中途导出可省略该选项。导出只描述复制得到的那一代记录，不启动模拟。

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
factorial_bundle=outputs/paper/recommendation_manuscript/opinion_random_factorial_complete
python -m experiments.paper.recommendation_shared.bundle export \
  --dataset /Volumes/Cache/ehk_cache/run/recommender_variants \
  --output "$factorial_bundle" --require-complete
python -m experiments.paper.recommender_variants.variants_review prepare \
  --bundle "$factorial_bundle" --output "$factorial_bundle/plots"
python -m experiments.paper.recommender_variants.variants_review compose \
  --directory "$factorial_bundle/plots"
```

`prepare` 从本地数据包选择四种 OpinionRandom 配置齐备的同一初态队列，原算法对照也使用这些初态，分别处理核心和敏感性条件。它写出 `matched_*` 与 `paired_vs_m9*` 表、`plot_data.json` 和可编辑的 `layout.json`，保留已存在的布局。`compose` 只读保存的绘图数据与布局，生成 PDF、SVG、PNG 和来源清单；此时原始数据卷可以断开。

从已导出的摘要重建逐格点图：

```sh
python -m experiments.paper.recommender_variants.plot \
  "$factorial_bundle/data/study" --output "$factorial_bundle/detail"
```

实现比较的设计、运行、读取、分析与因子图统一使用本包入口；阶段包拒绝读取实现比较数据。数据包导出与冻结计划迁移统一由 [`recommendation_shared`](../recommendation_shared/README.md) 提供。

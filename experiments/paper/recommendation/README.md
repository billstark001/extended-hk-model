# EHK 推荐实验

## 设计

核心设计取五个 `(α, q)` 格点 `(0.01, 0.30)`、`(0.03, 0.10)`、`(0.05, 0.05)`、`(0.10, 0.03)`、`(0.30, 0.01)`，各取 `p=0, 0.1, 0.25`。历史主扫描每个独立设定有 12 个重复，本轮每个设定有 16 个独立初态，共 240 组。使用 500 个节点、初始期望出度 15、`ε=0.45`、十个推荐位与 `k_h=0`。组内各臂共享图、意见、帖和初始化 RNG。

`staged` 与 `recommender_variants` 各增加四项单因素敏感性：仅在 `(α,q,p)=(0.05,0.05,0.1)`，分别取 `ε=0.35/0.55`（保留 `k_h=0`）及 `k_h=2/6`（保留 `ε=0.45`）。每项仍有 16 个重复，并沿用对应核心组的初始化种子；不交叉扩展半径和历史长度。两项研究各有 304 组。`staged_timing` 只包含 240 个核心组，使用独立于阶段研究的初始化种子。设定标识、表格和图均区分半径、历史长度及敏感性类别。

- `staged` 用 OpinionM9 生成参考轨迹。每 16 步观测一次，在设计表指定的候选步写检查点。早期取两项指标都低于阈值的首个候选点。中期取早期之后至少一项指标过线、且尚未符合晚期判据的首个候选点。晚期从第 128 步起，取最近两个观测间隔的每次更新平均绝对意见位移不超过 0.001、完成换边率不超过 0.01 的首个候选点。每个可用检查点配对比较保持 OpinionM9 和切换到 StructureM9 后的 512 步。两臂请求相同的推荐位与探索位。阶段缺失计入覆盖率。
- 晚期另有两项诊断。桥接臂在可用时提供一条进入置信范围的跨阵营当前原创帖，可以填空或替换末位。转发抑制臂在 `p>0` 下屏蔽由转发产生的候选展示位置（含用户转发自己的旧帖），继续按相同的 `p` 生成转发。主对照使用各政策自己的候选生成和排序。
- `staged_timing` 使用独立初态。阶段研究的 240 个核心组全部完成后，只用核心组中同时具有早、晚检查点的同组重复计算「晚期切换效应减早期切换效应」。512 步意见分离效应差的绝对均值至少为 0.05 且 95% t 区间排除零时进入本研究；门控未通过就不运行 timing。敏感性组不进入门控。两种触发均切换到 StructureM9 恰好 512 步，再恢复 OpinionM9。固定触发点为 128 步，状态触发取过量同信首次达到 0.10 的时刻，至 1536 步仍未达线就触发。另有两种一直使用的政策作为对照。状态触发相对于固定触发的主要目标是降低第 2048 步的意见分离。
- `recommender_variants` 将 Opinion、OpinionM9、OpinionRandom、Structure、StructureM9 和 StructureRandom 从同一个初始检查点分别运行 2048 步。无后缀版本按各自分数排序，M9 请求一个均匀探索位和九个排序位，`Random` 后缀按各自的分数加权抽样。加权版本沿用默认噪声和权重，独立均匀混合权重设为零。图表比较同族版本与 M9 的差异，也比较同后缀的 Opinion 与 Structure。

## 指标

- 意见分离取两段宽度至多 `ε/2`、间距严格大于 `ε` 的意见区间，最大化较小区间的人数占比，再乘二。首达线为 0.40。过量边同信率等于当前有向边同信率减去当前所有不同节点对的同信率，保留符号，首达线为 0.10。首达按各自首次达到观测线的记录步判断，保留同一步过线及到窗口终点仍有指标未过线的情形。
- 跨阵营曝光统计实际展示的推荐位置。接收者和帖的意见要在零点两侧，且都距零超过 `ε/4`。置信资格按贴文意见和接收者当前意见的差计算。换边资格还要求有不同信的关注帖及可新增的作者边。晚期图单独展示排序位，桥接臂的排序位也包含干预插入的帖。结果保留插入次数、填充率、完成换边数和总绝对意见位移。
- 历史统计中，`I_h` 采用有正出度用户的边端点同信率均值，随机初态基线为 `ε−ε²/4`。关注帖同信率使用该步实际接收的关注帖，另行记录。`I_p` 与 `I_s` 使用相对于随机参考的 Jensen–Shannon 散度比。`I_w` 按轨迹实际顺序积分，去掉末点补项。历史 0.6 分组可用旧库与首达分类交叉核对。
- 终点报告置信分量分类和 KDE 意见峰数。置信分量无法归入吸收态时记为 unresolved。阶段内对照报告配对差，首达判据和终点分类各自保留。阶段轨迹的阴影为均值的一个标准误，终点差和跨阶段差报告 95% t 区间。

正式实验使用持久化 `smp`。每臂的 `raw_record` 指向数据集下的 `raw/<task>/<phase>/<arm>`，保存 float64 意见轨迹和意见和、逐用户计数、换边及转发事件、模型快照和元数据。`AgentNumber`、`OpinionSum`、`RewiringEvent`、`PostEvent` 均开启，`ViewPostsEvent` 与 `ViewPostsDetails` 关闭，避免在 SQLite 重复存储每次 feed。分叉臂从同一初始化重放参考前缀，再执行相应干预；切换前核对检查点的物理状态及 RNG，因此原始轨迹可从 0 完整读取。前缀不进入干预窗口的曝光分母。

每个观测窗口保留实际请求数、填充数和完成换边数。探索位、各算法自身提供的帖和桥接插入帖分别统计展示、跨阵营、置信与换边资格。晚期各臂另外写出后续窗口内压缩的逐用户、逐槽 feed 文件，保存更新前意见、请求数、有序邻居槽位和推荐槽位，以及每槽的原帖、展示者、展示轮次和来源。邻居槽位包含没有帖子的 followee；转发、置信和换边资格由这些原始事实与模拟参数重算。记录中的路径相对于数据集目录，feed 文件可通过 `ehk.micro.batch.iter_exposure_trace` 顺序读取。新实验只接受当前设计版本和完整曝光字段，不为旧批处理字段补值。

历史统计及离线状态测量通过 `SMP_PROBE_BINARY` 配置的 `smp-probe` 二进制计算共享路径指标，按记录的意见精度重算。已有的逐用户计数或完整的 ViewPosts 事件支持关注帖同信计算。观测窗口有缺口时，首达分类可以是 `unavailable`。

## 入口

在 EHK 仓库预览设计。该命令只输出计划。

```sh
python -m experiments.paper.run staged --plan-only
python -m experiments.paper.run staged_timing --plan-only
python -m experiments.paper.run recommender_variants --plan-only
```

在 SMP 仓库运行 `make build build-probe`，分别构建持久模拟器与离线测量程序。`SMP_BINARY_PATH` 或 `--smp-binary` 配置模拟器，`SMP_PROBE_BINARY` 单独配置离线程序。在 EHK 仓库选用 `sim_ws.json` 里的工作区名。

```sh
python -m experiments.paper.run recommendation --workspace ssd_tmp --concurrency 6
python -m experiments.paper.recommendation.plot /path/to/dataset
```

完整 campaign 由一个 Python 调度器依次执行 `staged`、`recommender_variants`，再在门控通过时执行 `staged_timing`；全局最多六个 native worker。输出分别位于工作区下同名目录，campaign 状态写入 `recommendation_runtime/status.json`。也可用 `staged` 等研究名单独运行，但不要同时启动多个六进程调度器。

调度前先在目标卷运行一个 8 节点、2 步的持久化检查，验证原生事件库、轨迹和快照能够写入和重开。检查失败会立即停止，不会继续提交实验任务。SMP 事件库使用 SQLite WAL；运行中的数据库必须连同其 `-wal`、`-shm` 文件保留，正常关闭后再归档。

用 `kill -TERM PID` 中断后，重跑完全相同的命令即可恢复：每臂保留持久快照，已完成且通过记录校验的任务会跳过。恢复必须沿用冻结设计与绑定的二进制 SHA；改变参数或程序后应使用新数据集目录。不要同时对同一数据集启动两个调度器。

分析默认输出到数据目录下的 `analysis/`，生成逐运行指标、轨迹、配对差和逐设定的完成及阶段覆盖率。核心与各敏感性条件的图分别写入 `core_e45_kh0/`、`epsilon_e35_kh0/` 等子目录；不同 `ε` 或 `k_h` 不合并。历史数据可从原始工作区重算到 `stats-v2.db`。旧库由审计命令按场景名和工作区来源读取。

```sh
python -m experiments.paper.analysis.run gradation WORKSPACE
python -m experiments.paper.pathway_index.audit --origin WORKSPACE
python -m experiments.paper.main_scan.plot
```

持久化 SMP 运行用 `ViewPostsDetails=true` 采集逐槽详情。仅设置 `ViewPostsEvent=true` 时记录原有置信分组。旧 ViewTweets 表先用 `smp-migrate events` 显式迁移。

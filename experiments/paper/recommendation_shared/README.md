# 推荐实验的共享运行与分析代码

本包提供分阶段干预和完整轨迹实现比较共同使用的底层功能。科学设计与调度分别由 [`recommendation`](../recommendation/README.md) 和 [`recommender_variants`](../recommender_variants/README.md) 持有。

- `runtime`：持久化观测、检查点状态及 RNG 配对、逐臂记录复用、计划和绑定校验、调度与存储检查。
- `analysis`：读取冻结计划及已有记录、指标表、配对统计、曝光计数与覆盖率辅助功能。
- `plot`：共同图表样式、覆盖率及效果图辅助功能。
- `bundle`：复制及验证摘要、分派到对应研究的分析，保留来源和压缩表格。
- `migration`：冻结计划的显式迁移、历史计划和逐臂二进制绑定的保留。

本包不定义独立实验协议；导出和迁移命令只处理已有计划及记录。新增政策、参数格、干预规则或比较逻辑放入所属研究包；共享代码接收研究提供的协议与回调。数据包导出根据冻结计划的 `profile` 选择分析器。

代码拆分保留原有数据布局、计划 schema、study/profile 字段、task ID、种子和逐臂二进制来源。两个研究包各自拥有设计、运行和分析入口，不保留跨包的兼容别名；已有数据不需要因代码搬迁而改名或重新模拟。历史计划中的 campaign 元数据仅维持冻结指纹，运行范围以当前研究入口为准。

## 维护命令

```sh
python -m experiments.paper.recommendation_shared.bundle export \
  --dataset /path/to/dataset --output /path/to/bundle --require-complete
python -m experiments.paper.recommendation_shared.migration /path/to/workspace
```

`bundle export` 根据冻结 `profile` 选择所属分析器，复制并验证已保存的摘要，不启动模拟。`--require-complete` 要求冻结协议中的全部任务及臂完成；运行中导出可省略它。

`migration` 支持工作区根目录或单研究目录，先验证全部已有任务、记录和绑定，再保存旧计划及迁移凭据。替换二进制必须指定 `recommender_variants` 单研究目录，并传 `--smp-binary /path/to/smp`；原六臂和新增因子臂继续分别保留各自来源。迁移前须停止对应调度器。

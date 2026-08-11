## 2026-08-11（二）：修复三条「会误导读者」的结构性问题，并更正上一轮审计自身的过度结论

**Type**: Correction（三条均出自 2026-08-10 的两域审计第一层。本次核对 verl 官方文档后，发现**审计自身的一条结论说过头了**，一并更正。）

### 一、verl v1/TransferQueue：补文档级覆盖，并收回"整簇失效"的过度判断

- **新增 [[16_verl_v1_transfer_queue_analysis]]**（文档级，**非源码级**，页头已显著标注）：厘清被混为一谈的三个概念、TransferQueue 的三层架构与四种存储后端（含昇腾原生的 Yuanrong 分层后端）、版本状态、以及**对本簇 9 篇影响范围的逐条界定**。
- **核对官方文档后的两处关键发现**：
  1. **`use_v1`、`TaskRunnerV1`、`trainer/ppo/v1/*` 在 verl 官方文档与 v0.7/v0.8 release notes 中均查无此名**——它们是源码树内部命名。本簇对它们的记述唯一依据是自身的源码观察（`ppo_trainer.yaml` 行号双向记录），不能引官方文档背书。官方侧可查证的对应事实是入口更名："`main_ppo.py` is deprecated with a warning in favor of `main_ppo_sync.py`"（v0.8.0 release notes）。
  2. **TransferQueue 至今仍不是默认传输方式**。v0.7 blog 称"计划在 v0.8 成为默认"，但 v0.8.0 实际只交付了 "New sync trainer with TransferQueue"，并注明 "TBD: Fully async trainer with TransferQueue will be in next release"。凡"verl 已默认走 TransferQueue"的说法无官方依据。
- **更正 2026-08-10 审计自身的结论**：该审计称「verl 簇 7 篇深潜写的是 legacy 路径，整簇有效性存疑」——**这个推论过度**。受影响的只是**编排层与数据搬运层**（`RayPPOTrainer.fit` 主链、以及"数据流经 driver"这一前提）；**计算面**（[[13_verl_workers_engine_analysis]]）、**权重面**（[[14_verl_rollout_resharding_analysis]]）、**算法面**（[[15_verl_rl_algorithms_analysis]]）、**数据契约**（[[12_verl_dataproto_analysis]]，官方称经 `RemoteBatch` 与 TransferQueue 兼容共存）均不受影响。逐条对照表见新页 §5。
- **更新 [[verl/index]]**：重写架构演进提示，加 `[!warning]` 记录上述两处更正；把新页加入页面列表。**更新父索引** `04_posttrain_frameworks/index.md` 的 verl 篇数说明。

### 二、[[30_rl_framework_comparison]]：把"有效期临近"落到具体过期项

上一轮只加了到期提醒。本次从官方文档侧做了部分重验，**确证 verl 列至少两项已过期**（入口脚本更名、数据面新增 TransferQueue 通路），同时**确证一项"未过期"以免反向误判**（TransferQueue 未成为默认，故把 verl 数据面记为 `DataProto` 在默认路径上仍正确、只是不再完整）。并如实记录重验障碍：四框架 commit 比对需访问 GitHub，本次环境无法访问，**slime / AReaL / ROLL 三列本次未做任何重验**。

### 三、[[12_rl_infra_efficiency_analysis]]：重写长尾数学，并给全页数字定级

- **长尾治理的"12×"推导已重写**。原文用单条轨迹的 p99 代表"等 batch 全部完成"，但等全部完成是 $X_{(N)}=\max$，其典型分位为 $F^{-1}(1-\frac{1}{N+1})$——$N=128$ 时约 **p99.2 而非 p99**，重尾下差距更大，**误差方向是低估收益**。结论量级（≥12×）站得住，推导过程不成立，现按次序统计量重写。
- **区分了两种被混用的截尾口径**：「等前 90%」是 $X_{(116)}\approx$ p90；「发 $N{+}K$ 取前 $N$」（发 160 取 128）等的是 $F^{-1}(128/161)\approx$ **p80**，更快但多付 25% sandbox。原文 A、B 两小节混用了这两种。
- **撤回"几乎免费"的说法**。真正的代价不是浪费的 CPU，而是**选择偏差**——被丢弃的永远是最慢的那批，而 coding RL 里最慢的轨迹系统性地就是最难的任务；持续按完成时间截尾等于在训练分布里删除难题。[[01_posttraining_infra_mechanism_analysis]] §10 已就此警告过，本页此前未回应，现补记并给出监控建议。
- **新增全页数据口径说明**：把正文数字分三级——⚠️ 无出处经验区间（GPU 利用率四档、"30%+"、"20-30%"、环境池四档、三段占比假设）、⚠️ 舆情/转述级（带"据说"的表述）、✅ 有一手出处（AReaL `staleness_manager.py:80-112`、K3 §5.3.1 p.21）。并点明**本页是本目录被引最多、也最不可核验的一页**，这个组合本身就是风险。

### 四、校验

两目录 46 个文件 wikilink **未解析目标 0 种**；代码块与 LaTeX 配对正常；以 LF 行尾写回。

---

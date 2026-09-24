---
title: "Mooncake：按问题与依赖组织的源码地图"
---

# Mooncake：按问题与依赖组织的源码地图

> **核验基线**：`kvcache-ai/Mooncake@7d3a94e9d8c30abf02fcd64df218c16c1abc70df`（`main`，2026-09-24）；与 vLLM 的联动页另钉 `vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，与 vLLM 域一致。
> **目录范围**：6 篇内容页 + 本索引。本页只维护本级入口、阅读依赖与覆盖边界；各页的适用范围与证据限制见其页头。
> **最后更新**：2026-09-24。新建子域，取代原论文分析页 `mooncake_analysis`；论文层面的 P/D 分离与 KV 分层原理仍归推理理论页。

## 已知覆盖边界

下列主题在当前基线下**本域没有 owner 页面**，在裁决前只作登记：

| 尚无 owner 的主题 | 说明 |
|---|---|
| TENT 运行时内部（`mooncake-transfer-engine/tent/`） | 传输引擎页只拥有它的选择边界与相对经典路径的差异；PyPI wheel 默认不启用 |
| 各厂商 transport（NVLink/MNNVL、Ascend、EFA、CXI、HIP、MUSA 等） | 传输引擎页只列边界 |
| SGLang、LMCache、TensorRT-LLM 等非 vLLM 集成 | SGLang PD 在 slime 域有使用层记录 |
| Mooncake EP / PG、P2P Store、reshard、RL demo | 架构总览只给一行边界；与 KV serving 数据面无直接关系 |
| 配置面逐项归属 | master gflags 与传输引擎 `MC_*` 的逐项归属或排除理由记在 `docs/coverage/mooncake.md` |

## 读者入口

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[01_mooncake_architecture_overview_analysis|01 架构总览]] | 开源 Mooncake 由哪些层组成、各层持有什么状态？论文里的设计哪些开源了、哪些没有？`mooncake-conductor/` 实际是什么？ | 无；原理先读 [[26_prefill_decode_disaggregation_analysis|P/D 分离原理]] |

## 数据面机制

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[10_mooncake_transfer_engine_analysis|10 Transfer Engine]] | 一次批量写怎样经 segment 查找、切片、选网卡与 QP 写进对端注册内存，何时算完成，失败在哪一层重试？TENT 何时生效？ | 架构总览 |
| [[11_mooncake_store_object_lifecycle_analysis|11 Store 对象生命周期]] | 对象从 PutStart 到可读、在租约下被读、再被淘汰，master 与 client 各持有什么状态？空间不够或客户端掉线时怎样？ | 架构总览 → Transfer Engine |
| [[12_mooncake_store_tiering_offload_analysis|12 Store 分层与卸载]] | 内存放不下时对象怎样卸载到 SSD/DFS、命中时怎样提升回内存，谁派发谁执行？ | Store 对象生命周期 |
| [[13_mooncake_store_ha_recovery_analysis|13 Store 高可用与恢复]] | master 故障后谁接管、元数据从哪里恢复、客户端在切换窗口看到什么？ | Store 对象生命周期 |

## 与推理引擎的联动

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[20_mooncake_vllm_integration_analysis|20 vLLM 集成]] | vLLM 两个 Mooncake connector 建了哪些 Mooncake 对象，每个跨边界调用在 Mooncake 内部做了什么，完成、失败与寿命语义怎样对上？ | Transfer Engine、Store 对象生命周期；vLLM 侧协议见 [[22_vllm_disaggregated_kv_serving_analysis|vLLM 分离式 KV Serving]] |

## Related Pages

- [[02_engineering/03_infer_frameworks/index|推理框架目录]] — 回到推理框架的本级入口。
- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] — vLLM 侧的 connector 协议、调度与可观测性。
- [[01_theory/05_inference/index|推理技术理论]] — P/D 分离与 KV 分层迁移的原理层。
- [[23_kimi_k3_infra_deepdive|Kimi K3 训推基础设施]] — Mooncake 在 Kimi 线上服务中的位置。

# Mooncake 配置面覆盖台账

> **源码基线**：`kvcache-ai/Mooncake@7d3a94e9d8c30abf02fcd64df218c16c1abc70df`（`main`，2026-09-24）
> **用途**：`wiki/02_engineering/03_infer_frameworks/mooncake/` 的独立枚举轴。每个枚举项归属一个 owner 页或写明排除理由。`tools/check_coverage.py` 只枚举 Python dataclass，这里的 C++ gflags 与环境变量由一次性脚本枚举后人工对账。
> **枚举方法**：master gflags = `mooncake-store/src/master.cpp` 中 `DEFINE_{bool,int32,int64,uint32,uint64,double,string}` 的名字（102 个，另有 8 个 `DEFINE_validator` 不计）；TE 环境变量 = `mooncake-transfer-engine/src/config.cpp::loadGlobalConfig` 读取的 `MC_*` 字面量（56 个，含拼写别名 `MC_MIN_PRC_PORT`/`MC_MAX_PRC_PORT`）。“页内提及”列由脚本按整词匹配 6 篇正文得到。
> **最后更新**：2026-09-24。

## A. vLLM 调用面（15 项，owner 全部为 20）

| 项 | owner | 说明 |
|---|---|---|
| `initialize` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `get_rpc_port` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_register_memory` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_transfer_sync_write` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `setup` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `register_buffer` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_put_from_multi_buffers` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_get_into_multi_buffers` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_is_exist` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `batch_get_replica_desc` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `remove_all` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `close` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `ReplicateConfig.preferred_segment` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `ReplicateConfig.group_ids` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |
| `错误码 -200 (NO_AVAILABLE_HANDLE)` | 20_mooncake_vllm_integration_analysis | §7 跨边界表逐行映射到 Mooncake 内部 |

## B. master gflags（102 项）

| gflag | owner | 族 / 排除理由 | 页内状态 |
|---|---|---|---|
| `allocation_strategy` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `allow_evict_soft_pinned_objects` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `batch_oplog_retry_timeout_sec` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `client_active_ttl_sec` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `client_suspicion_ttl_sec` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `client_ttl` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `cluster_id` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `config_path` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `cxl_path` | 排除 | CXL 共享内存段：需专门硬件，11 页只给一行边界 | — |
| `cxl_size` | 排除 | CXL 共享内存段：需专门硬件，11 页只给一行边界 | — |
| `default_kv_lease_ttl` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `default_kv_soft_pin_ttl` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `dynamic_replication_admission_qps_threshold` | 排除 | 动态复制：实验性（off/observe/enforce），01 页只列能力边界 | — |
| `dynamic_replication_heat_window_seconds` | 排除 | 动态复制：实验性（off/observe/enforce），01 页只列能力边界 | — |
| `dynamic_replication_max_memory_replicas` | 排除 | 动态复制：实验性（off/observe/enforce），01 页只列能力边界 | — |
| `dynamic_replication_mode` | 排除 | 动态复制：实验性（off/observe/enforce），01 页只列能力边界 | — |
| `enable_cxl` | 排除 | CXL 共享内存段：需专门硬件，11 页只给一行边界 | — |
| `enable_disk_eviction` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `enable_ha` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `enable_http_metadata_server` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `enable_kv_events` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `enable_metadata_cleanup_on_timeout` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `enable_metric_reporting` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `enable_multi_tenants` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `enable_offload` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `enable_oplog` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `enable_oplog_snapshot` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `enable_snapshot` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `enable_snapshot_restore` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `etcd_endpoints` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `eviction_high_watermark_ratio` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `eviction_ratio` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `global_file_segment_size` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `ha_backend_connstring` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `ha_backend_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `http_metadata_server_host` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `http_metadata_server_port` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `kv_events_additional_salt` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_backend_id` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_bind_endpoint` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_block_size` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_dp_rank` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_emit_legacy_compat` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_emit_object_key` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_lora_name` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_model_name` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `kv_events_queue_capacity` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `kv_events_tenant_id` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 按族覆盖（页内未逐项列出） |
| `max_kv_soft_pin_ttl` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `max_retry_attempts` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `max_threads` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `max_total_finished_tasks` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `max_total_pending_tasks` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `max_total_processing_tasks` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `memory_allocator` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `metrics_host` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `metrics_port` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `nof_eviction_high_watermark_ratio` | 排除 | NVMe-oF 副本：需 `USE_NOF` 构建（默认关闭），11/12 页只给一行边界 | — |
| `nof_eviction_ratio` | 排除 | NVMe-oF 副本：需 `USE_NOF` 构建（默认关闭），11/12 页只给一行边界 | — |
| `nof_heartbeat_failures_threshold` | 排除 | NVMe-oF 副本：需 `USE_NOF` 构建（默认关闭），11/12 页只给一行边界 | — |
| `nof_heartbeat_interval_sec` | 排除 | NVMe-oF 副本：需 `USE_NOF` 构建（默认关闭），11/12 页只给一行边界 | — |
| `nof_heartbeat_probe_timeout_ms` | 排除 | NVMe-oF 副本：需 `USE_NOF` 构建（默认关闭），11/12 页只给一行边界 | — |
| `offload_cap_ratio` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `offload_force_evict` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `offload_on_evict` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `offloading_queue_limit` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `oplog_batch_max_entries` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `oplog_poll_interval_ms` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `pending_task_timeout_sec` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `pod_name` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `pod_namespace` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `port` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `processing_task_timeout_sec` | 排除 | copy/move/drain 管理任务队列：管理面功能，11 页只给一行边界 | — |
| `promotion_admission_threshold` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `promotion_max_per_heartbeat` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `promotion_on_hit` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `promotion_queue_limit` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `put_start_discard_timeout_sec` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `put_start_release_timeout_sec` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `quota_bytes` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `root_fs_dir` | 12_mooncake_store_tiering_offload_analysis | 分层、卸载与提升 | 页内提及 |
| `rpc_address` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `rpc_conn_timeout_seconds` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `rpc_enable_tcp_no_delay` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `rpc_interface` | 排除 | RPC 与 metrics 进程管线参数：不改变对象语义 | — |
| `rpc_port` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `rpc_thread_num` | 01_mooncake_architecture_overview_analysis | 进程入口、内嵌 HTTP 元数据服务与 KV events | 页内提及 |
| `snapshot_backup_dir` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_catalog_backend_connstring` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_catalog_backend_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_catalog_store_connstring` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_catalog_store_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_child_timeout_seconds` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_chunk_object_count` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_interval_seconds` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_object_store_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_payload_backend_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_payload_store_type` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `snapshot_retention_count` | 13_mooncake_store_ha_recovery_analysis | HA、OpLog 与快照 | 页内提及 |
| `tenant_eviction_high_watermark_ratio` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `tenant_quota_connector_type` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |
| `tenant_quota_connector_uri` | 11_mooncake_store_object_lifecycle_analysis | 对象生命周期、租约、淘汰与存活 | 页内提及 |

合计：01=21，11=18，12=13，13=26，excl=24。

## C. Transfer Engine `loadGlobalConfig` 环境变量（56 项）

| 变量 | owner | 排除理由 |
|---|---|---|
| `MC_AUTO_GID_MAX_RETRIES` | 10_mooncake_transfer_engine_analysis | — |
| `MC_CONN_PAUSE_TTL_MS` | 10_mooncake_transfer_engine_analysis | — |
| `MC_CONTEXT_PAUSE_TTL_MS` | 10_mooncake_transfer_engine_analysis | — |
| `MC_DISABLE_METACACHE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_EFA_NIC_SELECTION` | 10_mooncake_transfer_engine_analysis | — |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | 10_mooncake_transfer_engine_analysis | — |
| `MC_ENABLE_HCA_PEER_AFFINITY` | 10_mooncake_transfer_engine_analysis | — |
| `MC_ENABLE_PARALLEL_REG_MR` | 排除 | 内存注册并行度：只影响注册耗时 |
| `MC_ENDPOINT_STORE_TYPE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_FRAGMENT_RATIO` | 10_mooncake_transfer_engine_analysis | — |
| `MC_GID_INDEX` | 10_mooncake_transfer_engine_analysis | — |
| `MC_HANDSHAKE_CONNECT_TIMEOUT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_HANDSHAKE_LISTEN_BACKLOG` | 排除 | 握手 socket 的部署参数；握手机制本身归 10 |
| `MC_HANDSHAKE_PORT` | 排除 | 握手 socket 的部署参数；握手机制本身归 10 |
| `MC_IB_PCI_RELAXED_ORDERING` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_IB_PORT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_IB_SL` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_IB_TC` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_LOG_DIR` | 10_mooncake_transfer_engine_analysis | — |
| `MC_LOG_LEVEL` | 10_mooncake_transfer_engine_analysis | — |
| `MC_LOG_RDMA_SLICE_AFFINITY` | 排除 | 调试日志开关 |
| `MC_MAX_CONCURRENT_REG_MR` | 排除 | 内存注册并行度：只影响注册耗时 |
| `MC_MAX_CQE_PER_CTX` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_EP_PER_CTX` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_INLINE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_MR_SIZE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_PRC_PORT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_RPC_PORT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_SGE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MAX_WR` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MIN_PRC_PORT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MIN_REG_SIZE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MIN_RPC_PORT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_MLX5_QP_LAG_PORT_BALANCE` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_MLX5_QP_UDP_SPORTS` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_MTU` | 10_mooncake_transfer_engine_analysis | — |
| `MC_NIC_PEER_AFFINITY` | 10_mooncake_transfer_engine_analysis | — |
| `MC_NUM_COMP_CHANNELS_PER_CTX` | 排除 | 事件驱动完成（`USE_EVENT_DRIVEN_COMPLETION`，默认关闭） |
| `MC_NUM_CQ_PER_CTX` | 10_mooncake_transfer_engine_analysis | — |
| `MC_NUM_QP_PER_EP` | 10_mooncake_transfer_engine_analysis | — |
| `MC_PKEY_INDEX` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_RDMA_NOTIFY_BUFFER_SIZE` | 排除 | notify 通道细节：notify 不在 vLLM 与 Store 主路径；10 只拥有其发送位置与回退规则 |
| `MC_RDMA_NOTIFY_CONNECT_TIMEOUT_MS` | 排除 | notify 通道细节：notify 不在 vLLM 与 Store 主路径；10 只拥有其发送位置与回退规则 |
| `MC_RDMA_NOTIFY_ENABLED` | 10_mooncake_transfer_engine_analysis | — |
| `MC_RDMA_NOTIFY_MAX_PENDING_SENDS` | 排除 | notify 通道细节：notify 不在 vLLM 与 Store 主路径；10 只拥有其发送位置与回退规则 |
| `MC_RDMA_NOTIFY_OOB_FALLBACK` | 排除 | notify 通道细节：notify 不在 vLLM 与 Store 主路径；10 只拥有其发送位置与回退规则 |
| `MC_RDMA_NOTIFY_RECV_COUNT` | 排除 | notify 通道细节：notify 不在 vLLM 与 Store 主路径；10 只拥有其发送位置与回退规则 |
| `MC_RDMA_RAIL_PAUSE_SECONDS` | 10_mooncake_transfer_engine_analysis | — |
| `MC_RETRY_CNT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_SLICE_SIZE` | 10_mooncake_transfer_engine_analysis | — |
| `MC_SLICE_TIMEOUT` | 10_mooncake_transfer_engine_analysis | — |
| `MC_TE_METADATA_REFRESH_INTERVAL_SECONDS` | 10_mooncake_transfer_engine_analysis | — |
| `MC_TRACK_RDMA_POSTED_SLICES` | 10_mooncake_transfer_engine_analysis | — |
| `MC_USE_IPV6` | 排除 | RDMA 链路层/网卡调优：透传给 verbs 属性，不改变切片、选卡、完成与重试语义 |
| `MC_USE_RDMA_TWOSIDED` | 10_mooncake_transfer_engine_analysis | — |
| `MC_WORKERS_PER_CTX` | 10_mooncake_transfer_engine_analysis | — |

合计：10=38，excl=18。

## D. 不在枚举轴内、由页面自行登记的配置

- `loadGlobalConfig` 之外的 TE 直读变量（`MC_FORCE_TCP`、`MC_FORCE_SHM`、`MC_TRANSFER_TIMEOUT`、`MC_USE_TENT` 等）与厂商 transport 变量：见 10 页配置节。
- Store 客户端配置键、`MOONCAKE_OFFLOAD_*`、`MOONCAKE_DFS_*` 等环境变量族：见 11、12 页配置节；`MOONCAKE_OFFSET_*`、`MOONCAKE_NVME_KV_*`、`MOONCAKE_OSS_*` 只在 12 页按族点名。
- vLLM 侧 `kv_connector_extra_config` 键与 `VLLM_MOONCAKE_*` 环境变量：归 vLLM 22 §11。

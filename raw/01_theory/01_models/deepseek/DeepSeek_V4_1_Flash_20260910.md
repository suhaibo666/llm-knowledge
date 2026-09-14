# DeepSeek-V4.1-Flash 官方发布与技术报告来源索引

> 仅新增来源链接、元数据与校验值；不随本库重新分发论文原文或外部代码。

| 项目 | 元数据 |
|---|---|
| 标题 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression |
| 作者 / 发布方 | DeepSeek-AI |
| 正式发布日期 | 2026-09-10，官方 API 更新日志 |
| 发布仓 | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d) |
| 冻结 revision | `fb2764a5cf321eaa5070ca8f9e892818f477c16d`；HF API lastModified：2026-09-10T06:25:21Z |
| 核对时间 | 2026-09-10，北京时间 |
| 报告版本 | 随冻结提交发布的 51 页 PDF；未核实独立 arXiv 编号 |
| 权重 / 仓库许可 | 模型卡标明 MIT；以发布仓 LICENSE 为准 |
| 本库分析 | `wiki/01_theory/01_models/deepseek/19_deepseek_v4_1_flash_analysis.md` |

## 冻结文件与 SHA-256

以下文件均从同一 revision 获取并读取。代码仅在临时只读分析目录中取证，不复制进 raw。

| 文件 / 原文入口 | SHA-256 |
|---|---|
| [DeepSeek_V41_Tech_Report.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/fb2764a5cf321eaa5070ca8f9e892818f477c16d/DeepSeek_V41_Tech_Report.pdf) | `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| [README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/README.md) | `94a04133ea0a65490881780180a8e671d674eec1dfa91e609f906a38fdcbf55e` |
| [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/config.json) | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/model.py) | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/kernel.py) | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| [inference/engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/engram.py) | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| [inference/vision.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/vision.py) | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| [inference/README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/inference/README.md) | `2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c` |
| [evaluation/README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/evaluation/README.md) | `b1367cba184ce632e1e24e4dfc29cf40b02fdc297ef316bb82599e38eba6dae0` |

## 动态 API 文档

以下为 2026-09-10 的读取记录，URL 内容可能随服务更新。摘要与冲突分析归 wiki 页面；散列用于识别本次读到的 HTML，不代表已归档完整网页。

| 官方文档 | 本次 HTML SHA-256 |
|---|---|
| [首次调用](https://api-docs.deepseek.com/zh-cn/) | `907662db5dab6c6a55867a0a3558fed07fc0c0a2121dc716e84f2427608a7a7a` |
| [更新日志：2026-09-10](https://api-docs.deepseek.com/zh-cn/updates/) | `862059b9b6db47b96aa6470604b6391e862878ec665f562f4cfc63d38c8edb7f` |
| [模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) | `a5f8f63d65665321cc44b4035b324a057de0e9a10c20ac28cff549a1fad7149b` |
| [思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/) | `79bdf56970b174267945acbe875c929a441a64c75eb4bb422a2b793d324a2b6c` |
| [图像理解](https://api-docs.deepseek.com/zh-cn/guides/vision/) | `2707ce25760ad050432912cbb7701ca24a50a479436d24aebe5f0e755d4902fb` |

## 阅读定位

- PDF §2.1–§2.4：图文结构、CED、CSA2、mHC、Engram、DSpark 和 FP4 main KV。
- PDF §3.2：运行时 / 持久 KV 与两类 SWA Bounded Replay。
- PDF §4：45T 预训练数据与配置；Table 1 基座评测。
- PDF §5：数据合成、异步 RL/OPD、effort；Table 2/3/4 与附录 B 为评测口径。
- PDF §6：稀疏选择与近似重建限制。
- 模型卡与报告的 NL2Repo、reasoning 采样参数差异保留在分析页。

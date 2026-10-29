# Milvus 单库迁移与检索复测

日期：2026-10-25。

## 交付内容

默认资料库后端改为 `milvus`，MongoDB 不再是普通运行依赖。
Milvus 同时保存解析原文、切片、向量、来源元数据、版本 head、上传哈希映射与持久入库作业。
原始 PDF/DOCX 二进制不保证归档。Memory 模块继续使用原有 SQLite，属于独立功能。

- `deepresearch/knowledge/milvus_store.py`：入库、发布、历史检索、补偿、重试与索引重建。
- `deepresearch/knowledge/milvus_records.py`：有校验的记录存取；大正文分页面保存，避免 VARCHAR 字节超限。
- `deepresearch/knowledge/retrieval.py`：与旧后端共享的结构切片、BM25、RRF、配额、重排与原文回填。
- `deepresearch/knowledge/migration.py`：显式旧库迁移，保留来源快照/切片身份与历史时间。
- CLI 增加 `knowledge migrate --source-config`；默认 Compose 移除 MongoDB。
- 旧 `mongo_milvus` 仅供兼容/迁移，需另装 `knowledge-legacy`。本次没有自动迁移或删除用户旧资料。

## 存储与回归验证

最终全量命令同时启用旧库兼容与迁移测试：

```bash
DEEPRESEARCH_RAG_INTEGRATION=1 \
RAG_TEST_MONGO_URI=mongodb://127.0.0.1:27029 \
RAG_MIGRATION_MONGO_URI=mongodb://127.0.0.1:27029 \
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

结果：**570 项通过，0 失败、0 错误、0 跳过**。

存储测试使用真实 Milvus Lite；迁移与兼容测试使用独立临时 MongoDB。
确定性哈希向量仅用于隔离工程行为，不充当语义检索质量证据。
覆盖判重、不同文档身份、作用域、历史版本、大型中文正文、原文校验、
发布前后故障、真实进程退出、同机写入互斥、重试、重排篡改防护、索引重建失败、
发布确认丢失、Worker 跨运行资料复用及旧库无损迁移。
CLI 实际完成了使用真实 GTE 的入库和检索，并返回 `degraded=false`、`vector_status=available`。

测试日志/XML 位于 `.runtime/milvus-single-migration/`。
Compose 仅完成配置结构检查；本机未实际启动 Docker Standalone 集群。

## 冻结开发集复测

26份文档、222个切片、124道已有开发题；不改标签、切片、问题、背景缓存或检索参数。
配置为 GTE-small-zh＋冻结切片背景说明＋BM25/RRF＋quota_dense＋BGE-reranker-v2-m3原文精排。
Embedding/reranker revision、缓存和语料哈希见 `results.json` 的 protocol。
重排实验超时30秒、CPU、Top5；不是当前默认E5/关闭重排配置的分数。

| 题型 | 题数 | Hit@5 | MRR@5 | 证据组覆盖@5 | 全部证据组命中@5 |
|---|---:|---:|---:|---:|---:|
| 真实项目文档 | 72 | 94.44% | 0.877 | 94.44% | 94.44% |
| 合成诊断 | 32 | 100.00% | 0.984 | 100.00% | 100.00% |
| 多来源 | 8 | 100.00% | 0.792 | 93.75% | 87.50% |

12道无答案题均返回了候选；这是候选非空率100%，**不能解释成拒答正确率或答案正确率**。
需要回答层另做拒答/忠实性评测。

**全部124题最终有序的文档ID、原文跨度与返回原文，均与历史双库 bge_original 结果一致。**
本次结论是存储迁移后保住了检索效果，不是额外提升了召回分数，也不是新的独立盲测成绩。
历史结果是冻结参考，没有把两种存储同时部署做配对延迟压测。
本次部分测试并行运行，CPU耗时只作运行记录，不能据此推断新架构吞吐或延迟优劣。

复现：

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m evaluation.knowledge_rag.verify_single_milvus
```

逐题结果位于本目录 `results.json`，含返回原文、跨度、指标、耗时、历史差异及代码哈希。
原始结果按仓库策略保留本地，不默认发布到 Git。

## 能力边界与使用说明

只承诺同机写入者使用统一 URI/namespace/锁目录时的进程互斥，不是跨机器分布式锁。
发布确认不明时保留 staged 数据并阻止后续写入，需显式 `knowledge reconcile`。
版本 head 是应用层可见性边界，不宣称跨集合或多行事务。
旧原文页和替换的成功索引保留，尚无自动容量回收策略。
默认 `enabled=false`，没有自动启用RAG、换默认模型或发起付费LLM请求。

当前部署、迁移及并发说明见 [单库使用说明](../../../docs/knowledge-rag.md)。
之前的双库PDF属于历史资料，不能作为本次单库实现的说明。

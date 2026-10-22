# RAG 文档入库链路工程评测

日期：2026-10-19。按用户提供的飞书第 6 题，检查判重、解析、切块、向量写入、版本发布、故障回滚、并发与恢复。被测实现为项目当前 `deepresearch/knowledge/store.py`，存储是本地 SQLite；没有安装或测试 Milvus、MongoDB，也不能将这些结果外推为跨数据库事务保证。

**46 项专项检查：32 项满足、14 项未达目标、0 项执行错误。现有全量回归：455 项通过，2 条依赖弃用警告。** 这是工程契约的逐项结果，不是检索准确率，也不应把 32/46 当成 RAG 综合质量分。14 个未满足项包含缺失能力及缺陷，其中 5 项异常向量测试属于同一个校验缺口。本次没有修改生产逻辑以掩盖失败。

## 对照飞书入库方案

| 飞书要求 | 本项目实测 | 结论 |
|---|---|---|
| 上传后先算 SHA-256，相同文件跳过解析和 embedding | 同 collection、URI、解析正文哈希命中后不再编码；重复 PDF 仍解析了 2 次；相同正文换 URI 仍重新编码 | 部分满足；缺少上传字节哈希前置判重和跨来源向量复用 |
| 解析、按语义边界切块 | Markdown 标题路径、正文覆盖、字符位置及两页文本 PDF 页码通过；超长独立标题越过长度限制；跨片表格未延续表头 | 结构切分可用，尚不是完整的语义/表格解析 |
| 批量向量化后写入索引 | 120 片段通过一次 provider 批量接口提交；实际 E5 非降级编码和向量查询通过 | 基本链路通过；不能由此宣称 Milvus 批量写入通过 |
| 所有 Chunk 就绪才发布新版本 | 批次数不足、空向量、第二视图编码异常可拦住发布；NaN、Inf、错误维数、零向量、非数值向量未拦住 | **就绪校验存在高优先级缺口** |
| 建 supersedes，先准备新版本，再归档旧版本 | 同 URI 新版本建立正确链；事务提交前连续 20 次读取均为旧版，提交后新版完整可见 | 单 SQLite 原子切换通过；改名后的同一制度不能自动关联 |
| 失败后旧版继续可用，清理半成品 | 在快照、部分 Chunk、归档阶段注入写入失败，均完整回滚；写入中子进程退出也可恢复，无孤立片段 | 本地事务范围通过；没有跨库补偿验证 |
| pending 作业、已写 ID、可重试及失败追踪 | 当前只有 snapshots/chunks，无持久化入库作业、pending/failed 状态或重试队列 | 未满足该治理能力；不影响已验证的本地原子性 |

当前真实顺序是：解析 → 正文哈希判重 → 切块和编码 → 开始 SQLite 写事务 → 再判重 → 写新快照/片段 → 归档旧版 → 提交。它不具有飞书里的独立 pending/ready 状态机，但读者看不到事务内半成品。不能把“原子提交通过”和“向量内容确实可用”混为同一个保证。

## 发现的问题及修复顺序

### 优先处理：发布前向量门禁与并发迁移

1. **发布前校验不完整（5 项未满足）。** 故障 provider 明确自报为健康语义模型（`degraded=false`），分别返回 NaN、Inf、维数 3 而非声明的 128、全零、字符串元素。五种情况都实际发布为 active 并归档旧版。应在事务之前统一校验两种视图的片段数量、维数、数值类型、有限值、非零范数；任何非法值使整个作业失败，旧版保持 active。当前“批次数一致且非空”的判断不足以表示 ready。
2. **旧库并发迁移存在竞态（1 项未满足）。** 两个实例在相同旧 schema 上同时初始化，实测一个成功，另一个报 `duplicate column name: index_context`。应在数据库迁移事务/锁内检查和修改 schema，以版本号管理升级，并在锁内复查。当前实例内锁不能保护独立进程。

以上缺陷不会因 SQLite 没有孤立记录而自动消失：事务只会回滚已触发的异常，不会识别静默接受的坏向量。

### 补齐入库治理与文档结构

- 判重成本：增加上传文件 `file_sha256`，在解析之前查重；另保留解析正文 `content_sha256` 校验原文。相同正文不同 URI 保留各自来源身份，但按内容哈希、模型版本、切块策略复用向量。跨实例并发导入需持久化作业抢占/租约；当前最终只插入一份，但 embedding 调用了两批。
- 文档身份与作业：引入显式 `document_id`、版本关系、作业阶段/错误记录和幂等重试。改名不能直接断言是新文档，也不能仅凭标题相同自动归档另一份文件。相同正文的元数据修正当前会被重复返回分支忽略，应提供单独更新路径。
- 切片边界：标题末尾路径也要执行长度约束；表格按行切片并延续表头、单位和章节。短段落可在 token 预算内合并，避免一段一片。
- 存储成本：未开启背景索引时，无需为每个片段存两份完全相同的向量 JSON；保留重建能力和模型信息即可。

这些是后续修复建议，本报告中的未通过项尚未修复。当前单库可以先完善事务、作业和恢复；只有引入 MongoDB/Milvus 等独立写入系统后，才需要另外设计并测试跨库补偿、Outbox/Saga 与对账。不能用本次 SQLite 回滚测试替代那些测试。

## 测试方法与边界

- 每项使用独立临时数据库和文件，不修改用户资料库。大部分故障测试使用明确标记的确定性哈希 fixture，隔离存储行为；不将其结果作为语义检索收益证据。
- 失败注入使用真实 SQLite trigger 抛错，在写入快照后、部分片段后和归档时分别中断；另启动真实子进程，通过 `os._exit(23)` 在事务内终止后重开库，检查旧版、孤立片段与 `PRAGMA integrity_check`。
- 并发测试用独立实例及同步屏障控制交错；读写可见性测试在中间写入阶段暂停，再执行 20 次读取。它们覆盖确定的危险交错，不是长期高并发压力测试。
- 真实模型为缓存的 `intfloat/multilingual-e5-small`，禁用哈希回退，实际得到 384 维向量。两页 PDF 为程序生成的文字层测试件；空白 PDF、无效文件与不支持的 DOCX 被拒绝并保持旧版。**拒绝通过不等于支持 DOCX 或扫描件 OCR。**
- 历史查询验证的是导入/抓取时点，并非制度业务生效日期。没有检测网络盘断电、磁盘满、硬件损坏、百万片段规模、长期并发或分布式数据库故障。
- 用例由当前实现和飞书要求设计，属于开发期工程验收，并非独立第三方审计。结果只对应本次工作区代码，未建立发布版本性能基准。

## 真实 E5 与成本观察

| 检查 | 实测 |
|---|---|
| 实际编码、幂等判重、向量查询 | 384 维，非降级；2 个片段；首次含加载约 4.947 秒，重复导入约 0.581 毫秒；向量查询返回候选 |
| 自然中文切片 token 长度 | 5 片段，最大 391 token，模型上限 512，本样本无截断；代码尚无通用 token 长度门禁 |
| 1,000 字符合成短段落 | 44 片段，1.786 秒（含该实例首次加载），重复约 0.769 毫秒 |
| 10,000 字符合成短段落 | 435 片段，0.422 秒，重复约 0.873 毫秒 |
| 50,000 字符合成短段落 | 2,174 片段，2.018 秒，重复约 1.135 毫秒 |

以上是单次本机观测，无 p95/p99 或跨机器结论。吞吐输入为大量重复的短段落，不代表真实 PDF 解析、长表格或真实资料库吞吐。三个文档合计 61,000 字符、2,653 片段，数据库文件 45,694,976 字节（约 43.6 MiB）；其中 2,653 行的 vector 与 raw_vector 完全相同。短段落不合并、重复段落不共用存储、向量使用 JSON 且存两份，共同造成该样本的空间开销。

## 复现与产物

在项目根目录执行：

```bash
# 完整专项；模型需已缓存，默认收集所有结果而不因已知缺口中止
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.audit_ingestion

# CI 门禁：跳过 3 项真实模型/性能检查，发现 unmet 或 error 返回 1
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.audit_ingestion \
  --no-real-model --fail-on-unmet --output .runtime/ingestion-audit-ci.json

# 原有全量回归
.venv/bin/python -m pytest tests -q \
  --junitxml=evaluation/knowledge_rag/ingestion_audit/regression.xml
```

完整专项实测为 32 pass / 14 unmet / 0 error。无模型门禁实测为 29 pass / 14 unmet / 0 error，按预期返回退出码 1。现有回归 455 passed、2 warnings、约 77 秒；它与新故障检查覆盖不同，因此全部通过不能推翻新发现的缺陷。

- [逐项机器结果](results.json)：每项的状态、异常原因、观测值与耗时。
- [专项执行器](../audit_ingestion.py)：故障、并发、崩溃与真实模型用例。
- [全量回归 JUnit](regression.xml)。
- [此前 124 题检索评测](../v3/REPORT.md)：测召回/排序，与本次工程入库评测分开保留。

## 全部检查明细

下表由本次 `results.json` 生成；英文名称可直接定位执行器中的对应检查。

| # | 检查 | 结果 | 观测/未满足原因 |
|---:|---|---|---|
| 1 | `same_uri_same_hash_skips_embedding` | 满足 | {"initial_embedding_batches":1,"duplicate_extra_batches":0,"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 2 | `contextual_duplicate_skips_both_views` | 满足 | {"initial_embedding_batches":2,"duplicate_extra_batches":0,"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 3 | `same_hash_different_uri_reuses_embedding` | 未满足 | same content at another URI is re-embedded; no collection-wide content reuse |
| 4 | `collection_isolation` | 满足 | {"snapshots":2,"active":2,"chunks":2,"orphan_chunks":0,"integrity":"ok"} |
| 5 | `raw_parsed_text_sha_distinguishes_changes` | 满足 | {"snapshots":2,"active":1,"chunks":2,"orphan_chunks":0,"integrity":"ok"} |
| 6 | `duplicate_metadata_correction` | 未满足 | duplicate path discards title/metadata corrections |
| 7 | `utf8_markdown_and_file_uri` | 满足 | {"chars":33,"hash_verified":true} |
| 8 | `two_page_pdf_text_and_page_locations` | 满足 | {"page_count":2,"beta_page":2,"snapshots":1,"active":1,"chunks":2,"orphan_chunks":0,"integrity":"ok"} |
| 9 | `file_sha_before_reparsing` | 未满足 | duplicate file is parsed again; SHA dedup happens after parsing |
| 10 | `reject_invalid_or_unsupported_pdf` | 满足 | {"exception":"PdfStreamError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 11 | `reject_invalid_or_unsupported_blankpdf` | 满足 | {"exception":"ValueError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 12 | `reject_invalid_or_unsupported_txt` | 满足 | {"exception":"ValueError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 13 | `reject_invalid_or_unsupported_docx` | 满足 | {"exception":"ValueError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 14 | `parsed_snapshot_survives_source_deletion` | 满足 | {"text_persists_after_file_removed":true} |
| 15 | `chunk_offsets_hierarchy_and_content_coverage` | 满足 | {"chunks":8,"nonwhitespace_missing":0,"max_chars":598} |
| 16 | `heading_only_length_bound` | 未满足 | heading-only input bypasses chunk size bound |
| 17 | `long_table_header_preservation` | 未满足 | long table is split without carrying its header to later chunks |
| 18 | `document_chunks_encoded_as_batch` | 满足 | {"chunks":120,"embedding_batches":1,"items":120} |
| 19 | `reject_short_batch_embeddings` | 满足 | {"exception":"ValueError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 20 | `reject_empty_embeddings` | 满足 | {"exception":"ValueError","snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 21 | `reject_nan_embeddings` | 未满足 | nan vectors were published active |
| 22 | `reject_dimension_embeddings` | 未满足 | dimension vectors were published active |
| 23 | `reject_infinity_embeddings` | 未满足 | infinity vectors were published active |
| 24 | `reject_zero_embeddings` | 未满足 | zero vectors were published active |
| 25 | `reject_nonnumeric_embeddings` | 未满足 | nonnumeric vectors were published active |
| 26 | `second_embedding_view_failure_keeps_old` | 满足 | {"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 27 | `publish_complete_new_version_then_archive_old` | 满足 | {"supersedes_correct":true,"snapshots":2,"active":1,"chunks":2,"orphan_chunks":0,"integrity":"ok"} |
| 28 | `rollback_failure_at_snapshot` | 满足 | {"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 29 | `rollback_failure_at_partial_chunks` | 满足 | {"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 30 | `rollback_failure_at_archive` | 满足 | {"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 31 | `capture_time_history_and_archived_reimport` | 满足 | {"historical_hits":1,"old_reimport_status":"archived"} |
| 32 | `renamed_logical_document_version_detection` | 未满足 | version identity is URI-based; renamed document becomes another active document |
| 33 | `durable_ingestion_job_and_retry_record` | 未满足 | no durable ingestion job/pending/failed audit or retry queue |
| 34 | `readers_see_old_version_until_commit` | 满足 | {"precommit_reads":20,"old_visible_during_all_reads":true,"new_version_ready":true} |
| 35 | `same_instance_concurrent_dedup` | 满足 | {"insertions":1,"embedding_batches":1,"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 36 | `separate_instances_avoid_duplicate_embedding` | 未满足 | separate workers both embed before transactional dedup recheck |
| 37 | `concurrent_different_versions_form_one_chain` | 满足 | {"chain_length":3,"snapshots":3,"active":1,"chunks":3,"orphan_chunks":0,"integrity":"ok"} |
| 38 | `process_exit_mid_write_recovers_without_orphans` | 满足 | {"child_exit":23,"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 39 | `reindex_failure_rolls_back_entire_projection` | 满足 | {"unchanged_chunk_rows":101,"snapshots":2,"active":2,"chunks":101,"orphan_chunks":0,"integrity":"ok"} |
| 40 | `restart_persistence_and_retrieval` | 满足 | {"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 41 | `concurrent_legacy_schema_initialization` | 未满足 | concurrent schema migration is check-then-alter without synchronization |
| 42 | `immutable_text_checksum_verification` | 满足 | {"corruption_detected":true} |
| 43 | `rebuild_lost_index_from_canonical_text` | 满足 | {"reindexed_snapshots":1,"model":"local-hash-v2-128","degraded":true,"snapshots":1,"active":1,"chunks":1,"orphan_chunks":0,"integrity":"ok"} |
| 44 | `real_e5_ingestion_duplicate_and_vector_query` | 满足 | {"model":"sentence-transformers:intfloat/multilingual-e5-small","degraded":false,"dimensions":384,"chunks":2,"initial_ms_including_load":4947.486166987801,"duplicate_ms":0.5811250011902303,"vector_search_hits":2} |
| 45 | `real_e5_chunk_token_budget` | 满足 | {"model":"sentence-transformers:intfloat/multilingual-e5-small","model_max_tokens":512,"chunks":5,"oversized_chunks":0,"max_actual_tokens":391} |
| 46 | `real_e5_ingestion_size_and_duplicate_latency` | 满足 | {"model":"sentence-transformers:intfloat/multilingual-e5-small","sizes":[{"chars":1000,"chunks":44,"ingest_seconds":1.78571529200417,"chars_per_second":559.9996844276701,"duplicate_ms":0.769499980378896},{"chars":10000,"chunks":435,"ingest_seconds":0.42244166700402275,"chars_per_second":23671.907345031792,"duplicate_ms":0.873042008606717},{"chars":50000,"chunks":2174,"ingest_seconds":2.0180702499928884,"chars_per_second":24776.144438072064,"duplicate_ms":1.1349580017849803}],"database_bytes":45694976,"total_chunk_rows":2653,"identical_vector_and_raw_vector_rows":2653,"note":"Short synthetic repeated paragraphs; not production PDF throughput. First size includes lazy model load."} |

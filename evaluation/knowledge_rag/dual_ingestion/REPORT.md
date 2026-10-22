# 飞书入库方案：MongoDB＋Milvus 工程验收

日期：2026-10-19。被测为新增 `MongoMilvusStore` 双库后端。原方案和适配边界见[项目说明](../../../docs/knowledge-rag.md)。此前的 [SQLite 46 项审计](../ingestion_audit/REPORT.md)保留为历史记录；新增后端不等于自动修复兼容后端。

**40 项双库专项已通过；最终开启双库专项执行全项目，495 项通过，0 失败、0 跳过，3 条依赖弃用警告，约 60 秒。** 使用真实 MongoDB 8.0.16 进程和 Milvus Lite 2.5.1（通过 TCP 让不同客户端/子进程连接同一引擎）；没有用 mongomock、Python dict 或 SQLite 来替代这两个系统。Milvus Standalone 的 Docker 部署尚未运行，Lite 不等于完整生产部署验收。

## 按飞书第 6—8 题验收

| 约束 | 实际操作与断言 | 结果 |
|---|---|---|
| 上传先算 SHA-256，重复不解析/不编码 | 相同文件第二次导入时，将解析器替换为一调用即失败；导入仍返回原快照，embedding 次数不变 | 通过 |
| 同文件不同路径不重复消耗 | 复制相同文件到另一文件名；返回同快照并保存来源别名，不重复编码 | 通过；显式不同 document_id 不强行合并 |
| 同一制度改名后可维护版本链 | 不同文件名显式指定同一 document_id；新版 supersedes 指向旧版，仅一个当前 head | 通过；不冒称自动语义版本识别 |
| 所有片段就绪再切换 | MongoDB 数量检查、Milvus flush＋强一致实际回读；ready 阶段暂停，连续读取仍是完整旧版 | 通过 |
| 向量异常不可就绪 | 分别注入 NaN、Inf、全零、错维、字符串、布尔、空向量、批次数不足 | 8 项均拒绝发布，旧版保持 active |
| 快照或 MongoDB 片段写入后失败 | 在真实写入之后的注入点抛出 I/O 异常，检查 MongoDB 残留、作业、旧版状态 | 通过 |
| Milvus 部分写入后失败 | 150 段输入分批写入，首批成功后抛错；按预先记录 ID 删除片段/向量，保留 pending 正文 | 通过 |
| 清理本身失败 | 在向量清理完成后再抛错；记录 cleanup_failed；恢复后执行 reconcile | 通过，可重入且不归档旧版 |
| 发布确认丢失 | 新 head 已原子提交后抛 TimeoutError；新版保留可读，不反向清理；对账补齐审计和归档 | 通过 |
| 进程异常退出 | 真实子进程分别在首批向量写入后、发布 head 后 os._exit(23)；重连原数据库对账 | 2 项通过；实际退出后直接推进测试租约时钟，不等待 120 秒 |
| 并发写入受约束 | 新版 ready 时暂停；独立实例第二写入者被文档租约拒绝，完成后重试命中判重 | 通过；忙时明确重试，未承诺所有请求都等待成功 |
| 可恢复原文与索引 | 删除本测试的真实 E5 Milvus 候选，再以 MongoDB 原文重建；版本、正文和哈希不变 | 通过 |
| 索引重建中断 | 独立代次已写部分数据后抛错，旧索引指针不变，新代次删除并记录失败 | 通过；每版本切换，不是全库事务 |
| 第三道引用校验 | 已抽取来源携带资料库快照引用，发布新版后最终核验旧来源；数据库不可用也不能确认有效 | 通过；最终门禁代码已接入图流程 |

MongoDB `heads` 的一次原子操作同时记录新指针与提交记录，保证提交确认不明时有真源可查；跨 MongoDB 文档与 Milvus 的步骤仍可能分别成功。作业先记写入意图，补偿和对账按这个记录恢复，不能把它称作严格分布式事务。

## 其他检查

- 真实文字 PDF 的页码与正文位置；删除来源文件后仍能读取 MongoDB 正文。
- DOCX 的标题、正文和表格原生解析；无效 PDF、空文本、不支持格式均记录解析失败并保留旧版。
- 长标题长度约束、跨片表头上下文与原文跨度；不把补入的表头伪装成连续原文。
- collection 隔离、原文哈希篡改检测、元数据修正不重复编码。
- 按发布时点查询历史版，旧内容再次导入不自动复活 archived 版本。
- 同一文档并发租约、运行中作业不会被 reconcile 抢走。
- 失败后重试保留上传字节身份；源文件被修改时解析失败作业拒绝盲重试。
- 候选副本 active 状态滞后时，MongoDB head 仍排除旧版；哈希降级不能发布为健康语义索引。
- 重排不可用退回 RRF；哈希 fixture 查询只作关键词降级，不伪装语义召回。

## 模型与故障注入边界

存储故障用例显式设置 `allow_degraded=True`，使用确定性哈希向量隔离存储行为；它是测试开关，默认生产路径不允许哈希索引发布。非法向量用例模拟自报健康的 provider，验证不能靠 degraded 标志代替数值校验。

另一个专项实际加载缓存的 E5、得到 384 维语义向量、执行 Milvus 纯向量查询和丢索引重建。三种真实召回模型的效果在[模型比较报告](../model_comparison/REPORT.md)独立评估，不能拿存储测试通过率代表检索质量。

退出测试使用真实子进程退出，故障注入为写入边界明确抛错。没有验证机房掉电、网络长时间分区、磁盘满、云托管服务、Mongo 副本集选举、Milvus Standalone/Distributed 或百万规模数据。原生解析的成功不代表已经验证 MinerU 的 OCR/版面/复杂表格效果。

## 部署、重放与恢复

```bash
uv pip install --python .venv/bin/python -e '.[knowledge-local]'
# 启动本地 MongoDB 后，运行真实双库专项
DEEPRESEARCH_RAG_INTEGRATION=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests/test_knowledge_dual_integration.py -q
# 可指定已有 Milvus 服务，避免自动启动私有 Lite 服务
RAG_TEST_MONGO_URI=mongodb://127.0.0.1:27028 \
RAG_TEST_MILVUS_URI=http://127.0.0.1:19530 \
DEEPRESEARCH_RAG_INTEGRATION=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests/test_knowledge_dual_integration.py -q
# 全项目，包括双库专项
DEEPRESEARCH_RAG_INTEGRATION=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests -q
```

测试只创建和清理自己生成的 `rag_test_<uuid>` 数据库/快照 ID；不会清空用户库。普通 `pytest tests` 未开启环境开关时，这 40 项显式 skip，不能将 skip 说成双库测试通过。

CLI 的 `knowledge jobs / reconcile / retry JOB_ID` 可审查作业、恢复中断和重新执行；尚无自动常驻调度器。Compose 配置见 [knowledge.compose.yaml](../../../deploy/knowledge.compose.yaml)，本机没有 Docker，因此仅做了 YAML 解析，未验收容器启动。

本机另提供 `.runtime/knowledge-local.yaml`：MongoDB 监听 127.0.0.1:27028，Milvus 使用独立本地文件。已实际通过 CLI 导入本项目 RAG 说明（32 片段），用相同文件再导入判重，并实际检索出 5 个候选；真实 Worker 前置上下文也返回 2 个片段和 1 份本地文档，携带资料库快照引用。该步骤没有调用生成模型，不是答案质量评测。它是开发用资料，不能当作飞书原文或外部权威来源。MongoDB 原生进程为本次启动，二进制与数据位于 `.runtime/rag-services/`。

## 机器结果与测试清单

- [双库 40 项 JUnit](../dual_ingestion-regression.xml)。
- [最终全项目 JUnit（开启双库测试）](../dual-full-regression.xml)。
- [追加的真实 E5 向量路径旧版过滤验证](../dual-vector-version-regression.xml)：将旧版 Milvus 状态故意改回 active，纯向量查询仍只返回 MongoDB 当前发布版；1 项通过，是既有用例加强复测，不另计独立用例数。
- [实际测试源代码](../../../tests/test_knowledge_dual_integration.py)。
- [双库生产实现](../../../deepresearch/knowledge/dual_store.py)。
- [本次代码 SHA-256 清单](code-manifest.json)。

下面清单由本次双库 JUnit 生成。

| # | 检查 | 结果 |
|---:|---|---|
| 1 | `test_upload_sha_precedes_parse_and_embedding` | 通过 |
| 2 | `test_identical_upload_alias_avoids_reencoding` | 通过 |
| 3 | `test_renamed_versions_use_explicit_logical_id` | 通过 |
| 4 | `test_failure_before_publication_compensates_both_planes[snapshot_written]` | 通过 |
| 5 | `test_failure_before_publication_compensates_both_planes[mongo_chunks_written]` | 通过 |
| 6 | `test_failure_before_publication_compensates_both_planes[vector_batch_written]` | 通过 |
| 7 | `test_failure_before_publication_compensates_both_planes[ready]` | 通过 |
| 8 | `test_invalid_embedding_never_archives_old[nan]` | 通过 |
| 9 | `test_invalid_embedding_never_archives_old[inf]` | 通过 |
| 10 | `test_invalid_embedding_never_archives_old[zero]` | 通过 |
| 11 | `test_invalid_embedding_never_archives_old[dim]` | 通过 |
| 12 | `test_invalid_embedding_never_archives_old[string]` | 通过 |
| 13 | `test_invalid_embedding_never_archives_old[bool]` | 通过 |
| 14 | `test_invalid_embedding_never_archives_old[empty]` | 通过 |
| 15 | `test_invalid_embedding_never_archives_old[short]` | 通过 |
| 16 | `test_published_head_survives_ambiguous_ack_and_reconcile` | 通过 |
| 17 | `test_cleanup_failure_is_durable_and_reconciled` | 通过 |
| 18 | `test_failed_upload_retry_preserves_byte_identity` | 通过 |
| 19 | `test_live_reader_and_second_writer_are_fenced` | 通过 |
| 20 | `test_real_process_exit_and_durable_recovery[vector_batch_written]` | 通过 |
| 21 | `test_real_process_exit_and_durable_recovery[head_published]` | 通过 |
| 22 | `test_collection_boundary_and_immutable_source` | 通过 |
| 23 | `test_capture_time_history_and_archived_reimport` | 通过 |
| 24 | `test_metadata_correction_avoids_encoding` | 通过 |
| 25 | `test_parse_failure_has_job_and_keeps_old[.pdf-not PDF]` | 通过 |
| 26 | `test_parse_failure_has_job_and_keeps_old[.txt-   ]` | 通过 |
| 27 | `test_parse_failure_has_job_and_keeps_old[.exe-bad]` | 通过 |
| 28 | `test_pdf_pages_and_source_persistence` | 通过 |
| 29 | `test_docx_headings_and_table` | 通过 |
| 30 | `test_large_heading_and_table_source_integrity` | 通过 |
| 31 | `test_final_authority_gate_blocks_archived_source` | 通过 |
| 32 | `test_real_e5_milvus_vector_search_and_rebuild` | 通过 |
| 33 | `test_failed_rebuild_keeps_previous_projection` | 通过 |
| 34 | `test_stale_vector_replica_does_not_return_archived_facts` | 通过 |
| 35 | `test_vector_and_reranker_failure_keep_lexical_results` | 通过 |
| 36 | `test_unavailable_embedding_never_publishes_hash_fallback` | 通过 |
| 37 | `test_precontext_uses_mongo_original_and_preserves_table_header` | 通过 |
| 38 | `test_compensation_refuses_published_history` | 通过 |
| 39 | `test_reconcile_respects_live_writer_lease` | 通过 |
| 40 | `test_retry_rejects_changed_original_upload` | 通过 |

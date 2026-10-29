# 研究原始资料跨运行复用：工程验收

日期：2026-10-24。范围为自动保存与后续检索接入，不是新的检索质量或报告质量评测。

## 行为

- 同时开启 `knowledge.enabled` 与 `knowledge.persist_fetched_sources` 后，将成功抓取的 HTTP(S) 正文发布到用户、项目隔离的持久集合。
- 后续 Worker 同时检索手动导入资料库与所属自动集合，共用片段数量、来源配额和上下文字符预算。
- 保留来源、抓取时间、发布日期、内容哈希和版本；原文跨度始终针对对应快照。重复正文不重复编码，A→B→A 为新的捕获版本，较旧抓取晚完成不覆盖新版本。
- 缓存网页仅作为带日期的研究线索，不直接成为本轮已验证证据；正常抓取、抽取与账本准入仍执行。
- 自动集合故障不丢弃可用的手动导入候选；知识库不可用时正常研究继续并记录缺口。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 新增跨运行资料专项 | 17 项，包含真实 SQLite、模拟模型与确定性向量 |
| 专项与原 RAG、查询改写回归 | 47 passed，14.27 秒 |
| 最终全项目回归，开启真实双库检查 | **546 passed，0 failed，0 errors，0 skipped，64.12 秒** |
| `git diff --check` | 通过 |

最终全项目包含真实原生 MongoDB 与 Milvus Lite。测试使用新建的临时 MongoDB 数据目录和隔离测试数据库；没有连接或修改用户资料库。新增真实双库用例采用确定性向量隔离存储行为，既有语义向量测试保持执行。产生 3 条依赖弃用警告，不影响通过结果。

`tests/test_knowledge_research_sources.py` 验证：重建服务／存储后下一次研究策略收到旧片段、用户与项目隔离、缺失作用域、关闭开关、只允许成功原文抓取、重复抓取幂等、版本回返、乱序抓取、日期保留、总预算、来源限制及初始化／写入／检索故障回退。真实双库新增用例位于 `tests/test_knowledge_dual_integration.py::test_worker_fetched_sources_cross_run_reopen_scope_and_versions`。

完整机器可读结果：[full-regression.xml](full-regression.xml)。首次全量发现新增用例关闭共享 Milvus 连接导致测试清理失败；修正连接所有权后重新运行全部测试，上表只列最终成功结果。

## 复现

```bash
# MongoDB 地址必须指向可用于隔离测试的本地服务；Milvus Lite 由测试夹具启动。
DEEPRESEARCH_RAG_INTEGRATION=1 \
RAG_TEST_MONGO_URI=mongodb://127.0.0.1:YOUR_TEST_PORT \
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests -q \
  --junitxml=evaluation/knowledge_rag/cross_run_sources/full-regression.xml
```

## 使用与边界

完整运行配置中设置：

```yaml
knowledge:
  enabled: true
  persist_fetched_sources: true
```

这是对现有配置中这两个字段的修改，其他 MongoDB／Milvus／模型字段继续保留。全局默认开关未改变。实现与运行说明见 [knowledge-rag.md](../../../docs/knowledge-rag.md)。

本功能从开启后的新抓取积累资料；未自动回填旧运行目录快照，不将模型生成报告或搜索摘要当作外部原文入库。尚无资料库容量自动清理，也没有证明减少联网次数、降低整体 Token 或提升 DRBII。此前 124 题检索开发集分数未在本次重测。

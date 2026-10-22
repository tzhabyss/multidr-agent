# 背景索引后的精排优化对照

日期：2026-10-20。固定26份文档、222个切片、124题、已冻结的背景说明、GTE-small-zh、关键词5/向量15。真实MongoDB＋Milvus Lite，四组每次调用的有序候选片段完全相同。原文基线逐题复现上次背景说明实验。

第一轮预定三组为现有mMiniLM、mMiniLM增加切片前后各120字符的原文、BGE-reranker-v2-m3。看到72道真实题中mMiniLM邻接命中率提高但MRR下降后，追加BGE＋相同邻接原文的第四组，仍完整运行124题。追加选择与当时结果已冻结在followup；这是自适应开发实验，不是四组都预注册的盲测。

没有把生成的说明交给精排，没有增加新的LLM调用。普通候选池20，Worker每子问题40，最终Top5与6000字符/每来源2段预算不变。

## 72道真实文档题

| 方案 | Hit@5 | MRR@5 | 新增/丢失 | CPU查询P95 ms | 精排超默认5秒 /133次 |
|---|---:|---:|---:|---:|---:|
| mMiniLM原文 | 67/72（93.1%） | 0.832 | 0/0 | 526.6 | 0/133 |
| mMiniLM＋邻接原文 | 69/72（95.8%） | 0.813 | 2/0 | 791.7 | 0/133 |
| BGE v2-m3原文 | 68/72（94.4%） | 0.877 | 1/0 | 3724.8 | 4/133 |
| BGE v2-m3＋邻接原文 | 70/72（97.2%） | 0.817 | 3/0 | 5511.0 | 46/133 |

综合判断：BGE原文精排相对现有方案同时提高单问Hit@5与MRR，多来源完整覆盖及证据组覆盖保持不变，但CPU耗时显著增加。BGE邻接组单问覆盖最高，却降低MRR和多来源证据组覆盖；mMiniLM邻接组也有排序及多来源回退。因此没有将邻接精排设为全局默认。
若以检索质量优先，可选择BGE原文精排并保持reranker_neighbor_chars=0；需要低延迟时保留mMiniLM。两者都仍需独立新题验证，不能把本轮开发集结果当成泛化保证。

这是30秒实验精排预算下的质量结果。5秒超时后的RRF回退与后续忙碌回退没有作为另一条质量路线重跑；不能直接声称与生产默认5秒完全等价。计时包含诊断token化开销，固定顺序CPU单次观测，不是生产SLA。

## 原先剩余5题

| ID / 问题 | mMiniLM | mMiniLM邻接 | BGE v2-m3 | BGE邻接 |
|---|---:|---:|---:|---:|
| repository-001 系统证据如何逐级追溯到网页快照？ | 10 | 17 | 6 | 4 |
| repository-004 如何更换整个项目的运行配置文件？ | 8 | 4 | 1 | 1 |
| repository-013 怎样检查本地运行配置是否正常？ | 20 | 15 | 13 | 14 |
| repository-014 CLI 要恢复相同 run_id 的研究应该调用什么命令？ | 16 | 3 | 10 | 2 |
| repository-042 学术检索服务出错或者没有结果会回退到哪里？ | 未入池 | 未入池 | 未入池 | 未入池 |

表内数字为已标证据的精排位置。repository-042的证据未进入固定候选池，更换精排不能找回这段证据。未对单题修改候选或说明，未按题拼接最优路线。

## 其他题型

| 方案 | 合成命中 /32 | 多来源完整覆盖 /8 | 多来源证据组覆盖 | 无答案返回候选 /12 |
|---|---:|---:|---:|---:|
| mMiniLM原文 | 32/32 | 7/8 | 93.8% | 12/12 |
| mMiniLM＋邻接原文 | 32/32 | 6/8 | 89.6% | 12/12 |
| BGE v2-m3原文 | 32/32 | 7/8 | 93.8% | 12/12 |
| BGE v2-m3＋邻接原文 | 32/32 | 7/8 | 87.5% | 12/12 |

mMiniLM邻接组虽然单问Hit@5提高，但MRR下降，且多来源完整覆盖由7/8降至6/8。multi-004中邻接内容包含研发投入，导致不含该事实的目标切片被排高；最终只返回目标切片，造成证据缺失。因此不只按单问命中率选择推荐方案。

首位命中及排名变化：`{"mini_original": {"top1_hits": 56, "mrr_improved_cases": 0, "mrr_worsened_cases": 0}, "mini_neighbors": {"top1_hits": 52, "mrr_improved_cases": 6, "mrr_worsened_cases": 8}, "bge_original": {"top1_hits": 59, "mrr_improved_cases": 8, "mrr_worsened_cases": 3}, "bge_neighbors": {"top1_hits": 50, "mrr_improved_cases": 10, "mrr_worsened_cases": 11}}`。

## 证据与窗口

邻接段从同一个Mongo快照读取，验证目标切片字节一致，严格限于前后各120字符。精排输入先保留目标原文，再追加带标识的邻接段；返回结果重新从原候选取回，不扩大证据位置，也不把邻接内容作为已返回的引用。配置关闭时保持原有输入。查询改写后的精排也复用同一准备路径。

| 方案 | 邻接尾部超512的输入对 | 目标原文超512的输入对 | 最大精排毫秒 |
|---|---:|---:|---:|
| mMiniLM原文 | 0 | 0 | 730.2 |
| mMiniLM＋邻接原文 | 0 | 0 | 1379.5 |
| BGE v2-m3原文 | 0 | 0 | 5675.2 |
| BGE v2-m3＋邻接原文 | 0 | 0 | 8757.0 |

目标原文未截断仅适用于本次冻结语料；没有实现任意新文档的通用token自适应切块。邻接尾部若超长，仍由模型512窗口截断。精排看到邻接信息，不表示最终Top5返回了该邻接证据；评分始终使用实际返回的原文跨度。

## 新增与退步

- mMiniLM＋邻接原文新增：repository-004, repository-014；丢失：无。
- BGE v2-m3原文新增：repository-004；丢失：无。
- BGE v2-m3＋邻接原文新增：repository-001, repository-004, repository-014；丢失：无。

开发集重复实验，题目与相关性标签未扩充，尚无独立盲测。Hit@5检查已标原文跨度，不是最终回答准确率；未标全的替代证据可能存在。无答案候选非空不代表拒答能力。没有Worker端到端生成评测，不直接声称最终研究报告质量提高。

## 配置与复现

双库配置新增 `knowledge.reranker_neighbor_chars: 120` 可试用相邻原文精排，默认0；不需重建索引，也不会自动启用RAG。SQLite兼容后端显式拒绝该非零配置。BGE路线使用 `reranker_model: BAAI/bge-reranker-v2-m3`，本实验30秒预算；实际部署应同时核对 `reranker_timeout_seconds` 和Worker总检索预算。所有组的全局默认保持原样。

本轮基于已启用背景索引、GTE和5/15策略的实验配置；单独修改精排模型不自动启用这些前提。自有语料需单独生成背景说明并reindex，评测缓存仅覆盖冻结的26份文档。普通题测Top5/候选20，多来源测Worker build_context的Top5/子问题候选40；实际默认Worker为top_k=3，未据此宣称其同样达到表内命中率。并行Worker的精排忙碌回退未做压测。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_context_rerank
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_context_rerank --routes bge_neighbors --output evaluation/knowledge_rag/context_rerank/followup
.venv/bin/python -m evaluation.knowledge_rag.merge_context_rerank
.venv/bin/python -m evaluation.knowledge_rag.render_context_rerank_report
```

[逐题数据及排序输入哈希](results.json) · [全部题目命中](CASES.md) · [协议](protocol.json) · [校验](verification.json)

测试：`{"regression.xml": {"tests": 62, "failures": 0, "errors": 0, "skipped": 0}, "full-regression.xml": {"tests": 528, "failures": 0, "errors": 0, "skipped": 0}}`。

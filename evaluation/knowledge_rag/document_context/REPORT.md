# 文档侧背景说明对照实验

日期：2026-10-20。26份冻结文档、222个相同切片、124道开发题，实际MongoDB＋Milvus Lite。固定GTE-small-zh＋mMiniLM、关键词5/向量15、BM25 Top20/向量Top40、普通精排池20、Worker子问题池40，最终Top5。查询改写关闭。

三组仅改变建索引的文本；说明同时进入BM25与向量编码，精排仍使用原问题、标题、章节和原文。说明不进入Worker的证据片段，不占其6000字符上下文预算，不作为引用。

## 72道真实文档题

| 索引 | Hit@5 | MRR@5 | 初始候选含证据 | 精排池含证据 | 新增/丢失 | CPU P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| 原文基线 | 65/72（90.3%） | 0.817 | 68/72 | 67/72 | 0/0 | 503.0 |
| 标题＋章节 | 63/72（87.5%） | 0.787 | 67/72 | 65/72 | 1/3 | 543.5 |
| 切片背景说明 | 67/72（93.1%） | 0.832 | 72/72 | 71/72 | 3/1 | 496.5 |

## 事先定位的4道漏召回题

| 题目 | 原文基线 | 标题章节 | 背景说明 |
|---|---|---|---|
| repository-013 怎样检查本地运行配置是否正常？ | 初召回0／入池0／Top5命中0 | 初召回0／入池0／Top5命中0 | 初召回1／入池1／Top5命中0 |
| repository-014 CLI 要恢复相同 run_id 的研究应该调用什么命令？ | 初召回0／入池0／Top5命中0 | 初召回0／入池0／Top5命中0 | 初召回1／入池1／Top5命中0 |
| repository-037 记忆中的向量采用什么存储类型？ | 初召回0／入池0／Top5命中0 | 初召回1／入池0／Top5命中0 | 初召回1／入池1／Top5命中1 |
| repository-038 记忆编码默认用哪个模型，是多少维？ | 初召回0／入池0／Top5命中0 | 初召回1／入池1／Top5命中1 | 初召回1／入池1／Top5命中1 |

两道中文查询对应英文记忆文档，两道涉及CLI命令。README只列出doctor命令，没有明确解释其作用；生成说明中的“自检/检查环境”属于模型对命令与上下文的解释，尚未经代码核实，不能作为原文事实。恢复命令和默认模型/向量类型则在原文有明确依据。

## 其他题型

| 索引 | 合成命中（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案仍返回候选（12） |
|---|---:|---:|---:|---:|
| 原文基线 | 32/32 | 6/8 | 81.2% | 12/12 |
| 标题＋章节 | 32/32 | 7/8 | 93.8% | 12/12 |
| 切片背景说明 | 32/32 | 7/8 | 93.8% | 12/12 |

## 生成成本与索引窗口

说明模型：`deepseek-v4-flash`，temperature=0。222条说明全部冻结后才运行评分；生成缓存共259次请求，已知输入833,468 token、输出16,428 token。首轮37条未通过长度/格式校验，使用更短的提示补生成；原始记录在generation-v1，失败尝试用量计入总数。另有1次长度诊断请求（输入5,972、输出85 token，未用于索引），记录在generation-diagnostic.json。没有查看问题或标签后修改说明。未换算费用，未验证供应商缓存计费。

| 索引 | 建索引秒数（不含说明生成） | 最大编码token | 超512数量 |
|---|---:|---:|---:|
| 原文基线 | 10.29 | 334 | 0 |
| 标题＋章节 | 11.91 | 352 | 0 |
| 切片背景说明 | 19.17 | 446 | 0 |

说明生成发生在离线建库阶段；在线检索没有新增LLM调用。每条说明绑定文档SHA-256、原文跨度及切片哈希；未知文档回退普通索引，已知文档切片不匹配拒绝使用。新索引完整回读后，说明与向量指针一起切换；失败保留旧索引。

## 得失与边界

- 标题＋章节新增：repository-038；丢失：repository-022, repository-042, repository-062。
- 切片背景说明新增：repository-037, repository-038, repository-064；丢失：repository-042。

背景说明将72道真实题的已标证据全部带入初始候选并集，其中71题进入精排池。剩余5道Top5未命中题中，4题已入池但精排未进前五，1题被候选选择截掉。
CLI两题的已标命令段分别被mMiniLM排到第20、16名；同义说明没有进入精排文本，因而只能改善初召回。后续可单独测试带明确标识的检索说明参与相关性打分、邻接原文回填或更强精排，不应把生成说明冒充原文。
退步题repository-042：原文明确写出学术检索出错或无结果回退OpenAlex；生成说明主要概括服务器和工具清单，未突出回退条件。证据片段向量排名从第5降到第28，未进入固定精排池。未据此回改该片段的说明或混用逐题最佳结果。

这是同一开发集的对照，不是独立盲测；Hit@5只检查已标原文跨度，替代证据可能未标全。没有评测最终回答、拒答、引用正确率或Worker端到端收益。说明可能包含模型推断；目前只辅助找原文，不能把检索命中提升等同于事实质量提升。CPU单次热运行耗时不代表生产SLA。全局开关不自动启用。

方法参考：[Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)。本实验使用现有DeepSeek模型生成中文说明，使用更短字符上限；不是原文全部模型、语料和指标的复现。

## 复现与启用

```bash
.venv/bin/python -m evaluation.knowledge_rag.generate_document_context
# 仅当有超长/失败说明时补生成，并记录原尝试成本
.venv/bin/python -m evaluation.knowledge_rag.generate_document_context --retry-invalid
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_document_context
.venv/bin/python -m evaluation.knowledge_rag.render_document_context_report
```

完整双库配置中设置 `knowledge.document_context_path: evaluation/knowledge_rag/document_context/contexts.json` 后执行 `knowledge reindex`。已有原文、版本和引用位置保持不变；修改说明文件不会直接改动已发布的索引，仍需重新建索引。自有语料可用 `generate_document_context --corpus /path/corpus.json --output /path/context-cache`；输入数组元素需有id、title、content，仅处理给定文档。

[完整逐题结果](results.json) · [生成说明与用量](contexts.json) · [冻结协议](protocol.json) · [逐题命中表](CASES.md) · [校验结果](verification.json)

测试记录：`{"regression.xml": {"tests": 49, "failures": 0, "errors": 0, "skipped": 0}, "full-regression.xml": {"tests": 521, "failures": 0, "errors": 0, "skipped": 0}, "description-integration.xml": {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}}`。

# MongoDB＋Milvus：三种召回模型对比

日期：2026-10-19。真实加载 E5-small、BGE-small-zh-v1.5、GTE-small-zh，分别完成同一份 124 题；反向排列模型顺序再跑一次，所有逐题质量指标一致。每轮产生 3×124×3＝1,116 组查询路线结果。复跑没有增加独立问题数量。

**72 条真实项目文档题的混合 Hit@5：E5 33/72，BGE 41/72，GTE 48/72。** GTE 比 E5 净增加 15 条命中（+20.83 个百分点），但仍有局部退化，且完整多来源覆盖没有提升。当前是开发集模型选择结果，不能代替独立盲测验收。

## 固定条件与口径

- 语料：26 个来源快照，72 条真实项目文档题、32 条合成诊断、8 条多来源、12 条无答案。各模型相同的 222 个结构切片；语料/标签原冻结哈希和本次切片哈希见 [protocol.json](protocol.json)。
- 存储：原生 MongoDB 8.0.16＋真实 Milvus Lite 2.5.1，FLAT/COSINE；不使用 SQLite 代替 Milvus，也不使用内存假数据库。不是 Standalone 的 ANN/并发性能测试。
- 固定 BM25 Top20、向量 Top40、RRF k=60、最终 Top5、背景索引关闭、reranker 关闭。只改变召回模型和其官方规定的编码前缀。禁止语义回退。
- 普通题分别比较 BM25、纯向量、混合；多来源题使用相同子问题前置上下文流程，最多 5 个片段、每来源最多 2 个、序列化预算 6,000 字符。
- Hit@5：Top5 覆盖至少一个已标有效证据组；MRR@5：首次覆盖所需证据组的排名倒数；证据组覆盖率：已覆盖的必需证据组比例。标签不是穷尽相关 chunk，因此不称为完整 chunk Recall@5。
- 本次评测使用新双库实现及结构切片，先前 SQLite v3 的分数不能直接作为本次模型基线。没有根据本次题目改标签、调融合权重或训练模型。

## 真实项目文档题（72）

| 模型 | 纯向量 Hit@5 | 混合 Hit@5 | 纯向量 MRR@5 | 混合 MRR@5 |
|---|---:|---:|---:|---:|
| intfloat/multilingual-e5-small | 25/72（34.7%） | 33/72（45.8%） | 0.264 | 0.305 |
| BAAI/bge-small-zh-v1.5 | 41/72（56.9%） | 41/72（56.9%） | 0.439 | 0.470 |
| thenlper/gte-small-zh | 48/72（66.7%） | 48/72（66.7%） | 0.537 | 0.483 |

共同 BM25 基线为 35/72（48.6%），MRR@5 为 0.321；执行器逐题验证三个模型的 BM25 返回跨度完全一致。E5 混合命中低于 BM25，说明混合召回并不保证更好。GTE 的混合与纯向量命中数相同，但混合 MRR 更低，说明关键词路线也可能拉低正确片段的排序。

| 对 E5 的真实题逐题比较 | 纯向量新增/丢失命中 | 混合新增/丢失命中 |
|---|---:|---:|
| BAAI/bge-small-zh-v1.5 | 18 / 2 | 10 / 2 |
| thenlper/gte-small-zh | 26 / 3 | 16 / 1 |

GTE 混合相对 E5 的 1 条丢失命中是 `repository-054`（英文问题：How can I change mode within an interactive session?）。中文模型在当前中文为主的题集占优，并不证明可全面替代多语言模型。各项是相关开发样本，未做独立样本显著性或泛化保证。

## 合成、多来源与无答案

| 模型 | 合成纯向量 Hit@5（32） | 合成混合 Hit@5（32） | 多来源混合证据组覆盖 | 多来源混合完整覆盖（8） |
|---|---:|---:|---:|---:|
| intfloat/multilingual-e5-small | 27/32（84.4%） | 25/32（78.1%） | 60.4% | 3/8 |
| BAAI/bge-small-zh-v1.5 | 27/32（84.4%） | 27/32（84.4%） | 50.0% | 3/8 |
| thenlper/gte-small-zh | 25/32（78.1%） | 27/32（84.4%） | 60.4% | 3/8 |

三个模型在 12 条无答案题上，纯向量和混合都返回候选（12/12）；BM25 返回 11/12。这只是候选非空率，既不是幻觉率，也不是拒答准确率。没有校准相关性阈值或评估最终生成答案。

多来源完整覆盖均为 3/8，不能宣称换模型解决了跨资料补证。GTE 混合的已标证据组覆盖为 60.4%，BGE 为 50.0%；BGE 在真实单问上的提升并不意味着所有任务受益。

## 模型、token 与本机成本

| 模型 | 参数量 | 维数 | 索引片段 token 上限实测 | 超出模型上限的片段 | 混合查询 p95 两轮观测（毫秒） |
|---|---:|---:|---:|---:|---:|
| intfloat/multilingual-e5-small | 117.7M | 384 | 282/512 | 0 | 51.6 / 51.4 |
| BAAI/bge-small-zh-v1.5 | 24.0M | 512 | 334/512 | 0 | 37.1 / 35.9 |
| thenlper/gte-small-zh | 30.3M | 512 | 334/512 | 0 | 32.5 / 32.8 |

CPU 推理，包含查询编码、MongoDB 读取、Milvus 检索和融合，排除首次模型加载；每轮每题每路线仅测一次，模型按顺序执行，本机还有其他工作负载。表中耗时是观测值，不用于服务 SLA 或严格硬件效率结论。首个模型还承担库初始化/缓存成本，模型加载耗时不能横比。模型文件已在本地缓存；没有调用托管 Embedding API。

该语料没有 token 截断；这不保证其他长文和复杂表格也不截断，当前切片仍有固定字符上限。三个模型的同维数并不表示向量可混用；模型名、commit、编码前缀均纳入索引身份。

| 模型 | 固定 commit | 查询前缀 |
|---|---|---|
| intfloat/multilingual-e5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | `query: ` |
| BAAI/bge-small-zh-v1.5 | `7999e1d3359715c523056ef9478215996d62a620` | `为这个句子生成表示以用于检索相关文章：` |
| thenlper/gte-small-zh | `af7bd46fbb00b3a6963c8dd7f1786ddfbfbe973a` | `无` |

编码依据：[BGE 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[GTE 模型卡](https://huggingface.co/thenlper/gte-small-zh)、[E5 模型卡](https://huggingface.co/intfloat/multilingual-e5-small)。前缀遵循各模型要求，不把三者都套用 E5 编码方式。

## 落地判断

当前中文资料开发集可优先试用 GTE-small-zh＋BM25/RRF；BGE-small-zh 也优于当前 E5 基线。代码已支持三者配置及独立索引重建，默认 E5 暂保留，避免把开发集胜出写成全场景最优。下一轮验收需要新增独立真实资料与英文/中英混合查询，冻结后再决定默认模型。

本轮不同时更换 reranker，避免把粗召回与精排收益混算。bge-reranker、稀疏学习模型、BM25 参数、阈值拒答和主题路由均没有在本轮得到效果证明。

## 复现与逐题检查

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_models
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_models \
  --models thenlper/gte-small-zh BAAI/bge-small-zh-v1.5 intfloat/multilingual-e5-small \
  --output evaluation/knowledge_rag/model_comparison/repeat_reverse
.venv/bin/python -m evaluation.knowledge_rag.render_model_report
```

需真实 MongoDB（默认 localhost:27028）和已缓存模型；执行器自动启动独立 Milvus Lite，使用并删除自己创建的临时数据库。读取模型失败、语义降级、题集哈希变化或 BM25 基线漂移均导致失败。

- [首轮完整结果](results.json)、[反向顺序完整结果](repeat_reverse/results.json)、[冻结协议](protocol.json)。
- [124 题命中对照](CASES.md)：每个模型的三路线与原始题号。
- [执行器](../compare_models.py)、[报告生成器](../render_model_report.py)。

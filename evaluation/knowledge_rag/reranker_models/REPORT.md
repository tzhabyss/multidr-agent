# 固定 GTE 召回的多精排模型对照

日期：2026-10-19。实际经过 MongoDB＋Milvus Lite 的 search/build_context 路径；三个精排器使用完全相同的有序候选池。保持 26 来源、222 切片、124 道开发题及原标注不变。

本轮结果：BGE base 与 mMiniLM 同为64/72命中；BGE v2-m3 为65/72，只多命中1题，CPU普通查询P95约3.53秒（mMiniLM约0.48秒），多来源完整覆盖均为6/8。当前CPU实验更倾向保留 mMiniLM 作为速度与质量的基线；v2-m3 可作为偏重排序质量的候选，但不能把30秒评测预算下的结果直接当作5秒生产配置成绩。

## 72 道真实项目文档题

| 精排器 | Hit@5 | MRR@5 | 相比 mMiniLM 新增/丢失命中 | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|
| 关闭精排 | 48/72（66.7%） | 0.483 | — | 38.2 / 41.9 |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 64/72（88.9%） | 0.808 | 0/0 | 442.2 / 481.8 |
| BAAI/bge-reranker-base | 64/72（88.9%） | 0.817 | 1/1 | 1029.8 / 1118.3 |
| BAAI/bge-reranker-v2-m3 | 65/72（90.3%） | 0.841 | 1/0 | 3330.4 / 3531.4 |

同一普通题前20候选池中，只有 **66/72** 题含已标证据。其余 6 题不能由仅改变排列的精排器补回。Hit@5 表示前五是否命中已标原文跨度，不是穷尽相关性标注的 Recall，也不是回答准确率。

## 其他题型

| 精排器 | 合成 Hit@5（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案返回候选（12） |
|---|---:|---:|---:|---:|
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 32/32 | 6/8 | 81.2% | 12/12 |
| BAAI/bge-reranker-base | 32/32 | 6/8 | 81.2% | 12/12 |
| BAAI/bge-reranker-v2-m3 | 32/32 | 6/8 | 81.2% | 12/12 |

返回候选不表示回答错误，也不表示正确拒答。精排没有增加拒答阈值；本次没有调用生成模型，未验证 Worker 最终报告是否更好。

## 与生产5秒精排预算的距离

| 精排器 | 实际调用数 | 其中耗时超过5秒 | 最慢单次精排 ms |
|---|---:|---:|---:|
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 133 | 0 | 755.8 |
| BAAI/bge-reranker-base | 133 | 0 | 1776.1 |
| BAAI/bge-reranker-v2-m3 | 133 | 3 | 5775.0 |

这里统计30秒评测预算下的观测耗时，并未把请求重跑为5秒预算；超过5秒说明相应调用不能据此宣称符合生产默认等待预算。生产超时会回退融合结果，而且仍在运行的任务可能使后续精排忙碌，不能直接套用本表质量。

## 冻结与运行协议

- 所有模型固定 CPU、FP32、最大输入512 token，输入为问题＋标题＋章节＋原文；不执行模型仓库自定义代码。原 mMiniLM 的 tokenizer 上限本身为512。
- 召回固定 GTE-small-zh、BM25 Top20、向量 Top40、RRF k=60，关闭背景索引；普通题精排前20，多来源每个子查询最多40。
- 新实验逐题校验关闭精排结果与此前保存结果完全相同；每次精排调用的问题、输入片段和顺序都必须与 mMiniLM 基线完全相同。验证摘要见 verification.json。
- 向量降级、精排超时或非法输出使实验中止，不能作为成功样本计入。评测精排预算30秒；生产默认5秒，冷启动及并发不在此表结论内。
- 模型按顺序执行，没有同时跑其他模型评测；耗时包含查询编码、双库访问及精排，排除模型首次加载。每题单次观测，未随机交叉安排模型运行顺序，温度和系统负载仍可能影响结果。
- 这是同一开发集上的模型选择，未增加题目或调整标注，不是独立盲测；8道多来源题样本尤其小。

此处限制所有模型最多512 token，只评当前短片段精排，不评价长上下文模型的完整能力。

## 模型与复现

| 模型 | 固定 commit | 单份权重文件 MB |
|---|---|---:|
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | `1427fd652930e4ba29e8149678df786c240d8825` | 470.6 |
| BAAI/bge-reranker-base | `2cfc18c9415c912f9d8155881c133215df768a70` | 1112.2 |
| BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 2271.1 |

BGE 模型信息来自官方模型卡：[base](https://huggingface.co/BAAI/bge-reranker-base)、[v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。本项目数字全部来自本地逐题记录，未引用官方榜单作为项目实测。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_rerank --models thenlper/gte-small-zh --reranker BAAI/bge-reranker-base --reference evaluation/knowledge_rag/rerank_comparison/results.json --output evaluation/knowledge_rag/reranker_models/bge_base
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_rerank --models thenlper/gte-small-zh --reranker BAAI/bge-reranker-v2-m3 --reference evaluation/knowledge_rag/rerank_comparison/results.json --output evaluation/knowledge_rag/reranker_models/bge_v2_m3
.venv/bin/python -m evaluation.knowledge_rag.render_reranker_models
```

需要已缓存模型、真实 MongoDB；脚本启动私有 Milvus Lite，只清理本次临时库。全局默认运行配置未自动更改。

也核对了 [GTE multilingual reranker](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) 的官方用法和配置：它使用自定义 NewForSequenceClassification/auto_map，官方示例需要 trust_remote_code=True。当前加载器的标准架构实验没有为其做适配，本轮不报告该模型的性能或质量成绩。

- [逐题比较](CASES.md)、[候选池一致性与结果重算验证](verification.json)、[权重记录](weights.json)、[代码哈希](code-manifest.json)。
- 完整结果：[mMiniLM](../rerank_comparison/results.json)、[BGE base](bge_base/results.json)、[BGE v2-m3](bge_v2_m3/results.json)。

## 相比 mMiniLM 的命中变化

- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1：新增 无；丢失 无。
- BAAI/bge-reranker-base：新增 repository-004；丢失 repository-063。
- BAAI/bge-reranker-v2-m3：新增 repository-004；丢失 无。

## 后续待验证的优化

针对6道候选池缺已标证据的真实题，优先比较更大的精排候选池、原查询与改写查询并行召回、主题软加权。保留原查询和跨主题候选，避免缩写、命令名或误分类造成漏召回。它们尚未在本轮实现或评测，不计入上述收益；仍需固定本题集做消融，再用独立新题验证。

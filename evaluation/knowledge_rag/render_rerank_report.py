"""Render paired reranker outcomes; never assume that enabling must improve."""
import json
from pathlib import Path

ROOT=Path(__file__).parent/'rerank_comparison'


def render(root=ROOT):
    r=json.loads((root/'results.json').read_text());models=r['models']
    if len(models)!=3:raise ValueError('all three paired runs must finish')
    lines=['# 双库 RAG：启用 Reranker 是否改善检索','',
           '日期：2026-10-19。在相同冻结 124 题上，对 GTE-small-zh、BGE-small-zh、E5-small 分别做启用/关闭精排的配对实验；实际通过 MongoDB＋Milvus Lite 的生产 search/build_context 路径执行。没有修改标注、切片、模型查询前缀或融合参数。','',
           f'精排器：`{r["reranker"]}`；固定 commit `{r["reranker_revision"]}`。这是当前已有的多语言 CrossEncoder，不是 BGE Reranker。','',
           '## 72 条真实项目文档题','',
           '| 召回模型 | 关闭 Hit@5 | 启用 Hit@5 | 关闭 MRR@5 | 启用 MRR@5 | 新增/丢失命中 |','|---|---:|---:|---:|---:|---:|']
    for name,entry in models.items():
        s=entry['summary']['repository'];off,on=s['off'],s['on']
        rows=[c for c in entry['cases'] if c['group']=='repository']
        wins=sum(c['routes']['on']['hit_at_5']>c['routes']['off']['hit_at_5'] for c in rows)
        losses=sum(c['routes']['on']['hit_at_5']<c['routes']['off']['hit_at_5'] for c in rows)
        lines.append(f'| {name} | {round(off["hit_at_5"]*72)}/72（{off["hit_at_5"]:.1%}） | {round(on["hit_at_5"]*72)}/72（{on["hit_at_5"]:.1%}） | {off["mrr_at_5"]:.3f} | {on["mrr_at_5"]:.3f} | {wins}/{losses} |')
    gte=models['thenlper/gte-small-zh']['summary']['repository']
    delta=gte['on']['hit_at_5']-gte['off']['hit_at_5']
    lines+=['',f'GTE 混合召回启用精排后，Hit@5 变化 **{delta*100:+.2f} 个百分点**。这只描述当前开发集和该精排器的实测结果，不外推到其他模型、语料或最终回答准确率。','',
            '## 区分召回失败与排序失败','',
            '| 召回模型 | 进入重排的候选池含已标证据（最多20） | 关闭 Top5 命中 | 启用 Top5 命中 | 候选池缺证据 | 有候选证据但精排后仍未入前五 |','|---|---:|---:|---:|---:|---:|']
    for name,entry in models.items():
        rows=[c for c in entry['cases'] if c['group']=='repository']
        available=sum(c['candidate_availability']['any_group'] for c in rows)
        off=sum(c['routes']['off']['hit_at_5'] for c in rows);on=sum(c['routes']['on']['hit_at_5'] for c in rows)
        lines.append(f'| {name} | {available}/72 | {off}/72 | {on}/72 | {72-available} | {available-on} |')
    lines+=['','这里的候选池指标按已标跨度计算，不是“语义上所有相关片段”的穷尽 Recall。候选池缺少已标证据时，单靠重排无法补回；候选池内可找到证据但 Top5 未命中时，才直接指向候选选择/排序问题。','',
            '## 合成诊断、多来源与无答案','',
            '| 模型 | 合成 Hit@5 关闭→启用（32） | 多来源完整覆盖 关闭→启用（8） | 多来源证据组覆盖 关闭→启用 |','|---|---:|---:|---:|']
    for name,entry in models.items():
        s=entry['summary']['synthetic'];m=entry['summary']['multi_source']
        lines.append(f'| {name} | {round(s["off"]["hit_at_5"]*32)}/32 → {round(s["on"]["hit_at_5"]*32)}/32 | {round(m["off"]["all_groups_at_5"]*8)}/8 → {round(m["on"]["all_groups_at_5"]*8)}/8 | {m["off"]["evidence_group_coverage_at_5"]:.1%} → {m["on"]["evidence_group_coverage_at_5"]:.1%} |')
    lines+=['','无答案题只统计候选非空率，不代表正确拒答：','',
            '| 模型 | 关闭返回候选 | 启用返回候选 |','|---|---:|---:|']
    for name,entry in models.items():
        s=entry['summary']['negative']
        lines.append(f'| {name} | {round(s["off"]["candidate_nonempty"]*12)}/12 | {round(s["on"]["candidate_nonempty"]*12)}/12 |')
    lines+=['','重排当前只改变顺序，不按校准阈值拒绝候选；本次未评最终生成、引用正确率或无依据问题拒答率。','',
            '## 耗时','',
            '| 模型 | 关闭 p50/p95（ms） | 启用 p50/p95（ms） |','|---|---:|---:|']
    for name,entry in models.items():
        s=entry['summary']['repository']
        lines.append(f'| {name} | {s["off"]["p50_ms"]:.1f}/{s["off"]["p95_ms"]:.1f} | {s["on"]["p50_ms"]:.1f}/{s["on"]["p95_ms"]:.1f} |')
    lines+=['','固定 CPU，耗时包含查询编码、MongoDB/Milvus 读取和排序，排除首次模型加载。每题每路线观测一次，先关闭再开启，可能存在缓存/调度影响；不是服务 SLA，也没有并发压测。','',
            '## 协议和实现边界','',
            '- 26 来源快照、222 切片、124 条开发题，运行前核对原语料/标签哈希及切片哈希。真实题72、合成32、多来源8、无答案12分别统计。',
            '- 三个召回模型固定为此前模型比较的 commit；关闭精排的每题返回跨度必须与此前保存的混合结果完全一致，否则中止。',
            '- BM25 Top20、向量 Top40、RRF k=60、背景索引关闭。普通 Top5 查询将融合前20名送入精排；多来源上下文按现有 search(limit=20) 逻辑将最多40名送入精排。',
            '- 精排联合读取问题、标题、章节和原文；实际候选输入、输出排列、耗时均保存。普通题还核对启用前候选池前五与关闭结果相同。没有使用 gold 标签决定排序。',
            '- 精排超时、异常、非法排列，或向量降级，都使该次评测失败；不把降级结果当成精排收益。评测使用30秒精排等待预算，生产默认5秒，未以此测试冷启动是否满足生产超时。',
            '- 本次可复现结果只支持该精排器和当前题集，未尝试 BGE Reranker、调融合权重或根据结果改标签。','',
            '## CPU 运行问题','',
            '首次 CPU 预热发生 SIGBUS，尚未进入正式题集。模型缓存 SHA-256 与其 blob 标识相同。将加载后的权重复制成独立连续张量后，CPU 预热及正式配对评测可运行；MPS 独立预热也可运行。底层触发原因未完全定位，不将其归因为已证实的模型文件损坏。',
            '生产加载器增加 CPU 权重物化；数值、dtype、模型结构和排序方式不变。模型加载包含这一步，不计入查询延迟。该处理不构成所有原生运行时故障的通用防护。','',
            '## 复现与记录','',
            '```bash','HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_rerank',
            '.venv/bin/python -m evaluation.knowledge_rag.render_rerank_report','```','',
            '需真实 MongoDB（默认127.0.0.1:27028）、缓存的三个召回模型与上述精排模型。执行器启动私有 Milvus Lite，只清理自己生成的临时数据库。','',
            '- [完整逐题结果、候选池与排列](results.json)、[冻结协议](protocol.json)、[逐题前后对照](CASES.md)。',
            '- [相关回归 JUnit](regression.xml)：28 项通过；[运行探针](runtime-probes.json)、[代码哈希](code-manifest.json)。',
            '- [配对执行器](../compare_rerank.py)、[报告生成器](../render_rerank_report.py)。',
            '- [此前仅替换召回模型的对照](../model_comparison/REPORT.md)。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    names=list(models);base=models[names[0]]
    lines=['# Reranker 逐题配对结果','','正例 V 为 Hit@5，多来源为完整证据组覆盖，负例为候选非空；所有列按关闭→开启。','',
           '| 题号 | 类别 | 问题 | '+' | '.join(names)+' |', '|---|---|---|'+'---:|'*len(names)]
    for i,c in enumerate(base['cases']):
        key='candidate_nonempty' if c['group']=='negative' else 'all_groups_at_5' if c['group']=='multi_source' else 'hit_at_5'
        values=[]
        for entry in models.values():
            row=entry['cases'][i]
            values.append(f'{row["routes"]["off"][key]} → {row["routes"]["on"][key]}')
        lines.append('| '+' | '.join([c['id'],c['group'],c['query'].replace('|','/'),*values])+' |')
    (root/'CASES.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':render()

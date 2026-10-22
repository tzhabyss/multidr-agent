"""Compare rerankers on exactly the same ordered GTE candidate pools."""
import hashlib
import json
from pathlib import Path

from .metrics_v3 import score
from .run_v3 import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / 'reranker_models'
EMBEDDING = 'thenlper/gte-small-zh'


def render():
    paths = [ROOT/'rerank_comparison/results.json', OUT/'bge_base/results.json', OUT/'bge_v2_m3/results.json']
    reports = [json.loads(p.read_text()) for p in paths]
    manifest, _, cases = load_dataset()
    entries = [r['models'][EMBEDDING] for r in reports]
    reference = entries[0]
    proof = {'embedding': EMBEDDING, 'models': {}, 'result_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    for report, entry in zip(reports, entries):
        assert report['dataset_manifest'] == manifest
        assert report['chunk_sha256'] == reports[0]['chunk_sha256']
        assert len(entry['cases']) == len(cases) == 124
        pools = []
        for row, base, case in zip(entry['cases'], reference['cases'], cases):
            assert row['id'] == base['id'] == case['id']
            assert row['routes']['off']['hits'] == base['routes']['off']['hits']
            inputs = [(c['query'], c['input']) for c in row['rerank_calls']]
            assert inputs == [(c['query'], c['input']) for c in base['rerank_calls']]
            pools.append((row['id'], inputs))
            for route in ['off', 'on']:
                for key, value in score(case['evidence_groups'], row['routes'][route]['hits']).items():
                    assert row['routes'][route][key] == value
        digest = hashlib.sha256(json.dumps(pools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        proof['models'][report['reranker']] = {'cases': len(cases), 'calls': sum(len(r['rerank_calls']) for r in entry['cases']), 'ordered_candidate_sha256': digest}
    assert len({v['ordered_candidate_sha256'] for v in proof['models'].values()}) == 1
    OUT.mkdir(exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 固定 GTE 召回的多精排模型对照', '',
             '日期：2026-10-19。实际经过 MongoDB＋Milvus Lite 的 search/build_context 路径；三个精排器使用完全相同的有序候选池。保持 26 来源、222 切片、124 道开发题及原标注不变。', '',
             '本轮结果：BGE base 与 mMiniLM 同为64/72命中；BGE v2-m3 为65/72，只多命中1题，CPU普通查询P95约3.53秒（mMiniLM约0.48秒），多来源完整覆盖均为6/8。当前CPU实验更倾向保留 mMiniLM 作为速度与质量的基线；v2-m3 可作为偏重排序质量的候选，但不能把30秒评测预算下的结果直接当作5秒生产配置成绩。', '',
             '## 72 道真实项目文档题', '',
             '| 精排器 | Hit@5 | MRR@5 | 相比 mMiniLM 新增/丢失命中 | CPU p50 / p95 ms |',
             '|---|---:|---:|---:|---:|']
    off = reference['summary']['repository']['off']
    lines.append(f'| 关闭精排 | {round(off["hit_at_5"]*72)}/72（{off["hit_at_5"]:.1%}） | {off["mrr_at_5"]:.3f} | — | {off["p50_ms"]:.1f} / {off["p95_ms"]:.1f} |')
    changes = []
    for report, entry in zip(reports, entries):
        s = entry['summary']['repository']['on']
        wins, losses = [], []
        for row, base in zip(entry['cases'], reference['cases']):
            if row['group'] != 'repository': continue
            delta = row['routes']['on']['hit_at_5'] - base['routes']['on']['hit_at_5']
            if delta > 0: wins.append(row['id'])
            if delta < 0: losses.append(row['id'])
        changes.append({'model': report['reranker'], 'wins': wins, 'losses': losses})
        lines.append(f'| {report["reranker"]} | {round(s["hit_at_5"]*72)}/72（{s["hit_at_5"]:.1%}） | {s["mrr_at_5"]:.3f} | {len(wins)}/{len(losses)} | {s["p50_ms"]:.1f} / {s["p95_ms"]:.1f} |')
    available = sum(c['candidate_availability']['any_group'] for c in reference['cases'] if c['group']=='repository')
    lines += ['', f'同一普通题前20候选池中，只有 **{available}/72** 题含已标证据。其余 {72-available} 题不能由仅改变排列的精排器补回。Hit@5 表示前五是否命中已标原文跨度，不是穷尽相关性标注的 Recall，也不是回答准确率。', '',
              '## 其他题型', '', '| 精排器 | 合成 Hit@5（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案返回候选（12） |', '|---|---:|---:|---:|---:|']
    for report, entry in zip(reports, entries):
        syn = entry['summary']['synthetic']['on']; multi = entry['summary']['multi_source']['on']; neg = entry['summary']['negative']['on']
        lines.append(f'| {report["reranker"]} | {round(syn["hit_at_5"]*32)}/32 | {round(multi["all_groups_at_5"]*8)}/8 | {multi["evidence_group_coverage_at_5"]:.1%} | {round(neg["candidate_nonempty"]*12)}/12 |')
    lines += ['', '返回候选不表示回答错误，也不表示正确拒答。精排没有增加拒答阈值；本次没有调用生成模型，未验证 Worker 最终报告是否更好。', '',
              '## 与生产5秒精排预算的距离', '',
              '| 精排器 | 实际调用数 | 其中耗时超过5秒 | 最慢单次精排 ms |', '|---|---:|---:|---:|']
    for report, entry in zip(reports, entries):
        calls = [call for case in entry['cases'] for call in case['rerank_calls']]
        lines.append(f'| {report["reranker"]} | {len(calls)} | {sum(c["rerank_ms"]>5000 for c in calls)} | {max(c["rerank_ms"] for c in calls):.1f} |')
    lines += ['', '这里统计30秒评测预算下的观测耗时，并未把请求重跑为5秒预算；超过5秒说明相应调用不能据此宣称符合生产默认等待预算。生产超时会回退融合结果，而且仍在运行的任务可能使后续精排忙碌，不能直接套用本表质量。', '',
              '## 冻结与运行协议', '',
              '- 所有模型固定 CPU、FP32、最大输入512 token，输入为问题＋标题＋章节＋原文；不执行模型仓库自定义代码。原 mMiniLM 的 tokenizer 上限本身为512。',
              '- 召回固定 GTE-small-zh、BM25 Top20、向量 Top40、RRF k=60，关闭背景索引；普通题精排前20，多来源每个子查询最多40。',
              '- 新实验逐题校验关闭精排结果与此前保存结果完全相同；每次精排调用的问题、输入片段和顺序都必须与 mMiniLM 基线完全相同。验证摘要见 verification.json。',
              '- 向量降级、精排超时或非法输出使实验中止，不能作为成功样本计入。评测精排预算30秒；生产默认5秒，冷启动及并发不在此表结论内。',
              '- 模型按顺序执行，没有同时跑其他模型评测；耗时包含查询编码、双库访问及精排，排除模型首次加载。每题单次观测，未随机交叉安排模型运行顺序，温度和系统负载仍可能影响结果。',
              '- 这是同一开发集上的模型选择，未增加题目或调整标注，不是独立盲测；8道多来源题样本尤其小。', '',
              '此处限制所有模型最多512 token，只评当前短片段精排，不评价长上下文模型的完整能力。', '',
              '## 模型与复现', '', '| 模型 | 固定 commit | 单份权重文件 MB |', '|---|---|---:|']
    weights = {w['model']: w for w in json.loads((OUT/'weights.json').read_text())}
    for report in reports:
        weight = weights[report['reranker']]
        assert weight['revision'] == report['reranker_revision']
        lines.append(f'| {report["reranker"]} | `{report["reranker_revision"]}` | {weight["file_bytes"]/1e6:.1f} |')
    lines += ['', 'BGE 模型信息来自官方模型卡：[base](https://huggingface.co/BAAI/bge-reranker-base)、[v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)。本项目数字全部来自本地逐题记录，未引用官方榜单作为项目实测。', '', '```bash']
    for model, folder in [('BAAI/bge-reranker-base','bge_base'), ('BAAI/bge-reranker-v2-m3','bge_v2_m3')]:
        lines.append(f'HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_rerank --models {EMBEDDING} --reranker {model} --reference evaluation/knowledge_rag/rerank_comparison/results.json --output evaluation/knowledge_rag/reranker_models/{folder}')
    lines += ['.venv/bin/python -m evaluation.knowledge_rag.render_reranker_models', '```', '',
              '需要已缓存模型、真实 MongoDB；脚本启动私有 Milvus Lite，只清理本次临时库。全局默认运行配置未自动更改。', '',
              '也核对了 [GTE multilingual reranker](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) 的官方用法和配置：它使用自定义 NewForSequenceClassification/auto_map，官方示例需要 trust_remote_code=True。当前加载器的标准架构实验没有为其做适配，本轮不报告该模型的性能或质量成绩。', '',
              '- [逐题比较](CASES.md)、[候选池一致性与结果重算验证](verification.json)、[权重记录](weights.json)、[代码哈希](code-manifest.json)。',
              '- 完整结果：[mMiniLM](../rerank_comparison/results.json)、[BGE base](bge_base/results.json)、[BGE v2-m3](bge_v2_m3/results.json)。', '', '## 相比 mMiniLM 的命中变化', '']
    for change in changes:
        lines += [f'- {change["model"]}：新增 {", ".join(change["wins"]) or "无"}；丢失 {", ".join(change["losses"]) or "无"}。']
    lines += ['', '## 后续待验证的优化', '',
              '针对6道候选池缺已标证据的真实题，优先比较更大的精排候选池、原查询与改写查询并行召回、主题软加权。保留原查询和跨主题候选，避免缩写、命令名或误分类造成漏召回。它们尚未在本轮实现或评测，不计入上述收益；仍需固定本题集做消融，再用独立新题验证。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    detail = ['# 精排模型逐题比较', '', '真实/合成题列为 Hit@5，多来源列为完整覆盖，无答案列为候选非空。', '',
              '| 题号 | 分组 | 问题 | 关闭精排 | '+' | '.join(r['reranker'] for r in reports)+' |', '|---|---|---|---:|'+'---:|'*len(reports)]
    for i, case in enumerate(cases):
        key = 'candidate_nonempty' if case['group']=='negative' else 'all_groups_at_5' if case['group']=='multi_source' else 'hit_at_5'
        values = [str(reference['cases'][i]['routes']['off'][key])]+[str(e['cases'][i]['routes']['on'][key]) for e in entries]
        detail.append('| '+' | '.join([case['id'], case['group'], case['query'].replace('|','/'), *values])+' |')
    (OUT/'CASES.md').write_text('\n'.join(detail)+'\n')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    render()

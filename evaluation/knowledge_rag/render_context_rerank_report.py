"""Verify fixed candidates, exact evidence, and report reranking tradeoffs."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from deepresearch.knowledge.reranker import document_for_rerank
from .compare_context_rerank import ROUTES, summarize
from .metrics_v3 import score
from .run_v3 import load_dataset

OUT = Path(__file__).parent/'context_rerank'
NAMES = {'mini_original': 'mMiniLM原文', 'mini_neighbors': 'mMiniLM＋邻接原文',
         'bge_original': 'BGE v2-m3原文', 'bge_neighbors': 'BGE v2-m3＋邻接原文'}


def first_rank(case, hits):
    return next((i for i in range(1, len(hits)+1)
                 if score(case['evidence_groups'], hits, k=i)['hit_at_5']), None)


def render():
    manifest, docs, cases = load_dataset()
    docmap = {d['id']: d for d in docs}
    r = json.loads((OUT/'results.json').read_text())
    prior = json.loads((OUT.parent/'document_context/results.json').read_text())
    assert manifest == r['protocol']['dataset_manifest']
    assert r['protocol']['routes'] == ROUTES
    phases = r['protocol']['phases']
    assert hashlib.sha256((OUT/'phase1-results.json').read_bytes()).hexdigest() == phases['initial']['results_sha256']
    assert hashlib.sha256((OUT/'followup/results.json').read_bytes()).hexdigest() == phases['adaptive_followup']['results_sha256']
    assert len(r['cases']) == len(cases) == 124
    calls = 0
    for row, case, old in zip(r['cases'], cases, prior['cases']):
        assert row['id'] == case['id'] == old['id']
        expected = old['routes']['descriptions']
        base = row['routes']['mini_original']
        assert base['hits'] == expected['hits']
        for route, config in ROUTES.items():
            v = row['routes'][route]
            assert all(v[k] == value for k, value in score(case['evidence_groups'], v['hits']).items())
            assert v['channels'] == expected['channels']
            assert [(c['query'], c['input']) for c in v['calls']] == [(c['query'], c['input']) for c in expected['calls']]
            pool = [h for c in v['calls'] for h in c['input']]
            assert v['candidate_availability'] == score(case['evidence_groups'], pool, k=len(pool))
            key = lambda h: (h['document'], h['start'], h['end'])
            for call, channel in zip(v['calls'], v['channels']):
                assert call['count'] == len(call['input']) <= channel['budget']
                assert sorted(map(key, call['input'])) == sorted(map(key, call['output']))
                assert list(map(key, call['input'])) == list(map(key, channel['selected']))
                assert max(call['target_pair_tokens']) <= 512
                for hit in [*call['input'], *call['output'], *v['hits']]:
                    content = docmap[hit['document']]['content']
                    assert hit['content'] == content[hit['start']:hit['end']]
                for hit, digest, counts in zip(call['input'], call['scoring_text_sha256'], call['neighbor_char_counts']):
                    doc = docmap[hit['document']]
                    # Recover the frozen heading for exactly this source span.
                    from deepresearch.knowledge.dual_store import structured_chunks
                    part = next(p for p in structured_chunks(doc['content']) if (p['start'], p['end']) == (hit['start'], hit['end']))
                    value = {**hit, 'title': doc['title'], 'heading': part['path']}
                    radius = config['neighbors']
                    if radius:
                        value['_rerank_neighbors'] = {'before': doc['content'][max(0,hit['start']-radius):hit['start']],
                                                       'after': doc['content'][hit['end']:hit['end']+radius]}
                    text = document_for_rerank(value)
                    assert hashlib.sha256(text.encode()).hexdigest() == digest
                    assert counts == {k:len(v) for k,v in value.get('_rerank_neighbors', {}).items()}
                calls += 1
    for group, s in r['summary'].items():
        assert s == summarize([row for row in r['cases'] if row['group'] == group])
    real = [row for row in r['cases'] if row['group'] == 'repository']
    changes, timings, rank_changes = {}, {}, {}
    for route in ROUTES:
        changes[route] = {'wins': [row['id'] for row in real if row['routes'][route]['hit_at_5'] > row['routes']['mini_original']['hit_at_5']],
                          'losses': [row['id'] for row in real if row['routes'][route]['hit_at_5'] < row['routes']['mini_original']['hit_at_5']]}
        rank_changes[route] = {'top1_hits': sum(row['routes'][route]['mrr_at_5'] == 1 for row in real),
            'mrr_improved_cases': sum(row['routes'][route]['mrr_at_5'] > row['routes']['mini_original']['mrr_at_5'] for row in real),
            'mrr_worsened_cases': sum(row['routes'][route]['mrr_at_5'] < row['routes']['mini_original']['mrr_at_5'] for row in real)}
        observed = [c for row in r['cases'] for c in row['routes'][route]['calls']]
        timings[route] = {'calls': len(observed), 'over_default_5s': sum(c['rerank_ms'] > 5000 for c in observed),
                          'max_rerank_ms': max(c['rerank_ms'] for c in observed),
                          'neighbor_tail_truncated_pairs': sum(n > 512 for c in observed for n in c['pair_tokens']),
                          'original_target_truncated_pairs': sum(n > 512 for c in observed for n in c['target_pair_tokens'])}
    lines = ['# 背景索引后的精排优化对照', '',
        '日期：2026-10-20。固定26份文档、222个切片、124题、已冻结的背景说明、GTE-small-zh、关键词5/向量15。真实MongoDB＋Milvus Lite，四组每次调用的有序候选片段完全相同。原文基线逐题复现上次背景说明实验。', '',
        '第一轮预定三组为现有mMiniLM、mMiniLM增加切片前后各120字符的原文、BGE-reranker-v2-m3。看到72道真实题中mMiniLM邻接命中率提高但MRR下降后，追加BGE＋相同邻接原文的第四组，仍完整运行124题。追加选择与当时结果已冻结在followup；这是自适应开发实验，不是四组都预注册的盲测。', '',
        '没有把生成的说明交给精排，没有增加新的LLM调用。普通候选池20，Worker每子问题40，最终Top5与6000字符/每来源2段预算不变。', '',
        '## 72道真实文档题', '',
        '| 方案 | Hit@5 | MRR@5 | 新增/丢失 | CPU查询P95 ms | 精排超默认5秒 /133次 |',
        '|---|---:|---:|---:|---:|---:|']
    for route in ROUTES:
        s = r['summary']['repository'][route]
        lines.append(f'| {NAMES[route]} | {round(s["hit_at_5"]*72)}/72（{s["hit_at_5"]:.1%}） | {s["mrr_at_5"]:.3f} | {len(changes[route]["wins"])}/{len(changes[route]["losses"])} | {s["p95_ms"]:.1f} | {timings[route]["over_default_5s"]}/133 |')
    lines += ['', '综合判断：BGE原文精排相对现有方案同时提高单问Hit@5与MRR，多来源完整覆盖及证据组覆盖保持不变，但CPU耗时显著增加。BGE邻接组单问覆盖最高，却降低MRR和多来源证据组覆盖；mMiniLM邻接组也有排序及多来源回退。因此没有将邻接精排设为全局默认。',
              '若以检索质量优先，可选择BGE原文精排并保持reranker_neighbor_chars=0；需要低延迟时保留mMiniLM。两者都仍需独立新题验证，不能把本轮开发集结果当成泛化保证。']
    lines += ['', '这是30秒实验精排预算下的质量结果。5秒超时后的RRF回退与后续忙碌回退没有作为另一条质量路线重跑；不能直接声称与生产默认5秒完全等价。计时包含诊断token化开销，固定顺序CPU单次观测，不是生产SLA。', '',
              '## 原先剩余5题', '', '| ID / 问题 | mMiniLM | mMiniLM邻接 | BGE v2-m3 | BGE邻接 |', '|---|---:|---:|---:|---:|']
    diagnostics = {}
    for row, case in zip(r['cases'], cases):
        if row['group'] != 'repository' or row['routes']['mini_original']['hit_at_5']:
            continue
        ranks = {route: first_rank(case, row['routes'][route]['calls'][0]['output']) for route in ROUTES}
        diagnostics[row['id']] = ranks
        lines.append('| '+row['id']+' '+row['query']+' | '+' | '.join(str(ranks[route]) if ranks[route] else '未入池' for route in ROUTES)+' |')
    lines += ['', '表内数字为已标证据的精排位置。repository-042的证据未进入固定候选池，更换精排不能找回这段证据。未对单题修改候选或说明，未按题拼接最优路线。', '',
              '## 其他题型', '', '| 方案 | 合成命中 /32 | 多来源完整覆盖 /8 | 多来源证据组覆盖 | 无答案返回候选 /12 |', '|---|---:|---:|---:|---:|']
    for route in ROUTES:
        s, m, n = (r['summary'][g][route] for g in ['synthetic', 'multi_source', 'negative'])
        lines.append(f'| {NAMES[route]} | {round(s["hit_at_5"]*32)}/32 | {round(m["all_groups_at_5"]*8)}/8 | {m["evidence_group_coverage_at_5"]:.1%} | {round(n["candidate_nonempty"]*12)}/12 |')
    lines += ['', 'mMiniLM邻接组虽然单问Hit@5提高，但MRR下降，且多来源完整覆盖由7/8降至6/8。multi-004中邻接内容包含研发投入，导致不含该事实的目标切片被排高；最终只返回目标切片，造成证据缺失。因此不只按单问命中率选择推荐方案。', '',
              '首位命中及排名变化：`'+json.dumps(rank_changes, ensure_ascii=False)+'`。']
    lines += ['', '## 证据与窗口', '',
        '邻接段从同一个Mongo快照读取，验证目标切片字节一致，严格限于前后各120字符。精排输入先保留目标原文，再追加带标识的邻接段；返回结果重新从原候选取回，不扩大证据位置，也不把邻接内容作为已返回的引用。配置关闭时保持原有输入。查询改写后的精排也复用同一准备路径。', '',
        '| 方案 | 邻接尾部超512的输入对 | 目标原文超512的输入对 | 最大精排毫秒 |', '|---|---:|---:|---:|']
    for route in ROUTES:
        t = timings[route]
        lines.append(f'| {NAMES[route]} | {t["neighbor_tail_truncated_pairs"]} | {t["original_target_truncated_pairs"]} | {t["max_rerank_ms"]:.1f} |')
    lines += ['', '目标原文未截断仅适用于本次冻结语料；没有实现任意新文档的通用token自适应切块。邻接尾部若超长，仍由模型512窗口截断。精排看到邻接信息，不表示最终Top5返回了该邻接证据；评分始终使用实际返回的原文跨度。', '', '## 新增与退步', '']
    for route in list(ROUTES)[1:]:
        lines.append(f'- {NAMES[route]}新增：'+(', '.join(changes[route]['wins']) or '无')+'；丢失：'+(', '.join(changes[route]['losses']) or '无')+'。')
    lines += ['', '开发集重复实验，题目与相关性标签未扩充，尚无独立盲测。Hit@5检查已标原文跨度，不是最终回答准确率；未标全的替代证据可能存在。无答案候选非空不代表拒答能力。没有Worker端到端生成评测，不直接声称最终研究报告质量提高。', '',
        '## 配置与复现', '',
        '双库配置新增 `knowledge.reranker_neighbor_chars: 120` 可试用相邻原文精排，默认0；不需重建索引，也不会自动启用RAG。SQLite兼容后端显式拒绝该非零配置。BGE路线使用 `reranker_model: BAAI/bge-reranker-v2-m3`，本实验30秒预算；实际部署应同时核对 `reranker_timeout_seconds` 和Worker总检索预算。所有组的全局默认保持原样。', '',
        '本轮基于已启用背景索引、GTE和5/15策略的实验配置；单独修改精排模型不自动启用这些前提。自有语料需单独生成背景说明并reindex，评测缓存仅覆盖冻结的26份文档。普通题测Top5/候选20，多来源测Worker build_context的Top5/子问题候选40；实际默认Worker为top_k=3，未据此宣称其同样达到表内命中率。并行Worker的精排忙碌回退未做压测。', '',
        '```bash', 'HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_context_rerank',
        'HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_context_rerank --routes bge_neighbors --output evaluation/knowledge_rag/context_rerank/followup',
        '.venv/bin/python -m evaluation.knowledge_rag.merge_context_rerank',
        '.venv/bin/python -m evaluation.knowledge_rag.render_context_rerank_report', '```', '',
        '[逐题数据及排序输入哈希](results.json) · [全部题目命中](CASES.md) · [协议](protocol.json) · [校验](verification.json)']
    tests = {}
    for name in ['regression.xml', 'full-regression.xml']:
        path = OUT/name
        if path.exists():
            suites = ET.parse(path).getroot().findall('testsuite')
            tests[name] = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
            assert all(tests[name][k] == 0 for k in ['failures', 'errors', 'skipped'])
    lines += ['', '测试：`'+json.dumps(tests, ensure_ascii=False)+'`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    table = ['# 全部124题', '', '| ID | 分组 | 问题 | mMiniLM | mMiniLM邻接 | BGE v2-m3 | BGE邻接 |', '|---|---|---|---|---|---|---|']
    for row in r['cases']:
        cells = [str(row['routes'][route].get('hit_at_5', '候选非空='+str(row['routes'][route].get('candidate_nonempty')))) for route in ROUTES]
        table.append('| '+row['id']+' | '+row['group']+' | '+row['query'].replace('|', '\\|')+' | '+' | '.join(cells)+' |')
    (OUT/'CASES.md').write_text('\n'.join(table)+'\n')
    proof = {'cases': 124, 'rescored_case_routes': 496, 'verified_search_calls': calls,
             'same_ordered_candidates': True, 'canonical_returned_source_bytes': True,
             'scoring_text_hashes_verified_from_sources': True, 'changes': changes, 'diagnostics': diagnostics,
             'timings': timings, 'rank_changes': rank_changes, 'tests': tests,
             'results_sha256': hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
    files = ['deepresearch/knowledge/reranker.py', 'deepresearch/knowledge/dual_store.py',
             'deepresearch/knowledge/query_rewrite.py', 'deepresearch/knowledge/__init__.py',
             'deepresearch/runtime_config.py', 'config/runtime.yaml',
             'evaluation/knowledge_rag/compare_context_rerank.py', 'evaluation/knowledge_rag/render_context_rerank_report.py',
             'evaluation/knowledge_rag/merge_context_rerank.py',
             'tests/test_knowledge_rerank_neighbors.py', 'tests/test_knowledge_dual_integration.py']
    (OUT/'code-manifest.json').write_text(json.dumps({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    render()

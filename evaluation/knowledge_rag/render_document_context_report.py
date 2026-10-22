"""Rescore every route, check source bytes, and publish gains AND regressions."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .compare_document_context import ROUTES, summarize
from .metrics_v3 import score
from .run_v3 import load_dataset

OUT = Path(__file__).parent/'document_context'
NAMES = {'plain': '原文基线', 'metadata': '标题＋章节', 'descriptions': '切片背景说明'}
FOCUS = ['repository-013', 'repository-014', 'repository-037', 'repository-038']


def render():
    manifest, docs, cases = load_dataset()
    docmap = {d['id']: d for d in docs}
    r = json.loads((OUT/'results.json').read_text())
    cache = json.loads((OUT/'contexts.json').read_text())
    assert hashlib.sha256((OUT/'contexts.json').read_bytes()).hexdigest() == r['protocol']['contexts_sha256']
    assert r['protocol']['dataset_manifest'] == manifest and len(r['cases']) == len(cases) == 124
    calls = 0
    for row, case in zip(r['cases'], cases):
        assert row['id'] == case['id']
        base = row['routes']['plain']
        for route in ROUTES:
            v = row['routes'][route]
            assert all(v[k] == n for k, n in score(case['evidence_groups'], v['hits']).items())
            pool = [h for c in v['calls'] for h in c['input']]
            coarse = [h for c in v['channels'] for h in c['fused']]
            assert v['candidate_availability'] == score(case['evidence_groups'], pool, k=len(pool))
            assert v['coarse_availability'] == score(case['evidence_groups'], coarse, k=len(coarse))
            assert [c['count'] for c in v['calls']] == [c['count'] for c in base['calls']]
            for call, channel in zip(v['calls'], v['channels']):
                assert channel['vector_status'] == 'available'
                assert channel['selected'] == channel['validated']
                key = lambda h: (h['document'], h['start'], h['end'])
                assert sorted(map(key, call['input'])) == sorted(map(key, call['output']))
                assert list(map(key, call['input'])) == list(map(key, channel['selected']))
                for h in [*call['input'], *call['output']]:
                    assert h['content'] == docmap[h['document']]['content'][h['start']:h['end']]
                calls += 1
    for group, summary in r['summary'].items():
        assert summary == summarize([row for row in r['cases'] if row['group'] == group])
    real = [row for row in r['cases'] if row['group'] == 'repository']
    changes = {}
    for route in ROUTES:
        changes[route] = {
            'wins': [row['id'] for row in real if row['routes'][route]['hit_at_5'] > row['routes']['plain']['hit_at_5']],
            'losses': [row['id'] for row in real if row['routes'][route]['hit_at_5'] < row['routes']['plain']['hit_at_5']]}
    lines = ['# 文档侧背景说明对照实验', '',
        '日期：2026-10-20。26份冻结文档、222个相同切片、124道开发题，实际MongoDB＋Milvus Lite。固定GTE-small-zh＋mMiniLM、关键词5/向量15、BM25 Top20/向量Top40、普通精排池20、Worker子问题池40，最终Top5。查询改写关闭。', '',
        '三组仅改变建索引的文本；说明同时进入BM25与向量编码，精排仍使用原问题、标题、章节和原文。说明不进入Worker的证据片段，不占其6000字符上下文预算，不作为引用。', '',
        '## 72道真实文档题', '',
        '| 索引 | Hit@5 | MRR@5 | 初始候选含证据 | 精排池含证据 | 新增/丢失 | CPU P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for route in ROUTES:
        s = r['summary']['repository'][route]
        coarse = sum(row['routes'][route]['coarse_availability']['hit_at_5'] for row in real)
        selected = sum(row['routes'][route]['candidate_availability']['hit_at_5'] for row in real)
        lines.append(f'| {NAMES[route]} | {round(s["hit_at_5"]*72)}/72（{s["hit_at_5"]:.1%}） | {s["mrr_at_5"]:.3f} | {coarse}/72 | {selected}/72 | {len(changes[route]["wins"])}/{len(changes[route]["losses"])} | {s["p95_ms"]:.1f} |')
    lines += ['', '## 事先定位的4道漏召回题', '',
              '| 题目 | 原文基线 | 标题章节 | 背景说明 |', '|---|---|---|---|']
    diagnostics = {}
    for row, case in zip(r['cases'], cases):
        if row['id'] not in FOCUS:
            continue
        diagnostics[row['id']] = {}
        cells = []
        for route in ROUTES:
            v = row['routes'][route]
            diag = {'coarse': v['coarse_availability']['hit_at_5'],
                    'selected': v['candidate_availability']['hit_at_5'], 'final': v['hit_at_5']}
            diagnostics[row['id']][route] = diag
            cells.append(f'初召回{diag["coarse"]}／入池{diag["selected"]}／Top5命中{diag["final"]}')
        lines.append('| '+case['id']+' '+case['query']+' | '+' | '.join(cells)+' |')
    lines += ['', '两道中文查询对应英文记忆文档，两道涉及CLI命令。README只列出doctor命令，没有明确解释其作用；生成说明中的“自检/检查环境”属于模型对命令与上下文的解释，尚未经代码核实，不能作为原文事实。恢复命令和默认模型/向量类型则在原文有明确依据。', '',
              '## 其他题型', '', '| 索引 | 合成命中（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案仍返回候选（12） |', '|---|---:|---:|---:|---:|']
    for route in ROUTES:
        s = r['summary']['synthetic'][route]
        m = r['summary']['multi_source'][route]
        n = r['summary']['negative'][route]
        lines.append(f'| {NAMES[route]} | {round(s["hit_at_5"]*32)}/32 | {round(m["all_groups_at_5"]*8)}/8 | {m["evidence_group_coverage_at_5"]:.1%} | {round(n["candidate_nonempty"]*12)}/12 |')
    f = r['protocol']['freeze']
    lines += ['', '## 生成成本与索引窗口', '',
        f'说明模型：`{f["protocol"]["model"]}`，temperature=0。222条说明全部冻结后才运行评分；生成缓存共{f["attempt_count"]}次请求，已知输入{f["input_tokens"]:,} token、输出{f["output_tokens"]:,} token。首轮37条未通过长度/格式校验，使用更短的提示补生成；原始记录在generation-v1，失败尝试用量计入总数。另有1次长度诊断请求（输入5,972、输出85 token，未用于索引），记录在generation-diagnostic.json。没有查看问题或标签后修改说明。未换算费用，未验证供应商缓存计费。', '',
        '| 索引 | 建索引秒数（不含说明生成） | 最大编码token | 超512数量 |', '|---|---:|---:|---:|']
    for route, data in r['indexing'].items():
        lines.append(f'| {NAMES[route]} | {data["seconds"]:.2f} | {data["max_index_tokens"]} | {data["overflow_count"]} |')
    lines += ['', '说明生成发生在离线建库阶段；在线检索没有新增LLM调用。每条说明绑定文档SHA-256、原文跨度及切片哈希；未知文档回退普通索引，已知文档切片不匹配拒绝使用。新索引完整回读后，说明与向量指针一起切换；失败保留旧索引。', '',
              '## 得失与边界', '']
    for route in ROUTES[1:]:
        lines.append(f'- {NAMES[route]}新增：'+(', '.join(changes[route]['wins']) or '无')+'；丢失：'+(', '.join(changes[route]['losses']) or '无')+'。')
    lines += ['', '背景说明将72道真实题的已标证据全部带入初始候选并集，其中71题进入精排池。剩余5道Top5未命中题中，4题已入池但精排未进前五，1题被候选选择截掉。',
        'CLI两题的已标命令段分别被mMiniLM排到第20、16名；同义说明没有进入精排文本，因而只能改善初召回。后续可单独测试带明确标识的检索说明参与相关性打分、邻接原文回填或更强精排，不应把生成说明冒充原文。',
        '退步题repository-042：原文明确写出学术检索出错或无结果回退OpenAlex；生成说明主要概括服务器和工具清单，未突出回退条件。证据片段向量排名从第5降到第28，未进入固定精排池。未据此回改该片段的说明或混用逐题最佳结果。']
    lines += ['', '这是同一开发集的对照，不是独立盲测；Hit@5只检查已标原文跨度，替代证据可能未标全。没有评测最终回答、拒答、引用正确率或Worker端到端收益。说明可能包含模型推断；目前只辅助找原文，不能把检索命中提升等同于事实质量提升。CPU单次热运行耗时不代表生产SLA。全局开关不自动启用。', '',
        '方法参考：[Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)。本实验使用现有DeepSeek模型生成中文说明，使用更短字符上限；不是原文全部模型、语料和指标的复现。', '',
        '## 复现与启用', '', '```bash',
        '.venv/bin/python -m evaluation.knowledge_rag.generate_document_context',
        '# 仅当有超长/失败说明时补生成，并记录原尝试成本',
        '.venv/bin/python -m evaluation.knowledge_rag.generate_document_context --retry-invalid',
        'HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_document_context',
        '.venv/bin/python -m evaluation.knowledge_rag.render_document_context_report', '```', '',
        '完整双库配置中设置 `knowledge.document_context_path: evaluation/knowledge_rag/document_context/contexts.json` 后执行 `knowledge reindex`。已有原文、版本和引用位置保持不变；修改说明文件不会直接改动已发布的索引，仍需重新建索引。自有语料可用 `generate_document_context --corpus /path/corpus.json --output /path/context-cache`；输入数组元素需有id、title、content，仅处理给定文档。', '',
        '[完整逐题结果](results.json) · [生成说明与用量](contexts.json) · [冻结协议](protocol.json) · [逐题命中表](CASES.md) · [校验结果](verification.json)']
    tests = {}
    for name in ['regression.xml', 'full-regression.xml', 'description-integration.xml']:
        path = OUT/name
        if path.exists():
            suites = ET.parse(path).getroot().findall('testsuite')
            tests[name] = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
            assert all(tests[name][k] == 0 for k in ['failures', 'errors', 'skipped']), f'unresolved test failures: {name}'
    lines += ['', '测试记录：`'+json.dumps(tests, ensure_ascii=False)+'`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    case_lines = ['# 全部逐题命中', '', '| ID | 分组 | 问题 | 原文 | 标题章节 | 背景说明 |', '|---|---|---|---|---|---|']
    for row in r['cases']:
        cells = []
        for route in ROUTES:
            value = row['routes'][route]
            cells.append(str(value.get('hit_at_5', '候选非空='+str(value.get('candidate_nonempty')))))
        case_lines.append('| '+row['id']+' | '+row['group']+' | '+row['query'].replace('|', '\\|')+' | '+' | '.join(cells)+' |')
    (OUT/'CASES.md').write_text('\n'.join(case_lines)+'\n')
    proof = {'cases': 124, 'case_routes_rescored': 124*3, 'verified_calls': calls,
             'canonical_source_bytes_only': True, 'same_actual_rerank_counts': True,
             'baseline_exactly_reproduced': True, 'changes': changes, 'focus_diagnostics': diagnostics,
             'tests': tests, 'results_sha256': hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
    files = ['deepresearch/knowledge/document_context.py', 'deepresearch/knowledge/dual_store.py',
             'deepresearch/knowledge/__init__.py', 'deepresearch/runtime_config.py',
             'evaluation/knowledge_rag/generate_document_context.py', 'evaluation/knowledge_rag/compare_document_context.py',
             'evaluation/knowledge_rag/render_document_context_report.py', 'config/runtime.yaml',
             'tests/test_knowledge_document_context.py', 'tests/test_knowledge_dual_integration.py']
    (OUT/'code-manifest.json').write_text(json.dumps({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    render()

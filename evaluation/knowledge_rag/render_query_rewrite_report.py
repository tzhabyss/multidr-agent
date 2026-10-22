"""Recompute scores and render query rewriting without assuming a benefit."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

from .compare_query_rewrite import ROUTES, summarize
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile
from deepresearch.knowledge.query_rewrite import validate_rewrite

ROOT = Path(__file__).parent
OUT = ROOT/'query_rewrite'
NAMES = {'original':'原问题，原候选预算', 'original_expanded':'原问题，双倍候选预算',
         'rewrite_rrf':'原问题＋改写，RRF固定预算', 'rewrite_union':'原问题＋改写，保留原池并扩容'}


def render():
    r = json.loads((OUT/'results.json').read_text())
    manifest, _, cases = load_dataset()
    frozen = json.loads((OUT/'rewrite-freeze.json').read_text())
    rewriting = json.loads((OUT/'rewrites.json').read_text())
    baseline = json.loads((ROOT/'rerank_comparison/results.json').read_text())['models']['thenlper/gte-small-zh']
    assert r['protocol']['dataset_manifest']==manifest
    assert hashlib.sha256((OUT/'rewrites.json').read_bytes()).hexdigest()==frozen['rewrites_sha256']==r['protocol']['rewrite_sha256']
    assert len(r['cases'])==len(cases)==124
    anchor_checks = 0
    for row, case, old in zip(r['cases'], cases, baseline['cases']):
        assert row['id']==case['id']==old['id']
        assert row['routes']['original']['hits']==old['routes']['on']['hits']
        for route in ROUTES:
            result = row['routes'][route]
            for key, value in score(case['evidence_groups'], result['hits']).items():
                assert result[key]==value
            calls = result['calls']
            original_calls = row['routes']['original']['calls']
            assert [c['query'] for c in calls]==[c['query'] for c in original_calls]
            cap = (40 if case['group']=='multi_source' else 20)*(2 if route in {'original_expanded','rewrite_union'} else 1)
            assert all(c['count']<=cap for c in calls)
            if route=='rewrite_union':
                for call, original in zip(calls, original_calls):
                    assert call['input'][:len(original['input'])]==original['input']
                    anchor_checks += 1
        expected_inputs = [(c['query'],c['input']) for c in old['rerank_calls']]
        assert [(c['query'],c['input']) for c in row['routes']['original']['calls']]==expected_inputs
    for group, summary in r['summary'].items():
        assert summary==summarize([row for row in r['cases'] if row['group']==group])
    proof = {'cases':124, 'paired_routes':len(ROUTES), 'rescored_case_routes':124*len(ROUTES),
             'original_baseline_identical': True, 'original_pool_prefix_checks':anchor_checks,
             'rewrites_sha256':frozen['rewrites_sha256'],
             'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()}
    test_summaries = {}
    for name in ['regression.xml','full-regression.xml']:
        path = OUT/name
        if path.exists():
            suites = ET.parse(path).getroot().findall('testsuite')
            summary = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
            assert summary['failures']==summary['errors']==summary['skipped']==0
            test_summaries[name] = summary
    proof['test_summaries'] = test_summaries
    (OUT/'verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n')
    generation = list(rewriting['queries'].values())
    assert all(validate_rewrite(g['original_query'],g['rewritten_query'])==g['rewritten_query']
               for g in generation if g['status']=='rewritten')
    times = [g['ms'] for g in generation]
    base_score = r['summary']['repository']['original']['hit_at_5']
    best_rewrite = max(r['summary']['repository'][route]['hit_at_5'] for route in ['rewrite_rrf','rewrite_union'])
    conclusion = ('本次通用查询改写没有改善已标证据命中，不建议默认启用。' if best_rewrite<=base_score
                  else '本次有改写路线提高已标证据命中，仍需结合排序质量、耗时和独立新题验证。')
    lines = ['# 查询改写配对评测', '',
             conclusion, '',
             '日期：2026-10-19。相同26份快照、222个切片、124道冻结开发题。真实 MongoDB＋Milvus Lite，固定 GTE-small-zh 和 mMiniLM CrossEncoder，CPU、最大512 token，精排等待预算5秒。没有改标注、切片、向量模型、BM25/向量候选数或RRF常数。', '',
             '## 72道真实项目文档题', '',
             '| 方案 | Hit@5 | MRR@5 | 已标证据进入精排池 | 检索 p50/p95 ms | 加生成后的 p95估算 ms |', '|---|---:|---:|---:|---:|---:|']
    real = [c for c in r['cases'] if c['group']=='repository']
    for route in ROUTES:
        s = r['summary']['repository'][route]
        available = sum(c['routes'][route]['candidate_availability']['hit_at_5'] for c in real)
        lines.append(f'| {NAMES[route]} | {round(s["hit_at_5"]*72)}/72（{s["hit_at_5"]:.1%}） | {s["mrr_at_5"]:.3f} | {available}/72 | {s["p50_ms"]:.1f}/{s["p95_ms"]:.1f} | {s["end_to_end_p95_ms"]:.1f} |')
    pool_means = {route: mean(c['routes'][route]['calls'][0]['count'] for c in real) for route in ROUTES}
    lines += ['', '原问题的普通查询精排池上限20，Worker每个子问题40；双倍预算对应40/80。RRF改写路线保持相同精排预算，用原问题与改写问题各自的混合结果再做等权RRF（k=60）；扩容路线保留整个原池，再加入改写召回的新片段。所有路线最终均用原问题精排。', '',
              '真实题每次实际进入精排的平均候选数：'+ '；'.join(f'{NAMES[route]} {pool_means[route]:.2f}' for route in ROUTES)+'。改写与原查询召回高度重合，扩容路线平均仅新增'+f'{pool_means["rewrite_union"]-pool_means["original"]:.2f}个片段；其已标证据入池题数未增加。', '',
              '**归因对照：** RRF固定预算与原问题路线比较，观察同预算改写效果；改写扩容与原问题扩容比较，观察收益能否超过仅多取候选。某些问题实际候选少于上限，未伪造补足数量。', '',
              '## 命中得失', '', '| 比较（前者相对后者） | 新增命中 | 丢失命中 | MRR变化 |', '|---|---:|---:|---:|']
    comparisons = [('rewrite_rrf','original'), ('rewrite_union','original'), ('rewrite_union','original_expanded'), ('original_expanded','original')]
    changes = []
    for a,b in comparisons:
        wins = [c['id'] for c in real if c['routes'][a]['hit_at_5']>c['routes'][b]['hit_at_5']]
        losses = [c['id'] for c in real if c['routes'][a]['hit_at_5']<c['routes'][b]['hit_at_5']]
        delta = r['summary']['repository'][a]['mrr_at_5']-r['summary']['repository'][b]['mrr_at_5']
        lines.append(f'| {NAMES[a]} / {NAMES[b]} | {len(wins)} | {len(losses)} | {delta:+.3f} |')
        changes.append({'route':a, 'reference':b, 'wins':wins, 'losses':losses})
    lines += ['', '## 其他题型', '', '| 方案 | 合成Hit@5（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案返回候选（12） |', '|---|---:|---:|---:|---:|']
    for route in ROUTES:
        syn = r['summary']['synthetic'][route]; multi = r['summary']['multi_source'][route]; neg = r['summary']['negative'][route]
        lines.append(f'| {NAMES[route]} | {round(syn["hit_at_5"]*32)}/32 | {round(multi["all_groups_at_5"]*8)}/8 | {multi["evidence_group_coverage_at_5"]:.1%} | {round(neg["candidate_nonempty"]*12)}/12 |')
    lines += ['', '多来源沿用真实 build_context：最多5个片段、每来源最多2个、序列化6000字符，保留子问题首选片段。该评测使用核心检索和上下文组装，未加入生产路径的改写用量摘要；摘要在真实Worker中也会占字符预算，生产集成另有预算与失败降级测试。', '',
              '## 改写生成与代价', '',
              f'- 改写模型 `{rewriting["protocol"]["model"]}`，temperature=0，最大输出256 token，单条超时15秒，API生成并发4。',
              f'- 共 {len(generation)} 条不同查询；状态 {json.dumps(frozen["statuses"],ensure_ascii=False)}。未改写或失败的查询保留原查询，并仍计入评测分母。',
              f'- API生成耗时 p50/p95：{percentile(times,.5):.1f}/{percentile(times,.95):.1f}ms，最长{max(times):.1f}ms。',
              f'- API报告输入/输出token共 {sum(g["input_tokens"] for g in generation)}/{sum(g["output_tokens"] for g in generation)}。这是预生成一次的用量，不是四条路线各付一遍。',
              '- 先生成并冻结改写文件与prompt哈希，之后才执行检索和打分。生成器不打开语料文件，模型每次只接收question，未接触原文、正确答案、gold跨度或检索结果。',
              '- 约束保留命令、标识符、缩写、数字和引号内术语；校验缺失的字面量会追加回查询，超长或无效结果降级。语义是否保持一致没有独立人工标注，不能以字面量校验代替语义正确率。',
              '- 检索耗时在预生成改写后测量，包含两路召回及精排；“加生成后”是逐题相加估算，多子问题取其已测生成耗时最大值，未重放真实线上端到端请求。模型加载不计入，固定路线顺序、每题单次观测，不能当作SLA。', '',
              '## 方法来源与实现边界', '',
              '参考[Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283)的先改写再检索思路，以及[RAG-Fusion](https://arxiv.org/abs/2402.03367)的多查询融合。本次为一次提示式改写＋保留原查询＋融合＋原问题精排，没有训练重写器或强化学习，不宣称复现论文全部实验。', '',
              '所有实验是相同开发集上的消融；Hit@5只验证已标原文跨度，不是穷尽Recall或最终回答准确率。未评生成、引用准确率、幻觉或拒答。默认改写开关保持false。', '',
              '## 生产开关', '', '```yaml', 'knowledge:', '  query_rewrite_enabled: true', '  query_rewrite_timeout_seconds: 15', '  query_rewrite_fusion: rrf  # 或 anchored_union', '```', '',
              '将这些字段合并进完整配置；不会自动启用RAG或更换召回/精排模型。Worker复用strategy模型做改写，每个任务最多4条查询；生成阶段最多用知识检索总预算的一半，异常退回原查询，原文/来源/版本检查仍生效。', '',
              '## 复现和记录', '', '```bash',
              '.venv/bin/python -m evaluation.knowledge_rag.generate_query_rewrites',
              'HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_query_rewrite',
              '.venv/bin/python -m evaluation.knowledge_rag.render_query_rewrite_report', '```', '',
              '生成器复用同协议缓存，禁止静默换prompt或模型；检索需要MongoDB和已缓存的固定版本模型，会启动私有Milvus Lite，只清理本次临时库。', '',
              '- [完整逐题/候选池记录](results.json)、[逐题比较](CASES.md)、[生成改写和用量](rewrites.json)。',
              '- [冻结改写](rewrite-freeze.json)、[运行协议](protocol.json)、[验证结果](verification.json)、[代码哈希](code-manifest.json)。',
              '- 回归记录：'+ '；'.join(f'[{name}]({name})：{summary["tests"]}通过，0失败、0跳过' for name,summary in test_summaries.items())+'。全项目运行已启用真实双库集成检查。', '', '## 发生变化的题目', '']
    for change in changes:
        lines.append(f'- {NAMES[change["route"]]} / {NAMES[change["reference"]]}：新增 {", ".join(change["wins"]) or "无"}；丢失 {", ".join(change["losses"]) or "无"}。')
    changed_ids = set(cid for c in changes for cid in c['wins']+c['losses'])
    lines += ['', '| 题号 | 原问题 | 改写查询 |', '|---|---|---|']
    for row in real:
        if row['id'] in changed_ids:
            lines.append('| '+' | '.join([row['id'], row['query'].replace('|','/'), '；'.join(x['rewritten_query'] or '保留原问题' for x in row['rewrites']).replace('|','/')])+' |')
    audits = []
    by_id = {c['id']:c for c in cases}
    for row in real:
        if row['id'] not in changed_ids: continue
        audit = {'id':row['id'], 'query':row['query'], 'rewrites':row['rewrites'], 'routes':{}}
        groups = by_id[row['id']]['evidence_groups']
        for route in ROUTES:
            call = row['routes'][route]['calls'][0]
            def first_hit(values):
                return next((i for i in range(1,len(values)+1) if score(groups,values,k=i)['hit_at_5']),None)
            audit['routes'][route] = {'pre_rerank_known_evidence_rank':first_hit(call['input']),
                                      'post_rerank_known_evidence_rank':first_hit(call['output'])}
        audits.append(audit)
    (OUT/'failure-analysis.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n')
    lines += ['', '## 下降机制核对', '',
              '- repository-036（记忆两通道召回数量）：原问题中已标片段在候选第20位，精排后第1位；RRF改写融合截断后片段不在精排池，扩容保留原池则仍精排到第1位。这是候选裁剪造成的漏取。',
              '- repository-069（API密钥能否提交Git）：已标 SECURITY 明文政策在基线最终第5位，在其余三条路线变成第6位。证据仍在池内，增加候选后排序竞争使其超出Top5。其他相关安全片段仍被召回，因此不能将一次已标跨度漏取直接认定为最终不安全回答。',
              '- 以上机制依据冻结逐题排列核对；不修改gold补分，不据此编写题目专用同义词规则。[变化题的候选/排序位置](failure-analysis.json)。', '',
              '仅扩大原问题候选池在本轮提高了Hit@5和多来源完整覆盖，但MRR略降（0.808→0.803）；同样不能仅凭一个指标自动启用。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    detail = ['# 查询改写逐题对照', '', '真实/合成列为Hit@5，多来源列为完整覆盖，无答案列为候选非空。', '',
              '| 题号 | 分组 | 问题 | '+' | '.join(NAMES.values())+' |', '|---|---|---|'+'---:|'*len(ROUTES)]
    for row in r['cases']:
        key = 'candidate_nonempty' if row['group']=='negative' else 'all_groups_at_5' if row['group']=='multi_source' else 'hit_at_5'
        detail.append('| '+' | '.join([row['id'],row['group'],row['query'].replace('|','/'), *[str(row['routes'][route][key]) for route in ROUTES]])+' |')
    (OUT/'CASES.md').write_text('\n'.join(detail)+'\n')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__=='__main__': render()

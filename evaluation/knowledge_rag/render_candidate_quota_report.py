"""Verify channel-prefix reservations and render fixed-budget paired outcomes."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

from deepresearch.knowledge.candidate_selection import POLICIES, select_candidates
from .compare_candidate_quota import ROUTES, summarize
from .metrics_v3 import score
from .run_v3 import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT/'candidate_quota'
NAMES = {'rrf':'原RRF', 'quota_equal':'关键词10＋向量10',
         'quota_dense':'关键词5＋向量15', 'quota_lexical':'关键词15＋向量5'}


def key(h): return h['document'],h['start'],h['end']


def render():
    r=json.loads((OUT/'results.json').read_text())
    manifest,_,cases=load_dataset()
    prior=json.loads((ROOT/'rerank_comparison/results.json').read_text())['models']['thenlper/gte-small-zh']
    assert r['protocol']['dataset_manifest']==manifest
    assert r['protocol']['policies']==json.loads(json.dumps(POLICIES))
    assert len(r['cases'])==len(cases)==124
    verified_calls=0
    changed={p:0 for p in ROUTES}
    for row,case,old in zip(r['cases'],cases,prior['cases']):
        assert row['id']==case['id']==old['id']
        base=row['routes']['rrf']
        assert base['hits']==old['routes']['on']['hits']
        assert [(c['query'],c['input']) for c in base['calls']]==[(c['query'],c['input']) for c in old['rerank_calls']]
        for policy in ROUTES:
            route=row['routes'][policy]
            for metric,value in score(case['evidence_groups'],route['hits']).items():
                assert route[metric]==value
            assert len(route['calls'])==len(route['channels'])==len(base['calls'])
            pool=[h for call in route['calls'] for h in call['input']]
            assert route['candidate_availability']==score(case['evidence_groups'],pool,k=len(pool))
            for call,channel,base_call,base_channel in zip(route['calls'],route['channels'],base['calls'],base['channels']):
                assert all(channel[k]==base_channel[k] for k in ['query','budget','lexical','semantic','fused'])
                assert channel['vector_status']=='available' and call['query']==channel['query']
                expected=select_candidates([key(h) for h in channel['lexical']], [key(h) for h in channel['semantic']],
                    [key(h) for h in channel['fused']],budget=channel['budget'],policy=policy)
                assert expected==[key(h) for h in channel['selected']]==[key(h) for h in call['input']]
                assert channel['selected']==channel['validated']
                assert call['count']==base_call['count']<=channel['budget']
                assert sorted(key(h) for h in call['input'])==sorted(key(h) for h in call['output'])
                changed[policy]+=set(expected)!=set(key(h) for h in base_call['input'])
                verified_calls+=1
    for group,summary in r['summary'].items():
        assert summary==summarize([c for c in r['cases'] if c['group']==group])
    tests={}
    for name in ['regression.xml','full-regression.xml']:
        path=OUT/name
        if path.exists():
            suites=ET.parse(path).getroot().findall('testsuite')
            tests[name]={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
            assert all(tests[name][k]==0 for k in ['failures','errors','skipped'])
    proof={'cases':124,'rescored_case_routes':124*len(ROUTES),'verified_search_calls':verified_calls,
           'baseline_and_raw_routes_identical':True,'same_actual_rerank_counts':True,
           'changed_candidate_pools':changed,'test_summaries':tests,
           'results_sha256':hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    real=[c for c in r['cases'] if c['group']=='repository']
    base=r['summary']['repository']['rrf']
    lead=[]
    for policy in ROUTES[1:]:
        s=r['summary']['repository'][policy]
        lead.append(f'{NAMES[policy]}：{round(s["hit_at_5"]*72)}/72，MRR {s["mrr_at_5"]:.3f}')
    lines=['# 固定候选预算的分路保留实验','',
           f'基线为{round(base["hit_at_5"]*72)}/72、MRR {base["mrr_at_5"]:.3f}；'+'；'.join(lead)+'。', '',
           '日期：2026-10-19。同一26份快照、222个切片、124道开发题；实际经过MongoDB＋Milvus Lite的search/build_context。固定GTE-small-zh和mMiniLM，CPU、最大512 token、精排预算5秒；查询改写和背景索引均关闭。', '',
           '## 72道真实文档题','',
           '| 候选策略 | Hit@5 | MRR@5 | 已标证据入池 | 新增/丢失命中 | CPU p50/p95 ms |', '|---|---:|---:|---:|---:|---:|']
    changes=[]
    for policy in ROUTES:
        s=r['summary']['repository'][policy]
        wins=[c['id'] for c in real if c['routes'][policy]['hit_at_5']>c['routes']['rrf']['hit_at_5']]
        losses=[c['id'] for c in real if c['routes'][policy]['hit_at_5']<c['routes']['rrf']['hit_at_5']]
        available=sum(c['routes'][policy]['candidate_availability']['hit_at_5'] for c in real)
        changes.append({'policy':policy,'wins':wins,'losses':losses})
        lines.append(f'| {NAMES[policy]} | {round(s["hit_at_5"]*72)}/72（{s["hit_at_5"]:.1%}） | {s["mrr_at_5"]:.3f} | {available}/72 | {len(wins)}/{len(losses)} | {s["p50_ms"]:.1f}/{s["p95_ms"]:.1f} |')
    coarse=sum(score(c['evidence_groups'],row['routes']['rrf']['channels'][0]['fused'],k=len(row['routes']['rrf']['channels'][0]['fused']))['hit_at_5']
               for row,c in zip(r['cases'],cases) if row['group']=='repository')
    lines+=['',f'合并前两路召回列表的完整候选并集中，{coarse}/72题存在已标证据。这是当前BM25 Top20＋向量Top40的已标跨度可达情况；分路保留只选择已有候选，无法产生并集中不存在的片段。', '',
            '## 规则和固定预算','',
            '- 普通查询四组都精排20个候选、最终取5；Worker每个子问题精排40个，保留比例按同一规则计算。每次实际精排数量逐一核对相等。',
            '- 先分别取两路前N名，按chunk_id去重，再从完整RRF列表补齐预算；进入精排前仍按原RRF顺序排列，最终统一用原问题精排。',
            '- 两路共同命中的片段同时满足两路前缀保留要求，占一个精排名额；因此10＋10是两路前缀保障，不是强制各贡献10个互不重叠片段。',
            '- 一路不足名额时取其所有可用候选，再由RRF补齐。原关键词路只取Top20，因此Worker的30＋10方案至多保留这20条关键词候选，其余按规则补齐，未扩大关键词召回数。',
            '- 四组的关键词、向量、完整融合列表及其顺序逐题保持相同；只改变截断前选哪些片段。原文、collection、版本和最终引用门禁保留，精排只接受合法排列。',
            '- 三种比例在评分前写入protocol.json；每个比例统一应用全部题目，没有按题选择最优结果。全局配置仍为rrf。','',
            '## 其他题型','', '| 策略 | 合成Hit@5（32） | 多来源完整覆盖（8） | 多来源证据组覆盖 | 无答案返回候选（12） |', '|---|---:|---:|---:|---:|']
    for policy in ROUTES:
        syn=r['summary']['synthetic'][policy];multi=r['summary']['multi_source'][policy];neg=r['summary']['negative'][policy]
        lines.append(f'| {NAMES[policy]} | {round(syn["hit_at_5"]*32)}/32 | {round(multi["all_groups_at_5"]*8)}/8 | {multi["evidence_group_coverage_at_5"]:.1%} | {round(neg["candidate_nonempty"]*12)}/12 |')
    lines+=['','多来源保持最多5个片段、每来源最多2个、序列化6000字符，并保留子问题首选。无答案候选非空不代表正确拒答或错误回答；本次没有运行最终生成模型。','',
            '## 候选池变化','', '| 策略 | 改变了候选集合的调用数 /133 | 新增已标证据入池题数 | 丢失已标证据入池题数 |', '|---|---:|---:|---:|']
    for policy in ROUTES:
        added=sum(c['routes'][policy]['candidate_availability']['hit_at_5']>c['routes']['rrf']['candidate_availability']['hit_at_5'] for c in real)
        lost=sum(c['routes'][policy]['candidate_availability']['hit_at_5']<c['routes']['rrf']['candidate_availability']['hit_at_5'] for c in real)
        lines.append(f'| {NAMES[policy]} | {changed[policy]}/133 | {added} | {lost} |')
    lines+=['','## 逐题得失','']
    changed_ids=set()
    for change in changes:
        lines.append(f'- {NAMES[change["policy"]]}：新增 {", ".join(change["wins"]) or "无"}；丢失 {", ".join(change["losses"]) or "无"}。')
        changed_ids.update(change['wins']+change['losses'])
    audits=[]
    by_id={c['id']:c for c in cases}
    for row in real:
        if row['id'] not in changed_ids:continue
        group=by_id[row['id']]['evidence_groups']
        def first_known(values):
            return next((i for i in range(1,len(values)+1) if score(group,values,k=i)['hit_at_5']),None)
        audit={'id':row['id'],'query':row['query'],'routes':{}}
        for policy in ROUTES:
            route=row['routes'][policy];channel=route['channels'][0]
            audit['routes'][policy]={**{k:first_known(channel[k]) for k in ['lexical','semantic','fused','selected']},
                                      'post_rerank':first_known(route['calls'][0]['output'])}
        audits.append(audit)
    (OUT/'case-analysis.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n')
    lines+=['','## 为什么5＋15有小幅改善','',
            '- repository-046、repository-048的已标证据分别在向量路第14、11位，原融合列表第23、24位，原Top20截断会漏掉。向量保留前15名后，两段都进入精排并排到第1位；向量只保留前10名没有补回它们。',
            '- repository-064的已标证据在关键词路第10位，未进入向量Top40。关键词只保留前5名时，它被截掉；这是5＋15丢失的一道题。',
            '- 这说明配额会交换召回机会。本轮5＋15净增加一道命中、MRR小幅提高；它适合作为后续独立新题验证的候选，未据此更改全局默认策略。']
    lines+=['','[变化题的两路召回、融合、入池及精排后位置](case-analysis.json)。null表示该列表没有完整覆盖已标证据；未标注替代证据可能存在，不能直接解释为语义上完全无关。','',
            '## 使用和复现','', '```yaml','knowledge:','  candidate_policy: quota_equal  # quota_dense / quota_lexical / rrf','```','',
            '需合并进完整MongoDB＋Milvus配置；仅改变候选选择，不会启用RAG或切换模型。SQLite兼容后端不支持此开关，显式报错，不静默忽略。','',
            '```bash','HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.knowledge_rag.compare_candidate_quota',
            '.venv/bin/python -m evaluation.knowledge_rag.render_candidate_quota_report','```','',
            '需要真实MongoDB和已缓存的固定版本模型。脚本使用私有Milvus Lite与临时Mongo库，只清理本次拥有的数据库。','',
            '## 验证与限制','',
            f'- {verified_calls}次实际search/精排调用核对；124×4条路线指标重算通过。基线最终片段和精排输入与先前保存结果完全一致。',
            '- [逐题结果及完整两路候选](results.json)、[逐题比较](CASES.md)、[协议](protocol.json)、[一致性验证](verification.json)、[代码哈希](code-manifest.json)。',
            '- 回归：'+'；'.join(f'[{name}]({name})：{s["tests"]}通过、0失败、0跳过' for name,s in tests.items())+'。',
            '- 同一开发集上的三种预定比例对照，未做独立盲测。Hit@5衡量已标原文跨度，不是穷尽相关性Recall或最终回答准确率。',
            '- 固定路线顺序、单次热CPU观测，耗时包含双库读取、编码和精排，不包含模型加载；未做并发吞吐测试，不能作为SLA。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    detail=['# 分路保留逐题对照','','真实/合成列为Hit@5，多来源为完整覆盖，无答案为候选非空。','',
            '| 题号 | 分组 | 问题 | '+' | '.join(NAMES.values())+' |','|---|---|---|'+'---:|'*len(ROUTES)]
    for row in r['cases']:
        metric='candidate_nonempty' if row['group']=='negative' else 'all_groups_at_5' if row['group']=='multi_source' else 'hit_at_5'
        detail.append('| '+' | '.join([row['id'],row['group'],row['query'].replace('|','/'), *[str(row['routes'][p][metric]) for p in ROUTES]])+' |')
    (OUT/'CASES.md').write_text('\n'.join(detail)+'\n')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':render()

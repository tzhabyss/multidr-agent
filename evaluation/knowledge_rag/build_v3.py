"""Freeze manually authored retrieval questions and source snapshots before evaluation.

No model generates labels. Alternatives are known valid source spans, not an
exhaustive annotation of every semantically relevant passage in the corpus.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
from .run import ROOT

OUT=Path(__file__).parent/'v3'


def build():
    docs={}
    cases=[]
    def document(key,title,text,uri=None):
        docs[key]={'id':key,'title':title,'content':text,'uri':uri or f'file:///evaluation/{key}'}
    for path in ['README.md','docs/memory-system.md','docs/mcp.md','docs/model-modes.md','docs/progress-stream.md','SECURITY.md']:
        document(path,path,(ROOT/path).read_text())
    def add(query,path,quotes,category,group='repository',as_of=None):
        if isinstance(quotes,str):
            quotes=[quotes]
        alternatives=[{'document':path,'text':q} for q in quotes]
        item={'id':f'{group}-{sum(c["group"]==group for c in cases)+1:03}',
              'group':group,'category':category,'query':query,'evidence_groups':[alternatives]}
        if as_of:
            item['as_of_snapshot']=as_of
        cases.append(item)
        return item
    def multi(query,groups,subqueries,category='multi_source'):
        cases.append({'id':f'multi-{sum(c["group"]=="multi_source" for c in cases)+1:03}',
                      'group':'multi_source','category':category,'query':query,
                      'subqueries':subqueries,'evidence_groups':[[{'document':d,'text':q} for d,q in alternatives] for alternatives in groups]})

    p='README.md'
    data=[
      ('系统证据如何逐级追溯到网页快照？','Source -> SourceSnapshot -> Evidence -> Claim -> ClaimEvidenceLink -> Citation','provenance'),
      ('普通研究默认会同时请求哪两个搜索服务？',['deepseek-web-search` 与 `anysearch-search','Standard 默认并行调用 DeepSeek Search 与 AnySearch'],'exact_identifier'),
      ('一轮研究发现资料不足时怎么办？',['need_more -> 结构化缺口返回 Planner','后续轮次根据 Synthesizer 的具体缺口定向补证'],'multiple_valid_passages'),
      ('如何更换整个项目的运行配置文件？',['DEEPRESEARCH_CONFIG','ResearchService(config_path=...)','--config path/to/runtime.yaml'],'configuration'),
      ('Worker 每次执行是否会共用同一套循环状态？','每个并行 Worker 都创建独立、有界的 `AgentLoop`','isolation'),
      ('搜索引擎摘要是否可以直接作为报告证据？',['搜索摘要不能直接成为证据','正文必须再经增强 Safe Fetch 和证据抽取门禁'],'evidence_boundary'),
      ('Skill 的说明文字和执行授权分别放在哪两个文件？','`SKILL.md` 保存给角色使用的操作说明，`skill.json` 保存角色/Profile 授权','configuration'),
      ('Python 和大模型分别负责哪些研究判断？','LLM 负责语义充分性、冲突和不确定性判断；Python 负责工具授权','responsibility'),
      ('源码最低要求哪个 Python 版本？','需要 Python 3.11 或更高版本。','exact_number'),
      ('macOS 推荐采用什么版本 Python？','建议使用 Python 3.12','exact_number'),
      ('Windows 的 npx.cmd 在 macOS 上怎么处理？','映射为 `npx`','cross_platform'),
      ('浏览器 Chromium 下载不可用时怎样指定已安装的浏览器？','SCRAPLING_EXECUTABLE_PATH','configuration'),
      ('怎样检查本地运行配置是否正常？','python -m deepresearch doctor','command'),
      ('CLI 要恢复相同 run_id 的研究应该调用什么命令？','python -m deepresearch resume bci-research --json','command'),
      ('CLI 返回 partial 和配置错误时的退出码分别是什么？','`3` partial，`1` failed，`2` 配置错误','exact_number'),
      ('输入批量评测 JSONL 最少需要哪两个字段？',['每行至少包含 `case_id` 和 `question`','{"case_id":"case-1","question":"Research question"}'],'data_schema'),
      ('哪些地址会被 Safe Fetch 的 SSRF 防护拦截？','阻止私网、localhost、metadata 地址和内嵌凭据','security'),
      ('allowed domain 与 blocked domain 冲突时谁优先？','allowed/blocked domain 在 Fetch 前执行，blocked 优先','security'),
      ('来源质量 Tier 4 包括哪些页面？','Tier 4：社交、视频、论坛、搜索页和用户投稿页面','classification'),
      ('Tier 4 来源会直接被程序删除吗？','它不会被代码直接删除','evidence_boundary'),
      ('运行生成的报告和审计材料如何放目录？',['报告根目录只放 `report.md` 和 `details/`','Plan、来源、证据、Agent trace、事件、用量和审计文件全部放在 `details/`'],'artifact'),
      ('如何启动本地研究 Web 控制台？','python -m deepresearch web --host 127.0.0.1 --port 8765','command'),
      ('交互式命令行怎样切换对话？','/switch <session_id>  switch conversations','command'),
      ('交互式命令历史默认保存到哪里？','.runtime/shell-history','artifact'),
    ]
    for q,g,c in data:add(q,p,g,c)

    p='docs/memory-system.md'
    data=[
      ('为什么一个会话里的多次研究不会共用 LangGraph 状态？','LangGraph always uses `run_id` as\nits checkpoint `thread_id`','isolation'),
      ('L0 这一层记忆负责什么？','L0: LangGraph checkpoints for pause, resume, and retries.','architecture'),
      ('L1 包含哪些本次研究状态？','L1: current-run state, plans, tasks, gaps, and private worker tool traces.','architecture'),
      ('L3 长期记忆保存什么内容？','L3: atomic facts and cross-session research Episodes','architecture'),
      ('Extractor 和 Writer 会直接查记忆数据库吗？','Extractor, Synthesizer, and Writer never read the memory database directly.','permission'),
      ('项目记忆层是否依赖 Mem0 云端服务？','There is no Mem0\nSDK, Cloud API, REST service, or Mem0 credential in the runtime.','architecture'),
      ('MemoryPacket 会永久保存成新记忆层吗？',['an ephemeral runtime projection, not a new\nmemory layer','discarded after the agent call.'],'lifecycle'),
      ('默认工作上下文多大，什么时候开始持久化压缩？','default working context is 100k estimated tokens. Durable session\ncompaction starts at 60k','exact_number'),
      ('must_keep 会保护哪些用户信息？','user-authored corrections, constraints, decisions,\nnumbers, dates, citations, and unresolved questions','protected_state'),
      ('压缩验证或 Segment 持久化失败时会丢掉热消息吗？','If LLM validation or Segment persistence fails, the hot messages are retained.','failure'),
      ('滚动状态是用增量更新还是每次让模型重写全部摘要？','Rolling state is updated from `StateDelta`; the model no longer rewrites the\nentire previous rolling summary.','architecture'),
      ('记忆检索的两个候选通道各自召回多少条？','FTS5 top 50 and dense-vector top 50','retrieval'),
      ('记忆中的向量采用什么存储类型？','normalized Float32 BLOBs in SQLite','storage'),
      ('记忆编码默认用哪个模型，是多少维？','intfloat/multilingual-e5-small` (384 dimensions)','embedding'),
      ('记忆向量模型不可用时如何标记降级结果？','labelled `hash_vector_fallback`, never `semantic_vector`','failure'),
      ('共享 Redis 会话后端应配置哪些选项？','`memory.session_backend: redis` and `DEEPRESEARCH_REDIS_URL`','configuration'),
    ]
    for q,g,c in data:add(q,p,g,c)

    p='docs/mcp.md'
    data=[
      ('没有 SERPAPI_KEY 时学术搜索先用什么服务？','Semantic Scholar first when `SERPAPI_KEY` is absent','exact_identifier'),
      ('学术检索服务出错或者没有结果会回退到哪里？','falls back to\n  OpenAlex on errors or empty results.','failure'),
      ('如何安装本地PDF和Word支持？','pip install -e ".[documents]"','command'),
      ('Is an OpenAlex project key mandatory?','OpenAlex currently works through its public API without a project API key.','paraphrase'),
      ('外部工具如何限制角色和权限？',['role/profile-scoped `SkillDefinition`','roles` and `profiles` retain the existing fail-closed behavior'],'multiple_valid_passages'),
      ('MCP 服务器广告了很多工具时会全部开放给模型吗？','only names listed beneath `tools` are converted to Skills.','permission'),
      ('stdio 和 streamable_http 两种传输有什么区别？','`stdio` starts a local subprocess. `streamable_http` connects to a remote MCP','architecture'),
      ('MCP 环境变量缺失时怎么处理，密钥会写入产物吗？','A missing variable\nis a startup error for that server; secrets are never written to artifacts.','security'),
      ('MCP observation 类型的结果会直接变成证据吗？','It does not become a source candidate or publishable evidence by itself.','evidence_boundary'),
      ('可选 MCP 服务不可用时，本地 Skill 还能工作吗？','An unavailable optional server is logged and skipped. Local Skills continue\n  to work.','failure'),
      ('关键 MCP 服务连接失败时怎样强制启动失败？','Set `required: true` to make startup fail','configuration'),
      ('每次调用 MCP 工具是否都会重新启动服务进程？','a tool call does not restart its MCP server.','lifecycle'),
    ]
    for q,g,c in data:add(q,p,g,c)

    p='docs/model-modes.md'
    data=[
      ('Which mode has eight workers and six rounds?','| `max` | high-value, difficult research | 8 | 6 |','table'),
      ('How can I change mode within an interactive session?','/mode deep','command'),
      ('Does choosing a stronger model grant additional tool permissions?','Tool access is intentionally not granted by model tier.','permission'),
      ('What does model_tiers configure?','maps the logical `lite`, `default`, and `craft` tiers','configuration'),
      ('research_modes 配置会控制哪些内容？','controls research depth, concurrency, search/fetch budget,\n   wall-clock budget, and the tier used at each stage.','configuration'),
      ('报告审核由研究模式还是 Profile 决定？','Report review is selected by profile, not by research mode.','policy'),
      ('Reviewer 能联网或自己创建来源吗？','The Reviewer cannot browse or create sources.','permission'),
      ('Reviewer 出错时会丢弃 Writer 草稿吗？','A Reviewer failure leaves the Writer draft in\nplace.','failure'),
    ]
    for q,g,c in data:add(q,p,g,c)

    p='docs/progress-stream.md'
    data=[
      ('run_stream 和 resume_stream 的事件契约与 chat_stream 一样吗？','The same contract is available through `run_stream()` and `resume_stream()`.','streaming'),
      ('进度事件是只能解析终端文本吗？','Events are structured data rather than terminal strings','streaming'),
      ('并行 Worker 的进度计数按什么键归并？','Parallel-worker counters are\nreduced by `task_id`','isolation'),
      ('verbose 进度会输出 API 密钥、提示词或网页正文吗？','never keys,\n  prompts, webpage bodies, or raw tool payloads','security'),
      ('心跳事件能否显示虚构的完成百分比？','never\ninvents a completion percentage.','streaming'),
      ('默认项目时区是什么？',['timezone: Asia/Shanghai','project default remains Asia/Shanghai'],'timezone'),
      ('旧 UTC 会话时间戳会被直接重写吗？','convert legacy UTC timestamps to the configured timezone instead of rewriting\nold stored records.','timezone'),
      ('有哪些 Agent Loop 停止原因？','`model_finished`, `tool_budget_exhausted`,\n`max_iterations`, `tool_calling_unavailable`, and `no_executable_skills`','lifecycle'),
    ]
    for q,g,c in data:add(q,p,g,c)

    p='SECURITY.md'
    data=[
      ('真实 API 密钥可以提交到 Git 吗？','Real API keys, access tokens, cookies, request headers and `.env` files must never be committed.','security'),
      ('密钥已经进入提交历史后只删掉文件够吗？','revoke or rotate it before rewriting Git history. Removing the visible line is not sufficient after publication.','security'),
      ('安全测试中类似凭据的字符串是真密钥吗？','These fixtures are not usable credentials','security'),
      ('推送前用哪个测试检查密钥卫生？','python -m pytest tests/test_secret_hygiene.py -q','command'),
    ]
    for q,g,c in data:add(q,p,g,c)

    # Synthetic diagnostics: independently labelled scenario families, never mixed into repository scores.
    for name,amounts in [('星河',(7.2,8.5)),('远山',(2.4,3.1)),('海川',(5.0,5.8)),('晨光',(1.1,1.6))]:
        for year,amount in zip((2023,2024),amounts):
            key=f'synthetic/{name}-{year}.md'
            text=f'# 研发投入\n\n研发投入为{amount}亿元。统计口径包含费用化与资本化研发支出，金额单位为人民币。\n\n# 风险\n\n该数据仅对应本报告期，不代表下一年度。'
            document(key,f'{name}公司{year}年度报告',text)
            add(f'{name}公司{year}年研发投入是多少？',key,f'研发投入为{amount}亿元。','company_year','synthetic')
        add(f'{name}公司2024年研发投入的币种与金额单位是什么？',key,'金额单位为人民币。','units','synthetic')
    for name,unit,price in [('微光','百万次请求','100元'),('流星','百万输入Token','120元'),('云帆','百万输出Token','300元'),('青禾','每小时计算资源','8元')]:
        key=f'synthetic/{name}-pricing.md'
        document(key,f'{name}服务计费说明',f'# 标准档\n\n价格为{price}，计价单位为{unit}。\n\n# 限制\n\n不含税，不同计价单位的金额不能直接比较。')
        add(f'{name}服务标准档多少钱？',key,f'价格为{price}','price','synthetic')
        add(f'{name}服务的价格按什么单位计算？',key,f'计价单位为{unit}','units','synthetic')
    for name,values in [('北辰',(120,180)),('南岭',(90,150))]:
        key=f'synthetic/{name}-table.md'
        table=f'# 测试结果\n\n单位：毫秒，指标为P95延迟，测试环境为单机。\n\n| 模式 | P95延迟 |\n|---|---|\n| 快速 | {values[0]} |\n| 深度 | {values[1]} |'
        document(key,f'{name}服务延迟测试',table)
        add(f'{name}服务快速模式P95延迟是多少？',key,f'| 快速 | {values[0]} |','table','synthetic')
        add(f'{name}服务深度模式P95延迟是多少？',key,f'| 深度 | {values[1]} |','table','synthetic')
        add(f'{name}服务测试表的延迟单位与统计口径是什么？',key,'单位：毫秒，指标为P95延迟','table_context','synthetic')
    # Scope conflict: both statements are retained, with distinct measurement populations.
    document('synthetic/accuracy-a.md','青木系统内部实验','# 评估口径\n\n在内部封闭测试集上准确率为92%，样本量500条。')
    document('synthetic/accuracy-b.md','青木系统外部实验','# 评估口径\n\n在外部开放测试集上准确率为81%，样本量800条。')
    for scope,source,gold in [('内部','a','内部封闭测试集上准确率为92%'),('外部','b','外部开放测试集上准确率为81%')]:
        add(f'青木系统在{scope}测试集上的准确率是多少？',f'synthetic/accuracy-{source}.md',gold,'scope_disambiguation','synthetic')
    # Same URI versions. Historical queries use capture time, not legal effective dates.
    for name,old,new in [('东篱',(30,100),(60,180)),('西溪',(20,80),(45,140))]:
        uri=f'file:///evaluation/{name}-limits.md'
        oldkey=f'synthetic/{name}-limits-v1.md';newkey=f'synthetic/{name}-limits-v2.md'
        document(oldkey,f'{name}服务限额说明',f'# 服务限额\n\n单请求超时为{old[0]}秒，每日配额为{old[1]}次。',uri)
        document(newkey,f'{name}服务限额说明',f'# 服务限额\n\n单请求超时为{new[0]}秒，每日配额为{new[1]}次。',uri)
        add(f'{name}服务当前单请求超时与每日配额是多少？',newkey,f'单请求超时为{new[0]}秒，每日配额为{new[1]}次。','latest_version','synthetic')
        add(f'回到旧版快照已入库时，{name}服务限额是多少？',oldkey,f'单请求超时为{old[0]}秒，每日配额为{old[1]}次。','historical_capture','synthetic',as_of=oldkey)

    multi('比较星河与远山2024年的研发投入。',[[('synthetic/星河-2024.md','研发投入为8.5亿元。')],[('synthetic/远山-2024.md','研发投入为3.1亿元。')]],['星河公司2024研发投入','远山公司2024研发投入'])
    multi('海川2023和2024年的研发投入分别是多少？',[[('synthetic/海川-2023.md','研发投入为5.0亿元。')],[('synthetic/海川-2024.md','研发投入为5.8亿元。')]],['海川公司2023研发投入','海川公司2024研发投入'])
    multi('流星和云帆的计费单位是否相同？',[[('synthetic/流星-pricing.md','计价单位为百万输入Token')],[('synthetic/云帆-pricing.md','计价单位为百万输出Token')]],['流星服务计价单位','云帆服务计价单位'])
    multi('核对星河研发投入、海川研发投入与流星计费单位。',[[('synthetic/星河-2024.md','研发投入为8.5亿元。')],[('synthetic/海川-2024.md','研发投入为5.8亿元。')],[('synthetic/流星-pricing.md','计价单位为百万输入Token')]],['星河公司2024研发投入','海川公司2024研发投入','流星服务计价单位'])
    multi('比较北辰和南岭服务快速模式延迟。',[[('synthetic/北辰-table.md','| 快速 | 120 |')],[('synthetic/南岭-table.md','| 快速 | 90 |')]],['北辰快速模式P95延迟','南岭快速模式P95延迟'])
    multi('青木内外部测试准确率为什么不能不加条件直接比较？',[[('synthetic/accuracy-a.md','内部封闭测试集上准确率为92%')],[('synthetic/accuracy-b.md','外部开放测试集上准确率为81%')]],['青木内部测试准确率与口径','青木外部测试准确率与口径'],'conflicting_scopes')
    multi('研究工具的外部接入权限与模型档位权限如何控制？',[[('docs/mcp.md','only names listed beneath `tools` are converted to Skills.')],[('docs/model-modes.md','Tool access is intentionally not granted by model tier.')]],['MCP tools allowlist','模型档位是否授予工具权限'])
    multi('说明项目默认时区以及记忆向量模型。',[[('docs/progress-stream.md','timezone: Asia/Shanghai')],[('docs/memory-system.md','intfloat/multilingual-e5-small` (384 dimensions)')]],['项目默认时区','记忆默认Embedding模型和维数'])

    for query in [
      '星河公司2025年的研发投入是多少？','远山公司的首席执行官是谁？','晨光公司的2024年净利润是多少？',
      '微光服务的免费试用优惠码是什么？','青禾服务有没有学生折扣？','北辰服务在双机环境下的P99延迟是多少？',
      '青木系统在德语数据集上的准确率是多少？','东篱服务的客服电话是什么？','西溪服务最近一次宕机是什么时候？',
      '这个项目在官方BrowseComp排行榜的得分是多少？','项目使用的生产数据库管理员密码是什么？','论文资料库中AGI实现日期是哪一天？']:
        cases.append({'id':f'negative-{sum(c["group"]=="negative" for c in cases)+1:03}',
                      'group':'negative','category':'unanswerable','query':query,'evidence_groups':[]})

    # Every occurrence of each approved alternative within its source is valid.
    # This does not invent relevance for other documents with similar wording.
    for case in cases:
        for gi, group in enumerate(case['evidence_groups']):
            expanded=[]
            seen=set()
            for alt in group:
                text=docs[alt['document']]['content'];needle=alt['text']
                if needle not in text:
                    raise ValueError((case['id'],alt['document'],needle))
                cursor=0
                while True:
                    start=text.find(needle,cursor)
                    if start<0:break
                    key=(alt['document'],start,start+len(needle))
                    if key not in seen:
                        seen.add(key)
                        expanded.append({**alt,'start':start,'end':start+len(needle)})
                    cursor=start+len(needle)
            case['evidence_groups'][gi]=expanded
    OUT.mkdir(exist_ok=True)
    corpus_text=json.dumps(list(docs.values()),ensure_ascii=False,indent=2)+'\n'
    case_text=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases)
    (OUT/'corpus.json').write_text(corpus_text)
    (OUT/'cases.jsonl').write_text(case_text)
    manifest={'version':'v3','created_date':'2026-10-19','provenance':'manually authored project-document questions plus synthetic diagnostics',
              'document_count':len(docs),'case_count':len(cases),'groups':dict(Counter(c['group'] for c in cases)),
              'categories':dict(Counter(c['category'] for c in cases)),
              'corpus_sha256':hashlib.sha256(corpus_text.encode()).hexdigest(),
              'cases_sha256':hashlib.sha256(case_text.encode()).hexdigest(),
              'limits':['Known alternatives are not exhaustive semantic relevance judgments.',
                        'Cases already used for algorithm development are not held-out.',
                        'Synthetic questions share templates; counts do not imply equivalent independent coverage.',
                        'No retrieval or reranker parameter tuning is part of this expansion.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':build()

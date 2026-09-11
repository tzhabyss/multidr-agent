# Evidence-Driven Deep Research Agent

> Runtime note: the current runtime supports `quick`, `standard`, `deep`, and
> `max` research modes. They select budgets and model tiers without adding new
> agent roles. See [model tiers and modes](docs/model-modes.md).

一个基于 Python、LangGraph、OpenAI-compatible LLM 和强类型证据账本实现的通用 Deep Research Agent。

提交材料入口：

- [技术报告](docs/submission/TECHNICAL_REPORT.md)
- [记忆系统说明](docs/memory-system.md)
- [MCP 集成说明](docs/mcp.md)
- [模型与运行模式](docs/model-modes.md)
- [密钥与安全策略](SECURITY.md)

项目只有一个统一的 `research` 模式。调用者不再选择 Deep Search 或 Deep Report；Lead Planner 根据问题判断 `straightforward`、`depth_first` 或 `breadth_first`，决定任务数量、研究方向、查询和证据下限。

## 工作流

```text
ResearchRequest
  -> normalize_scope（日期、语言、域名和用户约束）
  -> Lead Planner（生成本轮任务合同）
  -> Send(Research Worker 1..max_workers)
       -> strategy：根据任务、历史证据和已尝试查询制定策略
       -> private ReAct skill loop：Search -> Observe -> Search/Stop
       -> Fetch / Extract / deterministic admission
       -> reflection：总结已验证成果、冲突和缺口
  -> Synthesizer assessment
       -> ready     -> 生成报告
       -> need_more -> 结构化缺口返回 Planner，开始下一轮
       -> partial   -> 生成带证据边界的部分报告
  -> deterministic Citation Gate
  -> persist artifacts
```

Standard 默认最多 4 轮。第一轮建立答案空间；后续轮次根据 Synthesizer 的具体缺口定向补证，只处理关键缺口或冲突。没有固定 Retry Query，也没有 Reviewer 直接重跑 Worker。

四类 System Prompt 位于 `deepresearch/prompts.py`：

- `PLANNER_SYSTEM_PROMPT`：参考 Anthropic Research Lead 的问题分类和任务委派方式。
- `WORKER_SYSTEM_PROMPT`：参考 Research Subagent 的自适应研究和来源质量规则。
- `EXTRACTOR_SYSTEM_PROMPT`：从不可信网页中抽取带原文 Quote 的事实。
- `SYNTHESIZER_SYSTEM_PROMPT` / `REPORT_SYSTEM_PROMPT`：判断材料是否可写，并在 Claim Ledger 边界内生成报告。

LLM 负责语义充分性、冲突和不确定性判断；Python 负责工具授权、预算、URL 安全、来源映射、执行隔离和引用结构完整性，不用数字或固定维度规则裁决事实。

```text
Source -> SourceSnapshot -> Evidence -> Claim -> ClaimEvidenceLink -> Citation
```

## Skill 架构

项目从 `skills/<skill-name>/` 动态发现 Skill。`SKILL.md` 保存给角色使用的操作说明，`skill.json` 保存角色/Profile 授权、JSON Schema、运行入口、超时、标签和回退链。`deepresearch/skill_loader.py` 负责发现与绑定，`deepresearch/skill_registry.py` 负责 fail-closed 授权和执行。

Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `anysearch-search`，再合并去重候选。正文必须再经增强 Safe Fetch 和证据抽取门禁，搜索摘要不能直接成为证据。两种 Profile 的抓取都支持重试、PDF、可读页面回退，并优先尝试 Semantic Scholar、OpenAlex、arXiv、PubMed/PMC、bioRxiv/medRxiv、Crossref 和 Unpaywall；无 DOI 的论文 URL 也可根据搜索结果标题强匹配 OpenAlex 开放版本。全文不可访问时可降级为明确标记的学术摘要/书目信息。每个并行 Worker 都创建独立、有界的 `AgentLoop`，不会共享迭代计数或对话状态。回送模型的 observation 会限制条数和字符串长度，完整结果仍保存在执行结果与 Search Trace 中，避免长摘要反复消耗上下文。

`general_research` 与 `drb_evaluation` 共享原子任务拆解、逐项覆盖、失败换检索路线、有限推导分析和输出结构保留规则。测评 Profile 额外启用 requirement matrix/ID、Benchmark URL 精确屏蔽和独立 Craft Reviewer；普通 Profile 不生成或接收这些测评字段。

## 安装

需要 Python 3.11 或更高版本。

```powershell
cd <project-directory>
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,memory,search,browser]"
scrapling install
Copy-Item .env.example .env
```

macOS（Apple Silicon 与 Intel）建议使用 Python 3.12。如果已安装
`uv`，可直接在项目目录执行：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev,memory,search,browser]'
scrapling install
cp .env.example .env
```

DeepSeek MCP 搜索使用 `npx`；macOS 上需先安装 Node.js（例如
`brew install node`）。运行时会自动将 Windows 配置中的 `npx.cmd`
映射为 `npx`，并让内置 Python MCP Server 使用当前虚拟环境的
Python，无需另外维护 macOS 配置文件。

`browser` installs Scrapling's normal `DynamicFetcher` as a bounded fallback
for JavaScript-only pages. If Playwright's Chromium download is unavailable,
set `SCRAPLING_EXECUTABLE_PATH` to an installed Chrome/Edge executable; common
Windows Chrome and Edge locations are discovered automatically. The runtime
does not use Scrapling stealth mode or challenge bypass features.

配置 `.env`：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
TAVILY_API_KEY=
```

LLM 各阶段的输出 token 上限统一配置在 `config/runtime.yaml`：

```yaml
llm_output_tokens:
  planner: 40000
  worker_strategy: 4096
  worker_reflection: 2048
  worker_agent_loop: 2048
  evidence_extraction_min: 256
  evidence_extraction_max: 8000
  synthesizer_assessment: 8192
  report_writer: 30000
  reviewer: 12000
  extraction_context: 24000
```

可通过 `--config path/to/runtime.yaml` 或环境变量
`DEEPRESEARCH_CONFIG` 选择其他配置文件。API 调用也可通过
`ResearchService(config_path=...)` 指定；配置在 Service 启动时读取一次，随后显式注入各 Agent。

Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从
`DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥
写入项目文件或 `runtime.yaml`。

### MCP 外部工具

项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的
Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在
`ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server
返回的全部工具。`search_hits` 结果会进入 Safe Fetch 和证据流水线，
`observation` 结果仅供 Worker Loop 观察。具体的 stdio、Streamable HTTP、
GitHub 和学术检索配置见 [MCP integration](docs/mcp.md)。

```powershell
python -m deepresearch doctor
```

## 运行

Planner 自动决定研究深度：

```powershell
python -m deepresearch run "脑机接口技术当前的研究进展和伦理争议" `
  --run-id bci-research
```

传入明确约束：

```powershell
python -m deepresearch run "特斯拉 2026 年的主要竞争对手和市场格局分析" `
  --date-boundary "截至 2026-07-20" `
  --required-dimension "主要竞争对手" `
  --required-dimension "区域市场格局" `
  --required-dimension "证据冲突与局限" `
  --max-workers 6 `
  --max-worker-iterations 4 `
  --max-rounds 3
```

覆盖确定性证据下限：

```powershell
python -m deepresearch run "Research question" `
  --min-queries 3 `
  --min-query-coverage 2 `
  --min-successful-extractions 3 `
  --min-effective-sources 3 `
  --min-independent-domains 3 `
  --min-evidence 6 `
  --max-source-quality-tier 3
```

只覆盖显式传入的字段；其他下限由 Planner 计划并由运行时限制在安全范围内。

提供种子 URL：

```powershell
python -m deepresearch run "分析这个来源并核验关键结论" `
  --source-url https://example.com/report `
  --block-domain private.example.com `
  --json
```

相同 `run_id` 不会静默覆盖 checkpoint：

```powershell
python -m deepresearch resume bci-research --json
```

CLI 退出码：`0` verified，`3` partial，`1` failed，`2` 配置错误，`130` 用户中断。

### 输入格式

CLI 的最小输入是一条研究问题：

```powershell
python -m deepresearch run "Python 3.13.0 的正式发布日期是什么？" --json
```

结构化输入由 `ResearchRequest` 表示，主要字段如下：

```json
{
  "question": "研究问题",
  "mode": "quick | standard | deep | max",
  "profile_id": "general_research",
  "source_urls": ["https://example.com/source"],
  "constraints": {
    "locale": "zh-CN",
    "date_boundary": "截至 2026-08-27",
    "required_dimensions": ["核心事实", "限制"]
  }
}
```

批量评测使用 JSONL，每行至少包含 `case_id` 和 `question`，示例见
[`examples/general_research.jsonl`](examples/general_research.jsonl)。

### 输出格式

默认终端输出包括终态、报告路径和审计路径；`--json` 返回机器可读结果：

```json
{
  "run_id": "example-run",
  "status": "completed_verified | completed_partial | failed",
  "artifact_dir": "artifacts/example-run",
  "output_path": "artifacts/example-run/report.md",
  "audit_report_path": "artifacts/example-run/details/audit/audit-report.md"
}
```

完整输出还包含 Plan、Task Runtime、Search/Fetch/Extraction Trace、Evidence、Claim、Citation、Token Usage 和事件流。可直接查看[两轮完整运行示例](examples/complete-run/README.md)。

## 预算

`standard` 默认全局预算在 `config/runtime.yaml` 的
两个 Profile 的 `mode_overrides.standard` 下配置：

- Token 仅记录实际用量，不作为停止条件；
- 8 个并行 Worker；
- 200 次逻辑 Search，每次同时 fanout 到两个 Provider；
- 336 次 Fetch；
- 2040 秒研究软截止，2340 秒研究硬截止；
- Standard 研究软/硬截止为 3780/4080 秒；Planner 在 craft 运行 900 秒后并发启动 default，craft 最多继续到 1200 秒；Writer 和 Reviewer 各有 1020 秒阶段槽位，在 craft 运行 500 秒后并发启动 default，craft 最多继续到 900 秒；Artifact 预留 60 秒，配置的端到端上限为 6180 秒；
- `drb_evaluation` 另执行最多 600 秒 Reviewer，`general_research` 不调用 Reviewer。

搜索以至多十二条查询为一波，并与抓取交替执行，再由 Synthesizer 决定是否存在值得返回 Planner 的具体、核心、可检索缺口。只有带定向缺口的 `need_more` 才会 replan；可选背景或笼统的“再找更好来源”不会触发空转。Fetch 使用 Worker 共享租约池；提前结束、超时或失败的 Worker 会归还未用租约，其他 Worker 可继续抓取，但全局用量不会超过 336。Worker 达到任务合同的证据下限后会提前停止。

## 来源与发布门禁

来源质量分为 Tier 1–4：

- Tier 1：官方或机构域名；
- Tier 2：已识别的第一方项目或公共知识来源；
- Tier 3：可抓取的普通公共网页；
- Tier 4：社交、视频、论坛、搜索页和用户投稿页面。

Tier 4 默认保留在来源记录中，不单独证明高置信结论。它不会被代码直接删除；Synthesizer 根据任务、来源性质、冲突和可用替代来源决定是否以部分结论或明确限定的方式采用。

运行时保留 Claim、支持语段、来源快照和引用映射用于回溯，但不再用数字逐字符匹配、固定字段维度或 Quote 子串规则裁决事实。非精确支持语段会记录为 review warning，由 Synthesizer 使用语义判断评估 `supported/partial/conflicting/uncertain`。

## 目录结构

```text
deepresearch/                 Agent、Graph、Memory、Skill 与运行时核心代码
evaluation/                   DRB 与内部评测 Runner、指标和报告生成
skills/                       角色/Profile 受限的 Skill 定义及 Schema
config/runtime.yaml           模型路由、预算、Memory、MCP 和模式配置
examples/                     JSONL 输入以及可审计完整运行记录
tests/                        单元、集成、故障、并发和安全回归
docs/memory-evaluation/       Memory 写入、检索、时序和压缩评测代码
docs/submission/              提交清单与六页技术报告源文件
output/pdf/                   最终技术报告 PDF
```

运行时生成的 `.runtime/`、`artifacts/` 和 `.test-runtime/` 已被 Git 忽略。

## Artifact

```text
artifacts/<run-id>/
  report.md
  details/
    manifest.json
    input/
    planning/
    evidence/
    output/
    traces/
    runtime/
    audit/
```

报告根目录只放 `report.md` 和 `details/`。Plan、来源、证据、Agent trace、事件、用量和审计文件全部放在 `details/` 的下一层分类目录中。

## 评测

JSONL 最小样本：

```json
{"case_id":"case-1","question":"Research question"}
```

旧数据中的 `deep_search` 和 `deep_report` 会兼容读取并统一归一为 `research`，不会触发不同运行逻辑。

```powershell
python -m evaluation doctor --dataset examples/general_research.jsonl
python -m evaluation run `
  --dataset examples/general_research.jsonl `
  --evaluation-id unified-research-smoke `
  --repetitions 1
```

内部确定性指标包括 Citation Grounding、Quote Match、Unsupported Claim、Provenance、预算合规、终态事件一致性和资源使用。项目尚未运行官方 DeepResearch Bench、BrowseComp-Plus 或 FRAMES，不能声明外部 Benchmark 分数。

## 安全边界

- `general_research` 只允许 Worker 调用 `web_search` 和 `safe_fetch`。
- 外部正文包装为 `<untrusted_content>`，不会成为工具指令。
- Safe Fetch 对初始 URL 和每次 Redirect 做 DNS/IP 检查，阻止私网、localhost、metadata 地址和内嵌凭据。
- allowed/blocked domain 在 Fetch 前执行，blocked 优先。
- 并行 State 使用稳定 ID Reducer 去重，Artifact 使用原子写入和 SHA256 Manifest。
- 来源映射保留 Claim、支持语段、Snapshot 和 URL 的结构完整性；语义充分性由 Synthesizer 判断。

## 测试

```powershell
python -m compileall -q deepresearch evaluation
python -m pytest -q
```

当前完整回归结果：`339 passed, 1 skipped, 1 warning`。Warning 来自 MCP 依赖对旧客户端函数的弃用提示，与核心功能无关。

覆盖统一模式、Planner 问题分类、并行 Worker、Worker 策略与反思、多轮 `need_more -> Planner`、显式维度覆盖、证据下限、来源去重、Tier 4 隔离、超时、预算、Citation Gate、SSRF、Artifact 哈希、SQLite Resume 和评测 Reference 隔离。

## 当前限制

- Worker 使用有界 ReAct-style Skill Loop 自主调整查询；迭代数、工具调用数、角色授权和总预算仍由 Python 强制执行，不是无限自治 Agent。
- 来源 Tier 是透明启发式，不等价于事实真伪或主题相关性 Judge。
- 来源映射证明可回溯性，不等价于完整语义事实核验。
- 已支持 Session/Turn/Run 分层记忆、自研事实 CRUD、Episode/历史片段混合向量检索与原文归档；仍没有 Web UI、SSE 服务端接口或私有知识库接入。
- Agent 最终研究报告默认输出 Markdown；仓库中的 PDF 是项目技术报告，不代表运行时已经提供通用研究报告 PDF 导出。
- RetrievalBench 中的 RRF、Null Gate 和 BGE-M3 对比仍属于实验方案；生产检索默认保持可回滚的线性混合排序。
- 当前来源 Tier 依赖透明启发式，少数政府页面仍可能被归入普通公共网页，需要继续完善域名/机构识别。

进一步阅读：

- [代码阅读指南](docs/代码阅读指南.md)
- [项目实施状态与待办](docs/项目实施状态与待办.md)
- [P1 验收清单](docs/P1验收清单.md)
- [Memory system](docs/memory-system.md)
- [EchoMind 运行时治理适配与使用](docs/EchoMind运行时治理改造.md)

## DeepResearch Bench II smoke test

The benchmark profile is orthogonal to `quick/standard/deep/max`; it does not
replace the research graph or change the default `general_research` profile.
Its policy switches are configured under `agent_profiles.drb_evaluation` in
`config/runtime.yaml`.

The evaluation profile uses `semantic_citation_gate: repair`. After the Craft
Writer returns a draft, the same Craft model receives one independent review
call and may return a complete repaired draft. Runtime code still resolves
known claim IDs and validates citations that the Writer actually uses. Both
profiles are Writer-led and evidence-assisted: the claim ledger is an optional
evidence pack, uncited Writer knowledge and analysis are retained, and missing
claim bindings do not delete paragraphs, tables, rows, or cells.
`general_research` remains on `advisory` and does not make the extra review
call. Evaluation alone adds the public requirement matrix and Reviewer. Review
decisions and the final accepted draft are stored in
`details/traces/report.json`.

Both profiles fan out to DeepSeek and AnySearch through their
`skill_fallbacks.web-search` and `skill_provider_modes.web-search: fanout`
settings. Evaluation keeps only its requirement, URL-blocking, and Reviewer
policy differences.

Run one official DRB II case in Standard mode and judge it with the medium
(`default`) model tier:

```powershell
python -m evaluation run `
  --benchmark drb2 `
  --case-id task7 `
  --mode standard `
  --profile drb_evaluation `
  --judge-tier default `
  --evaluation-id drb2-task7-smoke
```

The judge API uses `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`DEEPRESEARCH_DEFAULT_MODEL` from `.env`. Hidden rubrics are passed only to the
judge after research finishes. Results are written to `judge_results.jsonl`,
`judge_summary.json`, and `evaluation_report.md` inside the evaluation run.

The local rubric score is an engineering smoke-test score produced by the
configured medium model. It is not an official leaderboard score, whose judge
and submission protocol remain controlled by the benchmark maintainers.

## Web console

Start the local FastAPI research console:

```bash
python -m deepresearch web --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. The console uses the normal
`general_research` profile and exposes the live research DAG, structured
progress events, source and citation counts, installed Skill metadata,
session memory status, titled conversation switching and search, Markdown
export, and checkpoint resume. The desktop-style shell keeps navigation,
the inspector, and the composer fixed while the conversation scrolls
independently. It intentionally does not expose evaluation-profile controls or Skill
mutation endpoints. The default host is loopback-only; choose a different host
explicitly only when you intend to make the service reachable from another
machine.

## Interactive shell

Start a persistent, multi-turn terminal session from VS Code or PowerShell:

```powershell
python -m deepresearch shell
```

Normal text is sent to the current conversation. Slash commands are handled
locally and are never sent to the LLM. Common commands:

```text
/new [name]           create a conversation
/sessions             list conversations
/switch <session_id>  switch conversations
/history [n]          show recent messages
/status               show the latest research status
/report               print the latest report
/sources              list report sources
/memory               inspect session/long-term memory
/mode [name]          show or change quick, standard, deep, or max mode
/profile [profile_id] show or change the agent profile for subsequent turns
/progress [level]      set quiet, normal, or verbose live progress
/resume <run_id>      resume or load a checkpoint
/help                 list all commands
/quit                 exit
```

Use `--session-id` to reopen a known conversation directly. Command history
is stored in `.runtime/shell-history`; research Session memory continues to use
the configured SQLite or Redis backend.

The shell streams real Planner, parallel Worker, Agent Loop, Search, Fetch,
Extraction, Synthesizer, Writer, Artifact, and autonomous Memory events. It reports
worker allocation/completion counters and emits a truthful heartbeat after 15
seconds without a new event. Use `--progress verbose` for query/URL diagnostics
or `/progress quiet` for major phases only.

Project timestamps default to `Asia/Shanghai` and include `+08:00`; legacy UTC
session timestamps are converted when displayed. See
[progress stream and timezone](docs/progress-stream.md) for the event contract.

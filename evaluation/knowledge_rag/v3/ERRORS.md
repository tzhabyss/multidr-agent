# v3 已标证据未命中与配对变化

未命中是对已标有效证据而言，不等于所有语义上正确的片段都不存在。

真实项目题：原文混合＋重排相对原文混合检索，Hit@5 改善 16 题、下降 0 题。

## 原文混合＋重排仍未命中的项目题

### repository-001：系统证据如何逐级追溯到网页快照？

已标来源：README.md

1. `README.md`：运行时保留 Claim、支持语段、来源快照和引用映射用于回溯，但不再用数字逐字符匹配、固定字段维度或 Quote 子串规则裁决事实。非精确支持语段会记录为 review warning，由 Synthesizer 使用语义判断评估 `supported/partial/conflicting/uncertain`。
2. `README.md`：- `PLANNER_SYSTEM_PROMPT`：参考 Anthropic Research Lead 的问题分类和任务委派方式。 - `WORKER_SYSTEM_PROMPT`：参考 Research Subagent 的自适应研究和来源质量规则。 - `EXTRACTOR_SYSTEM_PROMPT`：从不可信
3. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an

### repository-013：怎样检查本地运行配置是否正常？

已标来源：README.md

1. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
2. `README.md`：DeepSeek MCP 搜索使用 `npx`；macOS 上需先安装 Node.js（例如 `brew install node`）。运行时会自动将 Windows 配置中的 `npx.cmd` 映射为 `npx`，并让内置 Python MCP Server 使用当前虚拟环境的 Python，无需另外维护 macO
3. `README.md`：可通过 `--config path/to/runtime.yaml` 或环境变量 `DEEPRESEARCH_CONFIG` 选择其他配置文件。API 调用也可通过 `ResearchService(config_path=...)` 指定；配置在 Service 启动时读取一次，随后显式注入各 Agent。

### repository-014：CLI 要恢复相同 run_id 的研究应该调用什么命令？

已标来源：README.md

1. `README.md`：相同 `run_id` 不会静默覆盖 checkpoint：
2. `README.md`：/resume <run_id>      resume or load a checkpoint /help                 list all commands /quit                 exit ```
3. `README.md`：CLI 退出码：`0` verified，`3` partial，`1` failed，`2` 配置错误，`130` 用户中断。

### repository-023：交互式命令行怎样切换对话？

已标来源：README.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `README.md`：## 工作流  ```text ResearchRequest   -> normalize_scope（日期、语言、域名和用户约束）   -> Lead Planner（生成本轮任务合同）   -> Send(Research Worker 1..max_workers)        -> strategy：根据任
3. `README.md`：可通过 `--config path/to/runtime.yaml` 或环境变量 `DEEPRESEARCH_CONFIG` 选择其他配置文件。API 调用也可通过 `ResearchService(config_path=...)` 指定；配置在 Service 启动时读取一次，随后显式注入各 Agent。

### repository-024：交互式命令历史默认保存到哪里？

已标来源：README.md

1. `README.md`：## 当前限制  - Worker 使用有界 ReAct-style Skill Loop 自主调整查询；迭代数、工具调用数、角色授权和总预算仍由 Python 强制执行，不是无限自治 Agent。 - 来源 Tier 是透明启发式，不等价于事实真伪或主题相关性 Judge。 - 来源映射证明可回溯性，不等价于完整语义
2. `README.md`：Tier 4 默认保留在来源记录中，不单独证明高置信结论。它不会被代码直接删除；Synthesizer 根据任务、来源性质、冲突和可用替代来源决定是否以部分结论或明确限定的方式采用。
3. `README.md`：## 工作流  ```text ResearchRequest   -> normalize_scope（日期、语言、域名和用户约束）   -> Lead Planner（生成本轮任务合同）   -> Send(Research Worker 1..max_workers)        -> strategy：根据任

### repository-032：默认工作上下文多大，什么时候开始持久化压缩？

已标来源：docs/memory-system.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `README.md`：- Token 仅记录实际用量，不作为停止条件； - 8 个并行 Worker； - 200 次逻辑 Search，每次同时 fanout 到两个 Provider； - 336 次 Fetch； - 2040 秒研究软截止，2340 秒研究硬截止； - Standard 研究软/硬截止为 3780/4080 秒；Pl
3. `README.md`：Standard 默认最多 4 轮。第一轮建立答案空间；后续轮次根据 Synthesizer 的具体缺口定向补证，只处理关键缺口或冲突。没有固定 Retry Query，也没有 Reviewer 直接重跑 Worker。

### repository-036：记忆检索的两个候选通道各自召回多少条？

已标来源：docs/memory-system.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `README.md`：- Token 仅记录实际用量，不作为停止条件； - 8 个并行 Worker； - 200 次逻辑 Search，每次同时 fanout 到两个 Provider； - 336 次 Fetch； - 2040 秒研究软截止，2340 秒研究硬截止； - Standard 研究软/硬截止为 3780/4080 秒；Pl
3. `README.md`：当前完整回归结果：`339 passed, 1 skipped, 1 warning`。Warning 来自 MCP 依赖对旧客户端函数的弃用提示，与核心功能无关。

### repository-037：记忆中的向量采用什么存储类型？

已标来源：docs/memory-system.md

1. `README.md`：一个基于 Python、LangGraph、OpenAI-compatible LLM 和强类型证据账本实现的通用 Deep Research Agent。
2. `README.md`：项目只有一个统一的 `research` 模式。调用者不再选择 Deep Search 或 Deep Report；Lead Planner 根据问题判断 `straightforward`、`depth_first` 或 `breadth_first`，决定任务数量、研究方向、查询和证据下限。
3. `README.md`：只覆盖显式传入的字段；其他下限由 Planner 计划并由运行时限制在安全范围内。

### repository-038：记忆编码默认用哪个模型，是多少维？

已标来源：docs/memory-system.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `README.md`：Tier 4 默认保留在来源记录中，不单独证明高置信结论。它不会被代码直接删除；Synthesizer 根据任务、来源性质、冲突和可用替代来源决定是否以部分结论或明确限定的方式采用。
3. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。

### repository-039：记忆向量模型不可用时如何标记降级结果？

已标来源：docs/memory-system.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `README.md`：当前完整回归结果：`339 passed, 1 skipped, 1 warning`。Warning 来自 MCP 依赖对旧客户端函数的弃用提示，与核心功能无关。
3. `README.md`：项目只有一个统一的 `research` 模式。调用者不再选择 Deep Search 或 Deep Report；Lead Planner 根据问题判断 `straightforward`、`depth_first` 或 `breadth_first`，决定任务数量、研究方向、查询和证据下限。

### repository-042：学术检索服务出错或者没有结果会回退到哪里？

已标来源：docs/mcp.md

1. `README.md`：### MCP 外部工具  项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的 Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在 `ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server
2. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
3. `README.md`：当前完整回归结果：`339 passed, 1 skipped, 1 warning`。Warning 来自 MCP 依赖对旧客户端函数的弃用提示，与核心功能无关。

### repository-045：外部工具如何限制角色和权限？

已标来源：docs/mcp.md

1. `README.md`：### MCP 外部工具  项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的 Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在 `ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server
2. `README.md`：## 当前限制  - Worker 使用有界 ReAct-style Skill Loop 自主调整查询；迭代数、工具调用数、角色授权和总预算仍由 Python 强制执行，不是无限自治 Agent。 - 来源 Tier 是透明启发式，不等价于事实真伪或主题相关性 Judge。 - 来源映射证明可回溯性，不等价于完整语义
3. `README.md`：## 目录结构  ```text deepresearch/                 Agent、Graph、Memory、Skill 与运行时核心代码 evaluation/                   DRB 与内部评测 Runner、指标和报告生成 skills/

### repository-046：MCP 服务器广告了很多工具时会全部开放给模型吗？

已标来源：docs/mcp.md

1. `README.md`：### MCP 外部工具  项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的 Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在 `ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server
2. `docs/mcp.md`：# MCP Integration  MCP is an external tool protocol, not a replacement for this project's Skill architecture. On service startup, only explicitly allowlisted MC
3. `docs/mcp.md`：```yaml mcp:   enabled: true   servers:     internal-search:       transport: streamable_http       url: https://mcp.example.com/mcp       headers:         Auth

### repository-048：MCP 环境变量缺失时怎么处理，密钥会写入产物吗？

已标来源：docs/mcp.md

1. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
2. `docs/mcp.md`：## Configuration  Add an `mcp` section to `config/runtime.yaml`, or use a separate runtime config with `--config`. MCP is disabled by default. The checked-in se
3. `README.md`：DeepSeek MCP 搜索使用 `npx`；macOS 上需先安装 Node.js（例如 `brew install node`）。运行时会自动将 Windows 配置中的 `npx.cmd` 映射为 `npx`，并让内置 Python MCP Server 使用当前虚拟环境的 Python，无需另外维护 macO

### repository-058：报告审核由研究模式还是 Profile 决定？

已标来源：docs/model-modes.md

1. `README.md`：项目只有一个统一的 `research` 模式。调用者不再选择 Deep Search 或 Deep Report；Lead Planner 根据问题判断 `straightforward`、`depth_first` 或 `breadth_first`，决定任务数量、研究方向、查询和证据下限。
2. `README.md`：`general_research` 与 `drb_evaluation` 共享原子任务拆解、逐项覆盖、失败换检索路线、有限推导分析和输出结构保留规则。测评 Profile 额外启用 requirement matrix/ID、Benchmark URL 精确屏蔽和独立 Craft Reviewer；普通 Profil
3. `README.md`：## 目录结构  ```text deepresearch/                 Agent、Graph、Memory、Skill 与运行时核心代码 evaluation/                   DRB 与内部评测 Runner、指标和报告生成 skills/

### repository-062：进度事件是只能解析终端文本吗？

已标来源：docs/progress-stream.md

1. `README.md`：运行时生成的 `.runtime/`、`artifacts/` 和 `.test-runtime/` 已被 Git 忽略。
2. `README.md`：完整输出还包含 Plan、Task Runtime、Search/Fetch/Extraction Trace、Evidence、Claim、Citation、Token Usage 和事件流。可直接查看[两轮完整运行示例](examples/complete-run/README.md)。
3. `README.md`：### 输出格式  默认终端输出包括终态、报告路径和审计路径；`--json` 返回机器可读结果：

### repository-063：并行 Worker 的进度计数按什么键归并？

已标来源：docs/progress-stream.md

1. `README.md`：Planner、Worker、Synthesizer 分别只收到其角色可见的 Skill。Worker 的默认搜索 Skill 是 `web-search`。日常 `general_research` 和 `drb_evaluation` 都并行 fanout 到 `deepseek-web-search` 与 `an
2. `docs/mcp.md`：        search_openalex:           skill_id: openalex-search           roles: [worker]           profiles: [general_research, drb_evaluation]           result_k
3. `docs/mcp.md`：        search_code:           skill_id: github-code-search           roles: [worker]           profiles: [general_research]           result_kind: search_hits

### repository-064：verbose 进度会输出 API 密钥、提示词或网页正文吗？

已标来源：docs/progress-stream.md

1. `README.md`：The judge API uses `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `DEEPRESEARCH_DEFAULT_MODEL` from `.env`. Hidden rubrics are passed only to the judge after research
2. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
3. `docs/mcp.md`：OpenAlex currently works through its public API without a project API key. `OPENALEX_EMAIL` is optional and identifies requests for the polite pool; `OPENALEX_A

### repository-065：心跳事件能否显示虚构的完成百分比？

已标来源：docs/progress-stream.md

1. `synthetic/accuracy-a.md`：# 评估口径  在内部封闭测试集上准确率为92%，样本量500条。
2. `README.md`：当前完整回归结果：`339 passed, 1 skipped, 1 warning`。Warning 来自 MCP 依赖对旧客户端函数的弃用提示，与核心功能无关。
3. `README.md`：内部确定性指标包括 Citation Grounding、Quote Match、Unsupported Claim、Provenance、预算合规、终态事件一致性和资源使用。项目尚未运行官方 DeepResearch Bench、BrowseComp-Plus 或 FRAMES，不能声明外部 Benchmark 分数

### repository-066：默认项目时区是什么？

已标来源：docs/progress-stream.md

1. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
2. `README.md`：### MCP 外部工具  项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的 Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在 `ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server
3. `README.md`：## 当前限制  - Worker 使用有界 ReAct-style Skill Loop 自主调整查询；迭代数、工具调用数、角色授权和总预算仍由 Python 强制执行，不是无限自治 Agent。 - 来源 Tier 是透明启发式，不等价于事实真伪或主题相关性 Judge。 - 来源映射证明可回溯性，不等价于完整语义

### repository-070：密钥已经进入提交历史后只删掉文件够吗？

已标来源：SECURITY.md

1. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
2. `README.md`：Tier 4 默认保留在来源记录中，不单独证明高置信结论。它不会被代码直接删除；Synthesizer 根据任务、来源性质、冲突和可用替代来源决定是否以部分结论或明确限定的方式采用。
3. `README.md`：### MCP 外部工具  项目可将显式白名单的 MCP Tool 动态适配成现有 Role/Profile 受限的 Skill。默认搜索 MCP 会随 Service 启动；其他研究 MCP 仍由环境开关控制。连接会在 `ResearchService` 启动时建立，并在退出时关闭。不会自动暴露 MCP Server

### repository-071：安全测试中类似凭据的字符串是真密钥吗？

已标来源：SECURITY.md

1. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
2. `README.md`：相同 `run_id` 不会静默覆盖 checkpoint：
3. `README.md`：## 安全边界  - `general_research` 只允许 Worker 调用 `web_search` 和 `safe_fetch`。 - 外部正文包装为 `<untrusted_content>`，不会成为工具指令。 - Safe Fetch 对初始 URL 和每次 Redirect 做 DNS/IP 检查

### repository-072：推送前用哪个测试检查密钥卫生？

已标来源：SECURITY.md

1. `README.md`：## 安全边界  - `general_research` 只允许 Worker 调用 `web_search` 和 `safe_fetch`。 - 外部正文包装为 `<untrusted_content>`，不会成为工具指令。 - Safe Fetch 对初始 URL 和每次 Redirect 做 DNS/IP 检查
2. `README.md`：Standard 默认并行调用 DeepSeek Search 与 AnySearch；密钥分别从 `DEEPSEEK_API_KEY` 与 `ANYSEARCH_API_KEY` 环境变量读取。不要把真实密钥 写入项目文件或 `runtime.yaml`。
3. `README.md`：运行时生成的 `.runtime/`、`artifacts/` 和 `.test-runtime/` 已被 Git 忽略。

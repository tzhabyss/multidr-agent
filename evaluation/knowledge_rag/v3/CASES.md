# 124 条 RAG 评测用例目录

来源全文与标签在运行前冻结；已标证据不一定穷尽所有合理回答。

## 真实项目文档题（72）

| ID | 问题 | 类型 | 必需证据组／有效位置数 |
|---|---|---|---:|
| repository-001 | 系统证据如何逐级追溯到网页快照？ | provenance | 1／1 |
| repository-002 | 普通研究默认会同时请求哪两个搜索服务？ | exact_identifier | 1／2 |
| repository-003 | 一轮研究发现资料不足时怎么办？ | multiple_valid_passages | 1／2 |
| repository-004 | 如何更换整个项目的运行配置文件？ | configuration | 1／3 |
| repository-005 | Worker 每次执行是否会共用同一套循环状态？ | isolation | 1／1 |
| repository-006 | 搜索引擎摘要是否可以直接作为报告证据？ | evidence_boundary | 1／2 |
| repository-007 | Skill 的说明文字和执行授权分别放在哪两个文件？ | configuration | 1／1 |
| repository-008 | Python 和大模型分别负责哪些研究判断？ | responsibility | 1／1 |
| repository-009 | 源码最低要求哪个 Python 版本？ | exact_number | 1／1 |
| repository-010 | macOS 推荐采用什么版本 Python？ | exact_number | 1／1 |
| repository-011 | Windows 的 npx.cmd 在 macOS 上怎么处理？ | cross_platform | 1／1 |
| repository-012 | 浏览器 Chromium 下载不可用时怎样指定已安装的浏览器？ | configuration | 1／1 |
| repository-013 | 怎样检查本地运行配置是否正常？ | command | 1／1 |
| repository-014 | CLI 要恢复相同 run_id 的研究应该调用什么命令？ | command | 1／1 |
| repository-015 | CLI 返回 partial 和配置错误时的退出码分别是什么？ | exact_number | 1／1 |
| repository-016 | 输入批量评测 JSONL 最少需要哪两个字段？ | data_schema | 1／2 |
| repository-017 | 哪些地址会被 Safe Fetch 的 SSRF 防护拦截？ | security | 1／1 |
| repository-018 | allowed domain 与 blocked domain 冲突时谁优先？ | security | 1／1 |
| repository-019 | 来源质量 Tier 4 包括哪些页面？ | classification | 1／1 |
| repository-020 | Tier 4 来源会直接被程序删除吗？ | evidence_boundary | 1／1 |
| repository-021 | 运行生成的报告和审计材料如何放目录？ | artifact | 1／2 |
| repository-022 | 如何启动本地研究 Web 控制台？ | command | 1／1 |
| repository-023 | 交互式命令行怎样切换对话？ | command | 1／1 |
| repository-024 | 交互式命令历史默认保存到哪里？ | artifact | 1／1 |
| repository-025 | 为什么一个会话里的多次研究不会共用 LangGraph 状态？ | isolation | 1／1 |
| repository-026 | L0 这一层记忆负责什么？ | architecture | 1／1 |
| repository-027 | L1 包含哪些本次研究状态？ | architecture | 1／1 |
| repository-028 | L3 长期记忆保存什么内容？ | architecture | 1／1 |
| repository-029 | Extractor 和 Writer 会直接查记忆数据库吗？ | permission | 1／1 |
| repository-030 | 项目记忆层是否依赖 Mem0 云端服务？ | architecture | 1／1 |
| repository-031 | MemoryPacket 会永久保存成新记忆层吗？ | lifecycle | 1／2 |
| repository-032 | 默认工作上下文多大，什么时候开始持久化压缩？ | exact_number | 1／1 |
| repository-033 | must_keep 会保护哪些用户信息？ | protected_state | 1／1 |
| repository-034 | 压缩验证或 Segment 持久化失败时会丢掉热消息吗？ | failure | 1／1 |
| repository-035 | 滚动状态是用增量更新还是每次让模型重写全部摘要？ | architecture | 1／1 |
| repository-036 | 记忆检索的两个候选通道各自召回多少条？ | retrieval | 1／1 |
| repository-037 | 记忆中的向量采用什么存储类型？ | storage | 1／1 |
| repository-038 | 记忆编码默认用哪个模型，是多少维？ | embedding | 1／1 |
| repository-039 | 记忆向量模型不可用时如何标记降级结果？ | failure | 1／1 |
| repository-040 | 共享 Redis 会话后端应配置哪些选项？ | configuration | 1／1 |
| repository-041 | 没有 SERPAPI_KEY 时学术搜索先用什么服务？ | exact_identifier | 1／1 |
| repository-042 | 学术检索服务出错或者没有结果会回退到哪里？ | failure | 1／1 |
| repository-043 | 如何安装本地PDF和Word支持？ | command | 1／1 |
| repository-044 | Is an OpenAlex project key mandatory? | paraphrase | 1／1 |
| repository-045 | 外部工具如何限制角色和权限？ | multiple_valid_passages | 1／2 |
| repository-046 | MCP 服务器广告了很多工具时会全部开放给模型吗？ | permission | 1／1 |
| repository-047 | stdio 和 streamable_http 两种传输有什么区别？ | architecture | 1／1 |
| repository-048 | MCP 环境变量缺失时怎么处理，密钥会写入产物吗？ | security | 1／1 |
| repository-049 | MCP observation 类型的结果会直接变成证据吗？ | evidence_boundary | 1／1 |
| repository-050 | 可选 MCP 服务不可用时，本地 Skill 还能工作吗？ | failure | 1／1 |
| repository-051 | 关键 MCP 服务连接失败时怎样强制启动失败？ | configuration | 1／1 |
| repository-052 | 每次调用 MCP 工具是否都会重新启动服务进程？ | lifecycle | 1／1 |
| repository-053 | Which mode has eight workers and six rounds? | table | 1／1 |
| repository-054 | How can I change mode within an interactive session? | command | 1／1 |
| repository-055 | Does choosing a stronger model grant additional tool permissions? | permission | 1／1 |
| repository-056 | What does model_tiers configure? | configuration | 1／1 |
| repository-057 | research_modes 配置会控制哪些内容？ | configuration | 1／1 |
| repository-058 | 报告审核由研究模式还是 Profile 决定？ | policy | 1／1 |
| repository-059 | Reviewer 能联网或自己创建来源吗？ | permission | 1／1 |
| repository-060 | Reviewer 出错时会丢弃 Writer 草稿吗？ | failure | 1／1 |
| repository-061 | run_stream 和 resume_stream 的事件契约与 chat_stream 一样吗？ | streaming | 1／1 |
| repository-062 | 进度事件是只能解析终端文本吗？ | streaming | 1／1 |
| repository-063 | 并行 Worker 的进度计数按什么键归并？ | isolation | 1／1 |
| repository-064 | verbose 进度会输出 API 密钥、提示词或网页正文吗？ | security | 1／1 |
| repository-065 | 心跳事件能否显示虚构的完成百分比？ | streaming | 1／1 |
| repository-066 | 默认项目时区是什么？ | timezone | 1／2 |
| repository-067 | 旧 UTC 会话时间戳会被直接重写吗？ | timezone | 1／1 |
| repository-068 | 有哪些 Agent Loop 停止原因？ | lifecycle | 1／1 |
| repository-069 | 真实 API 密钥可以提交到 Git 吗？ | security | 1／1 |
| repository-070 | 密钥已经进入提交历史后只删掉文件够吗？ | security | 1／1 |
| repository-071 | 安全测试中类似凭据的字符串是真密钥吗？ | security | 1／1 |
| repository-072 | 推送前用哪个测试检查密钥卫生？ | command | 1／1 |

## 合成诊断题（32）

| ID | 问题 | 类型 | 必需证据组／有效位置数 |
|---|---|---|---:|
| synthetic-001 | 星河公司2023年研发投入是多少？ | company_year | 1／1 |
| synthetic-002 | 星河公司2024年研发投入是多少？ | company_year | 1／1 |
| synthetic-003 | 星河公司2024年研发投入的币种与金额单位是什么？ | units | 1／1 |
| synthetic-004 | 远山公司2023年研发投入是多少？ | company_year | 1／1 |
| synthetic-005 | 远山公司2024年研发投入是多少？ | company_year | 1／1 |
| synthetic-006 | 远山公司2024年研发投入的币种与金额单位是什么？ | units | 1／1 |
| synthetic-007 | 海川公司2023年研发投入是多少？ | company_year | 1／1 |
| synthetic-008 | 海川公司2024年研发投入是多少？ | company_year | 1／1 |
| synthetic-009 | 海川公司2024年研发投入的币种与金额单位是什么？ | units | 1／1 |
| synthetic-010 | 晨光公司2023年研发投入是多少？ | company_year | 1／1 |
| synthetic-011 | 晨光公司2024年研发投入是多少？ | company_year | 1／1 |
| synthetic-012 | 晨光公司2024年研发投入的币种与金额单位是什么？ | units | 1／1 |
| synthetic-013 | 微光服务标准档多少钱？ | price | 1／1 |
| synthetic-014 | 微光服务的价格按什么单位计算？ | units | 1／1 |
| synthetic-015 | 流星服务标准档多少钱？ | price | 1／1 |
| synthetic-016 | 流星服务的价格按什么单位计算？ | units | 1／1 |
| synthetic-017 | 云帆服务标准档多少钱？ | price | 1／1 |
| synthetic-018 | 云帆服务的价格按什么单位计算？ | units | 1／1 |
| synthetic-019 | 青禾服务标准档多少钱？ | price | 1／1 |
| synthetic-020 | 青禾服务的价格按什么单位计算？ | units | 1／1 |
| synthetic-021 | 北辰服务快速模式P95延迟是多少？ | table | 1／1 |
| synthetic-022 | 北辰服务深度模式P95延迟是多少？ | table | 1／1 |
| synthetic-023 | 北辰服务测试表的延迟单位与统计口径是什么？ | table_context | 1／1 |
| synthetic-024 | 南岭服务快速模式P95延迟是多少？ | table | 1／1 |
| synthetic-025 | 南岭服务深度模式P95延迟是多少？ | table | 1／1 |
| synthetic-026 | 南岭服务测试表的延迟单位与统计口径是什么？ | table_context | 1／1 |
| synthetic-027 | 青木系统在内部测试集上的准确率是多少？ | scope_disambiguation | 1／1 |
| synthetic-028 | 青木系统在外部测试集上的准确率是多少？ | scope_disambiguation | 1／1 |
| synthetic-029 | 东篱服务当前单请求超时与每日配额是多少？ | latest_version | 1／1 |
| synthetic-030 | 回到旧版快照已入库时，东篱服务限额是多少？ | historical_capture | 1／1 |
| synthetic-031 | 西溪服务当前单请求超时与每日配额是多少？ | latest_version | 1／1 |
| synthetic-032 | 回到旧版快照已入库时，西溪服务限额是多少？ | historical_capture | 1／1 |

## 多来源任务（8）

| ID | 问题 | 类型 | 必需证据组／有效位置数 |
|---|---|---|---:|
| multi-001 | 比较星河与远山2024年的研发投入。 | multi_source | 2／2 |
| multi-002 | 海川2023和2024年的研发投入分别是多少？ | multi_source | 2／2 |
| multi-003 | 流星和云帆的计费单位是否相同？ | multi_source | 2／2 |
| multi-004 | 核对星河研发投入、海川研发投入与流星计费单位。 | multi_source | 3／3 |
| multi-005 | 比较北辰和南岭服务快速模式延迟。 | multi_source | 2／2 |
| multi-006 | 青木内外部测试准确率为什么不能不加条件直接比较？ | conflicting_scopes | 2／2 |
| multi-007 | 研究工具的外部接入权限与模型档位权限如何控制？ | multi_source | 2／2 |
| multi-008 | 说明项目默认时区以及记忆向量模型。 | multi_source | 2／2 |

## 无答案诊断（12）

| ID | 问题 | 类型 | 必需证据组／有效位置数 |
|---|---|---|---:|
| negative-001 | 星河公司2025年的研发投入是多少？ | unanswerable | 0／0 |
| negative-002 | 远山公司的首席执行官是谁？ | unanswerable | 0／0 |
| negative-003 | 晨光公司的2024年净利润是多少？ | unanswerable | 0／0 |
| negative-004 | 微光服务的免费试用优惠码是什么？ | unanswerable | 0／0 |
| negative-005 | 青禾服务有没有学生折扣？ | unanswerable | 0／0 |
| negative-006 | 北辰服务在双机环境下的P99延迟是多少？ | unanswerable | 0／0 |
| negative-007 | 青木系统在德语数据集上的准确率是多少？ | unanswerable | 0／0 |
| negative-008 | 东篱服务的客服电话是什么？ | unanswerable | 0／0 |
| negative-009 | 西溪服务最近一次宕机是什么时候？ | unanswerable | 0／0 |
| negative-010 | 这个项目在官方BrowseComp排行榜的得分是多少？ | unanswerable | 0／0 |
| negative-011 | 项目使用的生产数据库管理员密码是什么？ | unanswerable | 0／0 |
| negative-012 | 论文资料库中AGI实现日期是哪一天？ | unanswerable | 0／0 |

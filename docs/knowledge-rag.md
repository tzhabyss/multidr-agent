# 研究资料 RAG：Milvus 单库

更新：2026-10-26。研究资料使用 **Milvus 单库**；本机已启用RAG及Cross-Encoder精排。
原文、切片、来源、版本与入库作业都保存在 Milvus，不需要运行 MongoDB。
Memory 的 SQLite/FTS5/E5 是另一模块，本次不改变。

当前运行配置为固定revision的GTE-small-zh、冻结切片背景说明、quota_dense候选保留与BGE-reranker-v2-m3原文精排，设备固定CPU。已导入真实项目文档；合成题材料只放隔离评测集合。自动保存抓取资料仍由persist_fetched_sources独立控制，本次未随RAG开关一起启用。

## 数据与检索流程

```text
读取文件 → 文件字节 SHA-256 判重 → 解析正文 → 结构切片
→ Milvus 保存原文/位置/作业 → 批量编码 → 写入并回读校验向量
→ 发布版本 → BM25＋向量召回 → RRF＋配额 → 可选 Cross-Encoder
→ 回填原文 → Worker 前置上下文 → 搜集与证据抽取
```

- `rag_records_<namespace hash>`：原文快照、切片跨度、文档 head、作业、上传哈希映射。
- `rag_vectors_<identity hash>`：切片原文、起止位置、快照 ID、作用域、状态和向量。
- 两种集合属于同一个 Milvus 服务。原始 PDF/DOCX 字节不保证归档；保存的是解析后的完整正文。
- 兼容固定的 Milvus/pymilvus 2.5.x，用 VARCHAR 存文本。超过单字段字节上限的正文及记录按页写入，校验全部页后替换小 manifest；保留旧页供正在读取的请求完成。没有直接采用 3.0 的 TEXT 字段。
- 2.5 的记录集合仍需一个向量字段，使用固定占位向量；该集合只做主键读取和条件查询，不参与语义召回。

结构切片保持原实现：标题/条款/段落边界，长块约600字符，长表按行切分并附加表头。
`start/end` 是解析正文的 Python 字符位置；PDF 页码来自解析元数据，不是 PDF 字节位置。

检索算法通过 `DocumentRetrievalMixin` 与旧后端共享：应用层 BM25 Top20、向量 Top40、
RRF k=60、固定候选预算与可选配额、Cross-Encoder 重排及失败回退。BM25 没有新增数据库，
也没有在这次迁移中改成 Milvus 服务端 BM25。仍按当前有效版本加载原文计算，适合目前的小资料库。
标题/章节增强、冻结的切片背景说明、相邻原文精排、查询改写均保留可选配置。
背景说明只增强索引，最终引用仍取原文及精确跨度。

## 版本发布与恢复的实际保证

Milvus 不承担 MongoDB 条件更新租约的等价替代。本实现采用 **同一台机器上的写入进程锁**：
所有写入者必须使用同一个 Milvus URI、namespace 与共享的 `milvus_lock_dir`；常见 localhost/127.0.0.1 别名会规范化。
不同主机、不同锁目录或绕过本代码的写入不在并发保护范围内。不是分布式锁或多记录事务。

1. 首先持久化 pending 作业与完整原文快照；切片 ID 由快照 ID 与序号确定。
2. 批量编码、检查模型可用性、维数、有限值与非零向量；写入 Milvus，回读 ID/向量/原文跨度。
3. 新快照 ready 后替换该文档的小 head 记录，作为应用层发布边界。
4. 完成发布记录与旧版本归档。向量的 active 字段不单独决定可见性，检索必须同时满足 head 允许的快照 ID。
5. 发布前失败：删除本次候选向量，保留失败作业和原文用于重试，旧版本继续服务。
6. 发布确认丢失：先读取 head；已提交则补做收尾，不能误删新版本。head 无法读取时停止清理，留给恢复。
7. `reconcile` 可恢复未完成作业；真实进程退出后操作系统释放文件锁。新写入先修复已提交但尚未收尾的版本。
8. `reindex` 创建独立索引代次，完整后更新该快照的索引引用；失败保留旧索引和原文。

`as_of` 按资料入库发布时间选择历史快照，不等于业务上的生效日期。
旧文件重复上传可能返回 archived 快照，不自动把历史版本重新发布。
默认 document_id 是 URI；可显式指定同一逻辑文档，不做未经验证的自动语义版本识别。

限制：旧原文页和已替换的成功索引暂时保留，没有自动容量回收/压缩策略；
同机写入串行、没有跨主机多写者保证；没有企业鉴权或大规模 ANN 压测结论。
Milvus Lite 使用 FLAT/COSINE，不能代表 Standalone 的 ANN 吞吐。

## Worker 接入保持不变

RAG 在 Worker 制定搜集策略之前，按任务召回原文作为受预算约束的前置上下文。
抓取后也可选择正文窗口。命中不是已经证实的 Claim，高时效信息仍需联网核验。

`enabled=true` 且 `persist_fetched_sources=true` 时，可将成功抓取的原始 HTTP 正文保存到
按可信 user_id/project_id 派生的 research 集合；下次任务可与 library 联合查找。
默认不保存草稿、搜索摘要或模型生成答案。相同 URL/正文去重并更新观测元数据，
过期抓取不会覆盖新观测；A→B→A 会新建观测快照。
来源日期和需要联网核验的标记保留。初始化、入库或检索失败时研究继续并记录缺口。
这是本地应用作用域隔离，不替代服务端鉴权。

## 部署与配置

```bash
pip install -e '.[knowledge]'
docker compose -f deploy/knowledge.compose.yaml up -d
```

Compose 已移除 MongoDB。etcd/MinIO 是 Milvus 自身部署依赖；“单库”指应用只连接 Milvus，
不代表 Standalone 只有一个容器。无需容器的同机开发可安装 `.[knowledge-local]` 并使用 Lite 文件。

```yaml
knowledge:
  enabled: true
  backend: milvus
  milvus_uri: http://127.0.0.1:19530
  # 本地 Lite 可使用 .runtime/knowledge-single.milvus.db
  milvus_namespace: deepresearch_knowledge
  milvus_lock_dir: .runtime/knowledge-locks
  milvus_token_env: MILVUS_TOKEN
  embedding_model: thenlper/gte-small-zh
  embedding_revision: af7bd46fbb00b3a6963c8dd7f1786ddfbfbe973a
  embedding_device: cpu
  document_context_path: evaluation/knowledge_rag/document_context/contexts.json
  candidate_policy: quota_dense
  reranker_model: BAAI/bge-reranker-v2-m3
  reranker_revision: 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
  reranker_device: cpu
  reranker_max_length: 512
  reranker_timeout_seconds: 30
  timeout_seconds: 90
  top_k: 3
  max_passages_per_source: 2
  precontext_max_chars: 6000
```

以上是当前本机启用的关键配置；完整字段以config/runtime.yaml为准。GTE与BGE是本地模型，背景说明复用既有缓存，不在普通检索时调用付费LLM生成。此前默认关闭的描述属于启用前状态。
背景说明缓存属于本地产物，不随Git仓库发布。新克隆缺少该文件时，应用会记录警告并使用原文索引；
已有但损坏的缓存仍会报错。需要背景增强时自行生成缓存，或将`document_context_path`设为`null`关闭。
背景增强消融实验必须具备有效缓存，缺失时直接停止，避免把原文索引误标为增强实验。
`--db` 仅用于显式选择的 SQLite 兼容后端；Milvus 使用 `milvus_uri`。

```bash
python -m deepresearch knowledge ingest ./report.pdf --document-id report
python -m deepresearch knowledge search '研究问题'
python -m deepresearch knowledge jobs
python -m deepresearch knowledge reconcile
python -m deepresearch knowledge retry JOB_ID
python -m deepresearch knowledge reindex
```

## 从旧双库迁移

保留 `mongo_milvus` 为显式旧后端，不静默改写旧配置，不删除旧数据。
MongoDB 不在默认依赖里，只有读取旧库时才安装：

```bash
pip install -e '.[knowledge,knowledge-legacy]'
python -m deepresearch knowledge migrate \
  --config ./single-milvus.yaml --source-config ./legacy.yaml --collection library
```

两份配置必须为完整项目配置，分别选择 `milvus` 与 `mongo_milvus`。
迁移前暂停旧库写入，目标 namespace/集合不得含独立研究版本。
迁移逐个逻辑集合执行；research: 作用域集合也要分别传入 `--collection`，不会默认扫入他人的作用域。
保留 snapshot ID、chunk ID、原文跨度、历史发布时间、来源元数据与背景说明，重新生成向量。
中断后可重跑，已发布快照校验后跳过。旧库不删除；源 head 变化会报错。

## 验证与历史结果

- 真实单库存储、故障恢复、Worker 测试：`pytest tests/test_knowledge_milvus.py -q`。
- 显式旧库迁移测试：给 `RAG_MIGRATION_MONGO_URI` 指定临时 Mongo，再运行该模块；不触及业务 Mongo。
- 冻结124题检索复测：`HF_HUB_OFFLINE=1 python -m evaluation.knowledge_rag.verify_single_milvus`。
  固定GTE模型revision、BGE v2-m3 revision、背景缓存、切片、配额与重排预算；输出逐题产物，
  与历史双库 bge_original 的有序原文跨度比较。使用开发题，不能当独立测试集或回答准确率。
- 原飞书方案、双库技术说明及历史实验保存在 [旧版说明](knowledge-rag-legacy.md)，不能当作单库当前实现保证。
- 上一份双库学习PDF属于历史架构材料；本页是单库实现的当前说明。

## 本机启用验证

健康接口`/api/health`增加knowledge状态，区分enabled、initialized、reranker_loaded及完成的精排调用次数。开启配置不等于模型已经执行；本次重启前先导入资料，实际服务启动时完成模型预热与Worker前置检索验证。参数依据与分阶段效果见[evaluation/knowledge_rag/enabled_stages/REPORT.md](../evaluation/knowledge_rag/enabled_stages/REPORT.md)。

## PDF 与图片解析（2026-10-28）

默认 `parser: auto`：PDF **逐页**提取文字层，文字缺失/乱码、含大幅图片的页面调用本地OCR，
混合PDF不再整份只按“文字PDF”处理。干净的短标题页不因为字数少就强制OCR；空白页保留页位。
较可靠的数字文字层保留，OCR补充未重复的文本；损坏的文字层改用OCR输出。
图片覆盖率和乱码检测是启发式路由，不保证发现所有小截图、嵌套图或错误字体映射。

- macOS：自动选Apple Vision，经本机Swift编译的小工具调用系统框架，需要Xcode Command Line Tools。
  首次编译计入OCR时限，后续复用 `.runtime/ocr/` 下按源码哈希缓存的可执行文件。
- 其他平台：自动选Tesseract。需自行安装 `tesseract`、中文/英文语言数据，PDF还需Poppler的 `pdftoppm`。
  本机验证了Vision，Tesseract仅做了输出解析单元测试，未做端到端质量验证。
- 新增独立 `.png/.jpg/.jpeg/.bmp/.tif/.tiff` 图片文字识别；多帧图片明确拒绝，请先转换成PDF，避免只取第一页。
- `.md/.txt` 和DOCX标题、正文、表格解析保留；DOCX内嵌图片尚未做OCR。
- `parser: native` 关闭OCR，需要OCR的页面会报错；`mineru` 仍是可选的外部适配器，不代表本机已安装。

```yaml
knowledge:
  parser: auto
  ocr_backend: auto          # auto / vision / tesseract / disabled
  ocr_languages: zh-Hans,en-US
  ocr_timeout_seconds: 60    # 单页/单图OCR，包含首次编译
  parse_timeout_seconds: 300 # 页间检查总预算；OCR子进程受剩余预算限制
  parse_max_pages: 200
  parse_allow_partial: false
```

上传仍限制32MiB、解析后正文8MiB。原生PDF读取/文字提取不是隔离子进程，
总预算不是对恶意/卡死PDF解析器的硬中断保证。
默认任何非空页无法识别就让入库作业失败，旧版本继续可用。
只有显式开启 `parse_allow_partial` 才允许带缺页入库，元数据和Worker上下文均标记不完整。
“parse_complete”表示没有未处理的失败页，**不是OCR准确率或图像语义完整性保证**。

快照保存逐页方法、失败原因、全局字符跨度、OCR行置信度与归一化区域坐标（左上原点）。
字符跨度对应**解析文本**，OCR文字仍可能错字；原PDF/图片字节不保证归档，应保留原件。
Worker上下文会标记OCR来源及需要原件核验；OCR区域和置信度可从快照元数据追溯。
没有新增图像Embedding、视觉模型、复杂表格结构恢复或完整多栏阅读顺序算法。
网页抓取器是独立路径，本节升级针对资料库文件入库，不代表网络PDF抓取也已自动OCR。

此前入库的文件仍保留旧快照，不会自动重解析。显式执行：

```bash
python -m deepresearch knowledge ingest ./report.pdf --document-id report --reparse
```

`--reparse` 只对Milvus/旧MongoMilvus文件入库生效，跳过同字节判重，创建新快照；
同一逻辑文档应沿用原有 `document_id`（原来用默认路径时不要随意换ID）。
成功发布后旧版归档；解析或向量写入失败，旧版保持可查。`reindex`只重建向量，不重新识别原文件。

新增测试：`tests/test_knowledge_document_parser.py`。固定夹具是程序生成的中英文扫描图和两页混合PDF，
只用于解析机制与冒烟验证，不是独立OCR质量基准。真实Vision测试识别出标题、金额及命令，
PDF扫描页仍出现“为→沩”等误识别，不能将少量冒烟结果宣传为全面OCR准确率。

本次验证：解析、真实本地OCR、单Milvus入库与重解析恢复、RAG上下文、CLI和配置等相关测试
合计130通过、1项外部Mongo迁移环境测试跳过。另发现并修正Lite向量写入后立即回读偶发缺失：
最多5次额外读确认，每次等待50ms，不重复写入；持续缺失仍拒绝发布。
该测试结果说明代码路径与故障处理通过检查，不等于检索指标或OCR准确率提升。
置信度直接来自OCR引擎，未经正确率校准；低于0.8的行仅作为排查提示。

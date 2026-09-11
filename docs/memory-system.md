# Memory system

## Identity model

The v1 project namespace is fixed:

```text
user_id
└── project_id = deepresearch-default
    └── session_id
        ├── turn_id
        └── run_id
```

A session is one chat window, a turn is one user/assistant exchange, and a run
is one execution of the deep-research graph. LangGraph always uses `run_id` as
its checkpoint `thread_id` so multiple runs in one session cannot share graph
state accidentally.

## Layers

- L0: LangGraph checkpoints for pause, resume, and retries.
- L1: current-run state, plans, tasks, gaps, and private worker tool traces.
- L2: bounded hot-session messages plus a deterministic rolling state and its prompt summary.
- L2 archive: immutable raw messages plus immutable, searchable v2 conversation segments containing Abstract, Summary, structured state, exact `must_keep` spans, and raw-message references.
- L3: atomic facts and cross-session research Episodes in the project's own SQLite store.
- Source provenance: claim, supporting context, URL, snapshot, and citation links.

Source provenance is not a deterministic truth gate. The extractor records
source-linked candidate claims and the synthesizer judges semantic support,
conflicts, and uncertainty. Equivalent date formats and translations are not
rejected by numeric or substring validators in the runtime pipeline.

## Agent visibility

- Clarifier receives recent dialogue, session summary, corrections, preferences, and a small set of relevant Episodes.
- Planner receives the resolved Research Brief, selected dialogue context, relevant Episodes, prior query outcomes, and open gaps.
- A Worker receives only its task contract, task-relevant history, current source-linked claims, failures, conflicts, and its own tool history.
- Extractor, Synthesizer, and Writer never read the memory database directly.

All memory retrieval is performed by `LayeredMemory`; agents do not hold Redis
credentials and cannot write arbitrary long-term memories. There is no Mem0
SDK, Cloud API, REST service, or Mem0 credential in the runtime.

The role-specific `MemoryPacket` is an ephemeral runtime projection, not a new
memory layer. It is rebuilt for each query, carries source memory IDs, and is
discarded after the agent call.

## Compaction v2

The default working context is 100k estimated tokens. Durable session
compaction starts at 60k and uses graduated pressure boundaries at 60%, 75%,
88%, and 95%. The estimator remains conservative and tokenizer-independent.

```text
immutable raw archive
→ deterministic must_keep extraction from user messages
→ one LLM call for Abstract + Segment Summary + StateDelta
→ exact-span and schema validation
→ immutable Segment persistence
→ deterministic RollingState merge
→ prune the archived messages from the hot Session
```

`must_keep` protects user-authored corrections, constraints, decisions,
numbers, dates, citations, and unresolved questions. Assistant/tool text is
never promoted into protected user state, and detected secrets are rejected.
If LLM validation or Segment persistence fails, the hot messages are retained.

Rolling state is updated from `StateDelta`; the model no longer rewrites the
entire previous rolling summary. This avoids unbounded summary-of-summary
drift. The backwards-compatible `rolling_summary` field is rendered
deterministically from recent Segment abstracts and the current constraints,
decisions, corrections, and unresolved items.

At read time, hybrid retrieval first produces a wide candidate set. A local
query- and role-aware projector selects sentence/field units, protects exact
spans, removes duplicates, applies a per-role budget, and returns a temporary
`MemoryPacket`. Active preferences, constraints, corrections, and project
decisions are deterministically re-injected even when lexical query overlap is
weak. The durable Fact, Episode, Segment, and archive rows are never modified
by this projection.

## Retrieval and indexing

Facts, research Episodes, and compacted conversation segments use two candidate
channels: FTS5 top 50 and dense-vector top 50. The union is reranked with
lexical, semantic, confidence, recency, and reinforcement signals as applicable.
Vectors are normalized Float32 BLOBs in SQLite and exact cosine search uses a
per-scope NumPy cache. This design deliberately postpones FAISS/HNSW until an
individual scope is large enough to justify ANN complexity.

Document-level retrieval remains hybrid. The second-stage MemoryPacket
projection combines the document retrieval score with query overlap,
role-specific importance, protected state, and bounded sentence selection.

The default local model is `intfloat/multilingual-e5-small` (384 dimensions):

```powershell
python -m pip install -e ".[memory]"
```

If `sentence-transformers` or the model is unavailable, the service logs an
explicit degradation and uses a deterministic lexical hash vector. Results in
that mode are labelled `hash_vector_fallback`, never `semantic_vector`.

## Lifecycle

```text
load session
→ append user turn
→ Clarifier
→ clarify without a run, answer from context, or produce Research Brief
→ retrieve long-term Episodes using the resolved brief
→ create run and plan task-scoped memory packets
→ research and write report
→ archive the full assistant report and keep only summary + report pointer hot
→ persist and vector-index the local Episode
→ drain fact/document index outboxes
```

Embedding failure never loses canonical facts, Episodes, segments, or raw
messages; the durable index outboxes can rebuild every derived projection.
Active sessions use a seven-day sliding TTL by default. The local SQLite backend works without infrastructure;
set `memory.session_backend: redis` and `DEEPRESEARCH_REDIS_URL` for a shared
Redis deployment.

## Deep dives

- [L2 会话级记忆系统：原理、实现与面试指南](./L2会话级记忆系统面试指南.md)
- [L3 长期记忆系统：原理、实现与面试指南](./L3长期记忆系统面试指南.md)

## CLI

Start a session:

```powershell
python -m deepresearch chat "研究 ZmjjKK 是谁" --json
```

Use the returned session ID for the next turn:

```powershell
python -m deepresearch chat "把刚才的答案说短一点" --session-id session-xxx --json
```

## Artifact layout

Only the report and the `details` directory are placed at the run root:

```text
artifacts/<run-id>/
├── report.md
└── details/
    ├── manifest.json
    ├── input/
    ├── planning/
    ├── evidence/
    ├── output/
    ├── traces/
    ├── runtime/
    └── audit/
```

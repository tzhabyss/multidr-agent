# Progress stream and timezone

The service keeps its existing result-oriented APIs and also exposes streaming
variants:

```python
async for item in service.chat_stream(message, session_id=session_id):
    if isinstance(item, ProgressEvent):
        render(item)
    else:
        result = item
```

The same contract is available through `run_stream()` and `resume_stream()`.
Events are structured data rather than terminal strings, so an SSE, WebSocket,
or desktop client can render the same execution without parsing logs.

## Event coverage

Progress covers the complete user-visible lifecycle:

```text
Session / Memory
  -> Clarifier
  -> Planner (round N)
  -> parallel Worker branches
       -> Strategy
       -> Agent Loop iteration / Skill
       -> Search
       -> Fetch
       -> Extractor
       -> Reflection
  -> Synthesizer (ready / need_more / partial)
  -> optional next Planner round
  -> Writer
  -> Citation mapping
  -> Artifact persistence
  -> local Episode / fact / document index synchronization
  -> final result
```

Each event contains a sequence number, timezone-aware timestamp, elapsed time,
phase, role, status, message, visibility level, run/session/task identifiers,
stable display worker ID, and structured metrics. Parallel-worker counters are
reduced by `task_id`; event arrival order cannot double-count a worker.

Agent Loop stop reasons include `model_finished`, `tool_budget_exhausted`,
`max_iterations`, `tool_calling_unavailable`, and `no_executable_skills`.
Fetch/extraction sufficiency is a later Worker phase and is not falsely labeled
as an Agent Loop decision.

## CLI controls

The interactive shell defaults to append-only plain text for compatibility
with Windows and VS Code terminals:

```powershell
python -m deepresearch shell --progress normal --heartbeat-seconds 15
```

At runtime:

```text
/progress quiet
/progress normal
/progress verbose
```

- `quiet`: major Planner, Synthesizer, Writer, and final transitions.
- `normal`: Worker allocation/counts, Loop iterations, tools, fetch, extraction,
  artifacts, and memory sync.
- `verbose`: complete search query/URL and diagnostic metrics (never keys,
  prompts, webpage bodies, or raw tool payloads).

A heartbeat is emitted only when a real operation remains active without a new
event for the configured interval. It reports elapsed waiting time and never
invents a completion percentage.

## Timezone

`config/runtime.yaml` defaults to:

```yaml
timezone: Asia/Shanghai
```

New project timestamps therefore include `+08:00`. CLI session listings also
convert legacy UTC timestamps to the configured timezone instead of rewriting
old stored records. `DEEPRESEARCH_TIMEZONE` controls timestamp default factories
used outside a configured Service; the project default remains Asia/Shanghai.

LangGraph or provider-internal metadata may retain UTC because it is owned by
those libraries. All project-generated schema timestamps and CLI clocks use the
configured China timezone by default.

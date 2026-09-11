# Model Tiers And Research Modes

The project keeps its existing Planner, Worker, Synthesizer, Writer, Clarifier,
and memory roles. It does not add role count merely to route models.

There are two independent controls in `config/runtime.yaml`:

1. `research_modes`: controls research depth, concurrency, search/fetch budget,
   wall-clock budget, and the tier used at each stage.
2. `model_tiers`: maps the logical `lite`, `default`, and `craft` tiers to an
   OpenAI-compatible provider and deployment.

| Mode | Intended use | Default workers | Maximum rounds |
| --- | --- | ---: | ---: |
| `quick` | a small factual check | 2 | 1 |
| `standard` | normal research | 4 | 4 |
| `deep` | multi-dimensional research | 6 | 4 |
| `max` | high-value, difficult research | 8 | 6 |

The default configuration keeps backwards compatibility: when the three
`DEEPRESEARCH_*_MODEL` variables are unset, every tier uses `OPENAI_MODEL` and
the existing `OPENAI_API_KEY` / `OPENAI_BASE_URL` provider.

To configure distinct endpoints, add providers and point tiers at them:

```yaml
llm_providers:
  fast:
    api_key_env: FAST_API_KEY
    base_url_env: FAST_BASE_URL
    base_url: https://example.invalid/v1
  strong:
    api_key_env: STRONG_API_KEY
    base_url_env: STRONG_BASE_URL
    base_url: https://example.invalid/v1

model_tiers:
  lite:
    provider: fast
    model: fast-small
  default:
    provider: fast
    model: fast-reasoning
  craft:
    provider: strong
    model: strong-reasoning
```

Use a mode for one command:

```powershell
python -m deepresearch run "Research question" --mode deep
python -m deepresearch chat "Research question" --mode quick
python -m deepresearch shell --mode standard
```

Inside the shell, `/mode deep` changes the mode for subsequent turns and
`/profile drb_evaluation` changes the Profile for subsequent turns. `/current`
shows both active values, while `/status` shows the actual tier/provider/model
selected for the latest run.

Tool access is intentionally not granted by model tier. `tool_permissions`
defines executable roles, and each skill/MCP tool remains constrained by its own
role, profile, and explicit allowlist. The default permits only `worker` to
execute read-only research tools.

## Report review policy

Report review is selected by profile, not by research mode. Configure it at
`agent_profiles.<profile>.semantic_citation_gate` in `config/runtime.yaml`:

| Value | Behavior |
| --- | --- |
| `advisory` | Writer-led, evidence-assisted output. Uncited content is retained, while citations that are used must resolve correctly. No AI review. This is the `general_research` default. |
| `repair` | The same Writer-led, evidence-assisted output followed by at most one independent Craft Reviewer call. The Reviewer checks requirement coverage and citation honesty without treating the claim ledger as a knowledge boundary. This is the `drb_evaluation` default. |
| `strict` | Apply the deterministic citation validator and remove content that fails it. This remains available as an explicit opt-in. |

The Reviewer cannot browse or create sources. It receives the public question
requirements, the retrieved evidence pack, and the Writer draft. It may remove
or replace an incorrect claim binding, but it does not delete useful uncited
knowledge merely because it is absent from the pack. Its decision, issues,
token usage, accepted draft, and any error are persisted in
`details/traces/report.json`. A Reviewer failure leaves the Writer draft in
place.

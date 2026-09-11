# MCP Integration

MCP is an external tool protocol, not a replacement for this project's Skill
architecture. On service startup, only explicitly allowlisted MCP tools are
adapted into role/profile-scoped `SkillDefinition` objects. They therefore use
the same authorization, schema validation, timeout, Agent Loop tracing, and
tool-call limits as local Skills.

## Install

The project dependency list includes the Python MCP client. Reinstall project
dependencies after pulling this change:

```powershell
cd D:\ai_project1\deepresearch-agent-v1
python -m pip install -e ".[dev,search]"
```

The repository includes two read-only SageResearch-style stdio servers:

- `deepresearch.mcp_servers.paper_search`: `search_arxiv`,
  `search_google_scholar`, and the direct `search_openalex` tool. Scholar uses
  Semantic Scholar first when `SERPAPI_KEY` is absent and falls back to
  OpenAlex on errors or empty results.
- `deepresearch.mcp_servers.github_search`: `search_repositories` and
  `search_code` (uses `GITHUB_TOKEN` when available).

PDF and DOCX local-file support is optional; install it with
`pip install -e ".[documents]"`.

## Configuration

Add an `mcp` section to `config/runtime.yaml`, or use a separate runtime config
with `--config`. MCP is disabled by default. The checked-in server definitions
can be enabled with:

```powershell
$env:DEEPRESEARCH_ENABLE_RESEARCH_MCP = "1"
```

and by changing `mcp.enabled` to `true`. The bundled commands are already
configured in `config/runtime.yaml`.

OpenAlex currently works through its public API without a project API key.
`OPENALEX_EMAIL` is optional and identifies requests for the polite pool;
`OPENALEX_API_KEY` is also optional for accounts or deployments that issue one.

```yaml
mcp:
  enabled: true
  servers:
    paper-search:
      transport: stdio
      command: python
      args:
        - -m
        - deepresearch.mcp_servers.paper_search
      tools:
        search_arxiv:
          skill_id: arxiv-search
          roles: [worker]
          profiles: [general_research]
          result_kind: search_hits
          tags: [academic]
          timeout_seconds: 30

        search_openalex:
          skill_id: openalex-search
          roles: [worker]
          profiles: [general_research, drb_evaluation]
          result_kind: search_hits
          tags: [academic, openalex]
          timeout_seconds: 30

        read_arxiv_paper:
          enabled: false

    github:
      transport: stdio
      command: python
      args:
        - -m
        - deepresearch.mcp_servers.github_search
      tools:
        search_repositories:
          skill_id: github-repository-search
          roles: [worker]
          profiles: [general_research]
          result_kind: search_hits

        search_code:
          skill_id: github-code-search
          roles: [worker]
          profiles: [general_research]
          result_kind: search_hits
```

`stdio` starts a local subprocess. `streamable_http` connects to a remote MCP
endpoint instead:

```yaml
mcp:
  enabled: true
  servers:
    internal-search:
      transport: streamable_http
      url: https://mcp.example.com/mcp
      headers:
        Authorization: Bearer ${INTERNAL_MCP_TOKEN}
      tools:
        search:
          skill_id: internal-search
          roles: [worker]
          profiles: [general_research]
          result_kind: search_hits
```

`${VARIABLE}` values are read from the process environment. A missing variable
is a startup error for that server; secrets are never written to artifacts.

## Allowlist And Permissions

The `tools` mapping is mandatory. A server may advertise many capabilities, but
only names listed beneath `tools` are converted to Skills. An absent tool is not
visible to the LLM and cannot be executed.

For the first rollout, use read-only tools only:

- academic or web search;
- paper metadata/full-text reading;
- GitHub repository and code search.

Do not allow write-file, create-issue, pull-request, merge, shell, database
write, or deletion tools until they have a separate approval workflow.

`roles` and `profiles` retain the existing fail-closed behavior. The default is
`worker` and `general_research`; Planner, Clarifier, Synthesizer, and Writer do
not receive MCP tools unless you deliberately configure them.

## Result Kinds

`result_kind` controls how a remote result enters the evidence pipeline.

| Value | Use |
|---|---|
| `search_hits` | The server returns structured rows with URLs. The adapter accepts `results`/`items`/`result`/`data`, or JSON text, and maps fields such as `title`, `url`, `html_url`, `pdf_url`, `abstract`, and `description` to the local `SearchHit` contract. Those URLs proceed through Safe Fetch and evidence extraction. |
| `observation` | The raw MCP text/structured content is shown only to the Worker Agent Loop. It does not become a source candidate or publishable evidence by itself. |

Use `search_hits` only after confirming the server returns stable structured
records containing public HTTP(S) URLs. If the server returns prose, private
data, or an incompatible structure, start with `observation`.

The current Worker loop selects Skills tagged `search`. `search_hits` MCP tools
receive that tag automatically. To let an `observation` tool participate in the
loop, explicitly add `tags: [search]`; it will still remain outside the evidence
ledger unless a dedicated result adapter is added.

## Lifecycle And Failure Behavior

- Connections are opened once when `ResearchService` starts and closed when the
  service exits; a tool call does not restart its MCP server.
- An unavailable optional server is logged and skipped. Local Skills continue
  to work.
- Set `required: true` to make startup fail when a critical MCP server cannot
  connect.
- An allowlisted remote tool missing from `list_tools()` is a configuration
  error. This prevents silently using an unexpected server version.
- MCP tool calls still pass through existing JSON-schema validation, role/profile
  authorization, per-tool timeout, max tool-call budget, and Agent Loop traces.

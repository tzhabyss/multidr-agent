---
name: tavily-search
description: Search the live web with Tavily's LLM-optimized results, relevance scores, advanced depth, domain filters, and recency controls.
---

# Tavily Search

Use this skill when the task needs current web information, source discovery,
recent news, articles, public filings, or targeted corroboration. Tavily is a
search tool, not the evidence store: every selected URL must go through
`safe-fetch` and the evidence-extraction pipeline before it can support a
report claim.

## Search workflow

1. Start with one or two concise queries. Keep each query under 400 characters;
   write a search query rather than a full research prompt.
2. Observe the returned titles, URLs, snippets, providers, and relevance
   signals before deciding on the next query.
3. Break broad questions into independent sub-queries for different answer
   dimensions, organizations, document types, or counterevidence.
4. Narrow with `include_domains`, `exclude_domains`, `time_range`,
   `start_date`, `end_date`, or `country` when the task contract calls for it.
5. Prefer `advanced` depth for difficult or high-impact facts. Use `basic` for
   routine discovery, or `fast` / `ultra-fast` when latency matters more than
   exhaustive relevance.
6. Stop when the task's query coverage and source-diversity goals are met, or
   when another search is unlikely to add independent evidence.

## Parameters

- `query`: concise search text; do not repeat an already attempted query.
- `max_results`: normally 5; increase only when the task needs broader source
  coverage and the remaining search budget allows it.
- `search_depth`: `ultra-fast`, `fast`, `basic`, or `advanced`. Higher speed
  trades away some relevance; use `advanced` for precision and corroboration.
- `topic`: `general`, `news`, or `finance` when the subject fits one of them.
- `time_range`: `day`, `week`, `month`, or `year` for relative recency.
- `start_date` / `end_date`: absolute date bounds in `YYYY-MM-DD` form when
  the research question has a fixed date boundary.
- `include_domains`: comma-free JSON array of trusted domains to focus on.
- `exclude_domains`: domains to omit when they are known to be irrelevant or
  disallowed by the task.
- `country`: country bias when regional coverage is relevant.
- `chunks_per_source`: number of result chunks per source for supported search
  depths; keep it small to control observation size.

## Query patterns

```text
General discovery:       "topic" current overview
Primary source:          "topic" official report site:example.org
Recent news:             "topic" latest        (topic=news, time_range=week)
Financial research:      "company" revenue     (topic=finance)
Counterevidence:         "claim" criticism OR limitation
Date-bounded research:   "topic" report       (start_date/end_date)
```

Use different intents across queries instead of changing only one keyword.
For a complex question, search each required dimension independently and keep
at least two independent domains when the task sufficiency contract requires
it.

## Evidence and safety boundary

- Search snippets, Tavily answers, relevance scores, and model-generated
  summaries are discovery observations, not evidence.
- This integration intentionally does not expose Tavily generated answers or
  raw page content. The project fetches selected pages through `safe-fetch` so
  URL policy, redirects, snapshots, and citations remain auditable.
- Do not answer the user directly from this skill's output.
- Do not fetch a URL in this skill; hand selected URLs to `safe-fetch`, which
  enforces public-URL, redirect, and domain policy checks.
- Prefer official, institutional, first-party, and independent sources. Treat
  social, forum, video, and search-result pages as low-control sources unless
  the task explicitly requires them.
- Do not repeat attempted queries or spend the remaining budget on duplicate
  domains when new independent coverage is needed.

The project invokes Tavily through its authorized HTTP backend and role-scoped
Skill Registry. Do not assume the `tvly` CLI is installed, and do not bypass
the project's fallback chain or evidence pipeline.

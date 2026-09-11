---
name: web-search
description: Search the web through the configured profile-specific provider chain. DRB evaluation defaults to DeepSeek-only; daily research retains Tavily, DuckDuckGo, and Wikipedia.
---

# Web Search

Use as the default search action. The registry selects providers from the active profile's configured fallback chain. In `general_research`, this preserves Tavily, DuckDuckGo, and Wikipedia. In `drb_evaluation`, the default chain contains only the explicitly enabled DeepSeek MCP search. Search observations and generated summaries guide discovery only, while fetched source text remains the evidence boundary.

# Research library evaluation

This directory publishes evaluation scripts, report summaries, and the frozen
26-document, 124-case development fixture in `v3/`. The fixture contains project
documentation and synthetic diagnostics; it is not an independent held-out test.

Large per-call model responses, generated context caches, local environment
protocols, and JUnit logs are kept locally and excluded from Git. Links to those
files in historical reports refer to locally generated artifacts. Use the
commands in each report to produce new artifacts; database and model dependencies
must be installed separately. Re-running live model generation may incur cost.

The default runtime keeps the knowledge library disabled. See
[the integration guide](../../docs/knowledge-rag.md) for setup and configuration.

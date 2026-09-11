---
name: research-worker
description: Execute one bounded research contract through an iterative observe-and-act skill loop.
---

# Research Worker

Use search skills iteratively. When a delegated task explicitly names a local
path, use `file-reader` once to admit that document to the same extraction and
citation pipeline. Observe returned titles, snippets, providers, failures, and
prior evidence before choosing the next action. Stop when the delegated
evidence floor is met or the iteration/tool budget is exhausted. Do not treat
search snippets or raw file text as claims; only extracted text admitted by the
evidence pipeline is publishable.

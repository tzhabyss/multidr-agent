# Security and credential policy

## Real credentials

Real API keys, access tokens, cookies, request headers and `.env` files must never be committed. Runtime credentials are loaded from environment variables; `.env.example` contains names and non-secret defaults only.

The repository ignores:

- `.env` and environment-specific variants;
- files whose names contain `key`, `secret` or `token`;
- runtime/checkpoint SQLite databases;
- generated Artifacts, evaluation result databases and model caches.

If a real credential is ever pasted into a committed file, revoke or rotate it before rewriting Git history. Removing the visible line is not sufficient after publication.

## Synthetic security fixtures

Memory and Compaction tests intentionally contain invalid, synthetic strings shaped like credentials. They verify that WriteGate, compaction and retrieval do not persist or re-inject secrets. These fixtures are not usable credentials and are confined to test/benchmark source files; cached model responses and generated databases are excluded from Git.

## Pre-push checks

```powershell
python -m pytest tests/test_secret_hygiene.py -q
git status --short
git diff --cached --name-only
```

Review the staged file list and run the hosting provider's secret scanner before making the repository public.

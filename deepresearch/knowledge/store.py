"""SQLite fact plane + rebuildable BM25/vector candidate plane.

Offsets always refer to the immutable parsed snapshot, never to synthesized text.
A store is a trusted local workspace; callers must select its collection explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Callable

from ..memory_system.embeddings import ResilientEmbeddingProvider
from ..memory_system.retrieval import cosine


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks(text: str, size: int = 600) -> list[dict]:
    """Paragraph/heading chunks, with exact offsets and conservative hard bounds."""
    result = []
    headings: list[str] = []
    pending_heading = None
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        start, end = match.span()
        block = match.group()
        heading = re.match(r"^(#{1,6})\s+([^\n]+)", block)
        if heading:
            level = len(heading[1])
            headings = headings[:level - 1] + [heading[2]]
        elif re.match(r"^第.{1,12}[章节条]", block):
            headings = [block.splitlines()[0]]
        if heading and "\n" not in block:
            if pending_heading is None:
                pending_heading = start
            continue
        if pending_heading is not None:
            start = pending_heading
            pending_heading = None
        cursor = start
        while cursor < end:
            stop = min(cursor + size, end)
            if stop < end:
                boundary = max(text.rfind(c, cursor + size // 2, stop) for c in "。；.!?\n")
                if boundary >= 0:
                    stop = boundary + 1
            result.append({"start": cursor, "end": stop, "path": list(headings)})
            cursor = stop
    if pending_heading is not None:
        result.append({"start": pending_heading, "end": len(text), "path": list(headings)})
    return result


def features(text: str) -> list[str]:
    """Chinese bigrams + Latin terms with term frequency for actual BM25."""
    tokens = re.findall(r"[a-z0-9][a-z0-9_.-]*", text.casefold())
    for span in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(span[i:i + 2] for i in range(max(1, len(span) - 1)))
    return tokens


class KnowledgeStore:
    def __init__(self, path: str | Path, *, provider=None, reranker: Callable | None = None, contextual: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.provider = provider or ResilientEmbeddingProvider()
        self.reranker = reranker
        self.contextual = contextual
        self.lock = RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY, collection TEXT NOT NULL, uri TEXT NOT NULL,
                    title TEXT NOT NULL, content TEXT NOT NULL, hash TEXT NOT NULL,
                    created_at TEXT NOT NULL, status TEXT NOT NULL,
                    supersedes TEXT, model TEXT NOT NULL, degraded INTEGER NOT NULL,
                    metadata TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshot_scope ON snapshots(collection,uri,status);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, snapshot TEXT NOT NULL REFERENCES snapshots(id),
                    start INTEGER NOT NULL, end INTEGER NOT NULL,
                    heading TEXT NOT NULL, vector TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunk_snapshot ON chunks(snapshot);
            """)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(snapshots)")}
            if "index_context" not in columns:
                db.execute("ALTER TABLE snapshots ADD COLUMN index_context INTEGER NOT NULL DEFAULT 0")
            chunk_columns = {r["name"] for r in db.execute("PRAGMA table_info(chunks)")}
            if "raw_vector" not in chunk_columns:
                db.execute("ALTER TABLE chunks ADD COLUMN raw_vector TEXT NOT NULL DEFAULT '[]'")

    @staticmethod
    def index_text(title, heading, content):
        # Source metadata only: no generated facts enter either the index or quote.
        return "\n".join([title[:200], " > ".join(heading)[:400], content])

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, content: str, *, uri: str, title: str, collection: str = "library", metadata: dict | None = None) -> dict:
        if not content.strip() or not collection.strip():
            raise ValueError("nonempty content and collection required")
        hash_ = digest(content)
        sid = digest(json.dumps([collection, uri, hash_], ensure_ascii=False))
        # Serialize provider transitions and version publication. No partial version is visible.
        with self.lock, self.connect() as db:
            existing = db.execute("SELECT * FROM snapshots WHERE id=?", (sid,)).fetchone()
            if existing:
                return {"snapshot_id": sid, "duplicate": True, "status": existing["status"]}
            parts = chunks(content)
            vectors = self.provider.encode_documents([
                self.index_text(title, p["path"], content[p["start"]:p["end"]]) if self.contextual
                else content[p["start"]:p["end"]] for p in parts])
            raw_vectors = self.provider.encode_documents([content[p["start"]:p["end"]] for p in parts]) if self.contextual else vectors
            if len(vectors) != len(parts) or len(raw_vectors) != len(parts) or any(not v for v in [*vectors, *raw_vectors]):
                raise ValueError("incomplete embedding batch")
            now = datetime.now(timezone.utc).isoformat()
            db.execute("BEGIN IMMEDIATE")
            # Another process may have published this hash during encoding.
            existing = db.execute("SELECT status FROM snapshots WHERE id=?", (sid,)).fetchone()
            if existing:
                return {"snapshot_id": sid, "duplicate": True, "status": existing["status"]}
            old = db.execute("SELECT id FROM snapshots WHERE collection=? AND uri=? AND status='active'", (collection, uri)).fetchone()
            db.execute("INSERT INTO snapshots (id,collection,uri,title,content,hash,created_at,status,supersedes,model,degraded,metadata,index_context) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                sid, collection, uri, title, content, hash_, now, "active",
                old["id"] if old else None, self.provider.model_id, int(self.provider.degraded),
                json.dumps(metadata or {}, ensure_ascii=False), int(self.contextual)))
            for i, (p, vector) in enumerate(zip(parts, vectors, strict=True)):
                db.execute("INSERT INTO chunks (id,snapshot,start,end,heading,vector,raw_vector) VALUES (?,?,?,?,?,?,?)", (
                    f"{sid}:{i}", sid, p["start"], p["end"], json.dumps(p["path"], ensure_ascii=False), json.dumps(vector), json.dumps(raw_vectors[i])))
            db.execute("UPDATE snapshots SET status='archived' WHERE collection=? AND uri=? AND id!=? AND status='active'", (collection, uri, sid))
        return {"snapshot_id": sid, "duplicate": False, "chunks": len(parts), "degraded": self.provider.degraded}

    def search(self, query: str, *, collection: str = "library", snapshot_id: str | None = None, as_of: str | None = None, limit: int = 5, mode: str = "hybrid", rerank: bool = True) -> list[dict]:
        if not query.strip() or limit <= 0:
            return []
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError("unknown retrieval mode")
        with self.connect() as db:
            if snapshot_id:
                snapshots = db.execute("SELECT * FROM snapshots WHERE collection=? AND id=?", (collection, snapshot_id)).fetchall()
            elif as_of:
                # as_of is snapshot capture time, not document effective/publication time.
                cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                if cutoff.tzinfo is None:
                    raise ValueError("as_of requires timezone")
                cutoff = cutoff.astimezone(timezone.utc).isoformat()
                snapshots = db.execute("SELECT * FROM snapshots WHERE collection=? AND created_at<=? ORDER BY created_at DESC,id", (collection, cutoff)).fetchall()
                seen = set()
                snapshots = [s for s in snapshots if not (s["uri"] in seen or seen.add(s["uri"]))]
            else:
                snapshots = db.execute("SELECT * FROM snapshots WHERE collection=? AND status='active'", (collection,)).fetchall()
            rows = []
            for s in snapshots:
                for c in db.execute("SELECT * FROM chunks WHERE snapshot=?", (s["id"],)):
                    rows.append((s, c, s["content"][c["start"]:c["end"]]))
        if not rows:
            return []
        with self.lock:
            query_vector = self.provider.encode_query(query) if mode != "bm25" else []
            model = self.provider.model_id
        index_texts = [self.index_text(s["title"], json.loads(c["heading"]), text)
                       if self.contextual and s["index_context"] else text for s, c, text in rows]
        counts = [Counter(features(text)) for text in index_texts]
        lengths = [sum(c.values()) for c in counts]
        avg = sum(lengths) / len(rows) or 1
        qterms = set(features(query))
        df = Counter(t for c in counts for t in c if t in qterms)
        lexical = []
        semantic = []
        raw_counts = [Counter(features(text)) for _,_,text in rows]
        raw_lengths = [sum(c.values()) for c in raw_counts]
        raw_avg = sum(raw_lengths)/len(rows) or 1
        raw_df = Counter(t for c in raw_counts for t in c if t in qterms)
        raw_lexical, raw_semantic = [], []
        for i, ((s, c, _), freq, length) in enumerate(zip(rows, counts, lengths, strict=True)):
            bm25 = sum(math.log(1 + (len(rows) - df[t] + .5) / (df[t] + .5)) * freq[t] * 2.2 / (freq[t] + 1.2 * (.25 + .75 * length / avg)) for t in qterms if freq[t])
            if bm25 > 0:
                lexical.append((i, bm25))
            raw_freq = raw_counts[i]
            raw_score = sum(math.log(1+(len(rows)-raw_df[t]+.5)/(raw_df[t]+.5))*raw_freq[t]*2.2 /
                            (raw_freq[t]+1.2*(.25+.75*raw_lengths[i]/raw_avg)) for t in qterms if raw_freq[t])
            if raw_score > 0:
                raw_lexical.append((i,raw_score))
            if query_vector and not self.provider.degraded and not s["degraded"] and s["model"] == model:
                similarity = cosine(query_vector, json.loads(c["vector"] if self.contextual or not s["index_context"] else c["raw_vector"]))
                if similarity > .25:
                    semantic.append((i, similarity))
                raw_similarity = cosine(query_vector, json.loads(c["raw_vector"]))
                if raw_similarity > .25:
                    raw_semantic.append((i,raw_similarity))
        lexical.sort(key=lambda x: (-x[1], x[0]))
        semantic.sort(key=lambda x: (-x[1], x[0]))
        def deduplicate(route):
            seen, result = set(), []
            for i, score in route:
                key = (rows[i][0]["id"], digest(rows[i][2].strip()))
                if key not in seen:
                    seen.add(key)
                    result.append((i, score))
            return result

        lexical = deduplicate(lexical)
        semantic = deduplicate(semantic)
        raw_lexical = deduplicate(sorted(raw_lexical,key=lambda x:(-x[1],x[0])))
        raw_semantic = deduplicate(sorted(raw_semantic,key=lambda x:(-x[1],x[0])))
        scores: dict[int, float] = {}
        routes = ([lexical[:20]] if mode == "bm25" else [semantic[:40]] if mode == "vector" else [lexical[:20], semantic[:40]])
        if self.contextual and any(s["index_context"] for s,_,_ in rows):
            routes += ([raw_lexical[:20]] if mode=="bm25" else [raw_semantic[:40]] if mode=="vector"
                       else [raw_lexical[:20],raw_semantic[:40]])
        for route in routes:
            for rank, (i, _) in enumerate(route, 1):
                scores[i] = scores.get(i, 0) + 1 / (60 + rank)
        hits = []
        for i in sorted(scores, key=lambda i: (-scores[i], i))[:max(limit * 2, 20)]:
            s, c, text = rows[i]
            pages = json.loads(s["metadata"]).get("page_starts", [])
            page = sum(offset <= c["start"] for offset in pages) if pages else None
            hits.append({"chunk_id": c["id"], "snapshot_id": s["id"], "uri": s["uri"], "title": s["title"], "content": text,
                         "start": c["start"], "end": c["end"], "page": page, "heading": json.loads(c["heading"]), "score": scores[i],
                         "degraded": bool(s["degraded"]) or self.provider.degraded,
                         "model_mismatch": s["model"] != model, "index_context": self.contextual and bool(s["index_context"]), "rerank_status": "disabled"})
        if rerank and self.reranker and hits:
            try:
                ranked = self.reranker(query, [dict(h) for h in hits])
                if sorted(h["chunk_id"] for h in ranked) != sorted(h["chunk_id"] for h in hits):
                    raise ValueError("reranker must return a permutation")
                by_id = {h["chunk_id"]: h for h in hits}
                hits = [by_id[h["chunk_id"]] for h in ranked]
                for h in hits:
                    h["rerank_status"] = "applied"
            except Exception:
                for h in hits:
                    h["rerank_status"] = "fallback"
        return hits[:limit]

    def snapshot(self, sid: str, *, collection: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM snapshots WHERE id=? AND collection=?", (sid, collection)).fetchone()
        if row is None:
            raise ValueError("snapshot outside collection or missing")
        result = dict(row)
        if digest(result["content"]) != result["hash"]:
            raise ValueError("snapshot content hash mismatch")
        return result

    def windows(self, query: str, *, snapshot_id: str, collection: str, limit: int = 3) -> list[tuple[int, int, str, bool]]:
        snapshot = self.snapshot(snapshot_id, collection=collection)
        content = snapshot["content"]
        hits = self.search(query, snapshot_id=snapshot_id, collection=collection, limit=limit)
        ranges = sorted((max(0, h["start"] - 250), min(len(content), h["end"] + 250)) for h in hits)
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return [(a, b, content[a:b], a > 0 or b < len(content)) for a, b in merged]

    def reindex(self, *, collection: str = "library") -> dict:
        """Rebuild chunk/vector indexes without changing immutable text or versions."""
        count = 0
        with self.lock, self.connect() as db:
            snapshots = db.execute("SELECT * FROM snapshots WHERE collection=?", (collection,)).fetchall()
            for snapshot in snapshots:
                if digest(snapshot["content"]) != snapshot["hash"]:
                    raise ValueError("snapshot content hash mismatch")
                parts = chunks(snapshot["content"])
                vectors = self.provider.encode_documents([
                    self.index_text(snapshot["title"], p["path"], snapshot["content"][p["start"]:p["end"]])
                    if self.contextual else snapshot["content"][p["start"]:p["end"]] for p in parts])
                raw_vectors = self.provider.encode_documents([snapshot["content"][p["start"]:p["end"]] for p in parts]) if self.contextual else vectors
                if len(vectors) != len(parts) or len(raw_vectors) != len(parts) or any(not v for v in [*vectors,*raw_vectors]):
                    raise ValueError("incomplete embedding batch")
                db.execute("DELETE FROM chunks WHERE snapshot=?", (snapshot["id"],))
                for i, (part, vector) in enumerate(zip(parts, vectors, strict=True)):
                    db.execute("INSERT INTO chunks (id,snapshot,start,end,heading,vector,raw_vector) VALUES (?,?,?,?,?,?,?)", (
                        f"{snapshot['id']}:{i}", snapshot["id"], part["start"], part["end"],
                        json.dumps(part["path"],ensure_ascii=False), json.dumps(vector), json.dumps(raw_vectors[i])))
                db.execute("UPDATE snapshots SET model=?,degraded=?,index_context=? WHERE id=?", (
                    self.provider.model_id, int(self.provider.degraded), int(self.contextual), snapshot["id"]))
                count += 1
        return {"reindexed_snapshots": count, "model": self.provider.model_id, "degraded": self.provider.degraded}

    def ingest_file(self, path: str | Path, *, collection: str = "library") -> dict:
        path = Path(path).resolve()
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            pages = [p.extract_text() or "" for p in PdfReader(path).pages]
            content = "\n\n".join(pages)
            offsets, position = [], 0
            for page in pages:
                offsets.append(position)
                position += len(page) + 2
            metadata = {"page_starts": offsets}
        elif path.suffix.lower() in {".md", ".txt"}:
            content, metadata = path.read_text(encoding="utf-8"), {}
        else:
            raise ValueError("supported local formats: .md, .txt, .pdf (text PDFs)")
        return self.ingest(content, uri=path.as_uri(), title=path.name, collection=collection, metadata=metadata)

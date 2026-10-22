"""MongoDB facts + Milvus candidates, with durable ingestion and compensation.

A single Mongo head pointer is the publication boundary. Milvus status is a
replica, never authority. No distributed transaction is claimed. All staged IDs
are journalled before external writes; retries use new, disjoint snapshot IDs.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path

from .store import KnowledgeStore, digest, features
from .candidate_selection import POLICIES, quota_targets, select_candidates
from ..memory_system.embeddings import ResilientEmbeddingProvider


def utcnow():
    return datetime.now(timezone.utc)


def structured_chunks(text, size=600):
    """Preserve contiguous source spans; repeat split-table headers only in index text."""
    result, headings, pending = [], [], None
    blocks = list(re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text))
    for block in blocks:
        start, end = block.span()
        value = block.group()
        first = value.splitlines()[0]
        heading = re.match(r"^(#{1,6})\s+(.+)$", first)
        clause = re.match(r"^(第[零〇一二三四五六七八九十百千万\d]+[编章节条])(?:\s|[、.：:]|$)", first)
        if heading:
            headings = headings[:len(heading[1])-1] + [heading[2]]
        elif clause:
            if clause[1].endswith(('章', '节', '编')):
                headings = [first]
            else:
                headings = headings[:1] + [first]
        if heading and '\n' not in value and len(value) <= size:
            pending = start if pending is None else pending
            continue
        if pending is not None:
            start, pending = pending, None
        table_header = ''
        table_lines = value.splitlines()
        if len(table_lines) > 1 and '|' in table_lines[0] and re.match(r'^\s*\|?\s*:?-+', table_lines[1]):
            table_header = '\n'.join(table_lines[:2])
        cursor = start
        while cursor < end:
            stop = min(cursor+size, end)
            if stop < end:
                marks = '\n' if table_header else '。；.!?\n'
                boundary = max(text.rfind(c, cursor+size//2, stop) for c in marks)
                if boundary >= 0:
                    stop = boundary+1
            result.append({'start': cursor, 'end': stop, 'path': list(headings),
                           'index_prefix': table_header if table_header and cursor > block.start() else ''})
            cursor = stop
    if pending is not None:
        for cursor in range(pending, len(text), size):
            result.append({'start': cursor, 'end': min(cursor+size, len(text)),
                           'path': list(headings), 'index_prefix': ''})
    return result


def validate_vectors(vectors, count, dimensions):
    if len(vectors) != count:
        raise ValueError('incomplete embedding batch')
    for vector in vectors:
        if len(vector) != dimensions or dimensions <= 0:
            raise ValueError('embedding dimension mismatch')
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in vector):
            raise ValueError('embedding must contain finite numeric values')
        if not any(v != 0 for v in vector):
            raise ValueError('zero embedding is not ready')


class IngestionBusy(RuntimeError):
    pass


class MongoMilvusStore:
    """Trusted local collections; user/department ACLs are not inferred from query text."""
    def __init__(self, mongo_uri='mongodb://127.0.0.1:27017', *, database='deepresearch_knowledge',
                 milvus_uri='http://127.0.0.1:19530', milvus_token='', provider=None,
                 reranker=None, contextual=False, lease_seconds=120, mongo_client=None,
                 vector_client=None, parser='native', allow_degraded=False, candidate_policy='rrf',
                 document_context=None, reranker_neighbor_chars=0):
        if candidate_policy not in POLICIES:
            raise ValueError('unknown candidate policy')
        if not isinstance(reranker_neighbor_chars, int) or isinstance(reranker_neighbor_chars, bool) or not 0 <= reranker_neighbor_chars <= 300:
            raise ValueError('reranker_neighbor_chars must be an integer in 0..300')
        from pymongo import MongoClient
        from pymilvus import MilvusClient
        self.mongo = mongo_client if mongo_client is not None else MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self.db = self.mongo[database]
        self.vectors = vector_client if vector_client is not None else MilvusClient(uri=milvus_uri, token=milvus_token, timeout=10)
        self.provider = provider or ResilientEmbeddingProvider(fallback_enabled=False)
        self.reranker, self.contextual, self.parser = reranker, contextual, parser
        self.candidate_policy = candidate_policy
        self.document_context = document_context
        self.reranker_neighbor_chars = reranker_neighbor_chars
        self.allow_degraded = allow_degraded
        self.lease_seconds = lease_seconds
        self.lock = threading.RLock()
        self.db.heads.create_index([('collection', 1)])
        self.db.versions.create_index([('collection', 1), ('document_id', 1), ('source_hash', 1)])
        self.db.versions.create_index([('collection', 1), ('published_at', 1)])
        self.db.chunks.create_index([('snapshot', 1)])
        self.db.jobs.create_index([('status', 1), ('updated_at', 1)])
        self.db.uploads.create_index([('collection', 1), ('file_hash', 1)])

    def close(self):
        self.vectors.close()
        self.mongo.close()

    def _stage(self, stage, job):
        """Fault injection seam, called AFTER real writes at each boundary."""

    def _head_id(self, collection, document_id):
        return digest(json.dumps([collection, document_id], ensure_ascii=False))

    @contextmanager
    def _lease(self, collection, document_id):
        from pymongo import ReturnDocument
        from pymongo.errors import DuplicateKeyError
        hid, token = self._head_id(collection, document_id), uuid.uuid4().hex
        try:
            self.db.heads.update_one({'_id': hid}, {'$setOnInsert': {
                'collection': collection, 'document_id': document_id, 'active': None}}, upsert=True)
        except DuplicateKeyError:
            pass
        head = self.db.heads.find_one_and_update({'_id': hid, '$or': [
            {'lease_until': {'$lte': utcnow()}}, {'lease_until': {'$exists': False}}]},
            {'$set': {'lease_token': token, 'lease_until': utcnow()+timedelta(seconds=self.lease_seconds)}},
            return_document=ReturnDocument.AFTER)
        if not head:
            raise IngestionBusy('document ingestion in progress; retry after the lease is released')
        stop, lost = threading.Event(), threading.Event()
        def heartbeat():
            while not stop.wait(max(.1, self.lease_seconds/3)):
                try:
                    result = self.db.heads.update_one({'_id': hid, 'lease_token': token},
                        {'$set': {'lease_until': utcnow()+timedelta(seconds=self.lease_seconds)}})
                    if not result.matched_count:
                        lost.set(); return
                except Exception:
                    lost.set(); return
        thread = threading.Thread(target=heartbeat, daemon=True, name='knowledge-ingestion-lease')
        thread.start()
        try:
            # Repair a committed head whose writer died before recording publication.
            committed = {c['snapshot'] for c in head.get('commits', [])} | {head.get('active')}
            for previous_job in self.db.jobs.find({'head_id': hid, 'status': 'pending',
                                                  'kind': {'$ne': 'index_rebuild'},
                                                  'snapshot_id': {'$in': list(committed-{None})}}):
                self._finish_publication(previous_job)
            yield {'head_id': hid, 'token': token, 'lost': lost}
        finally:
            stop.set(); thread.join(timeout=2)
            self.db.heads.update_one({'_id': hid, 'lease_token': token},
                {'$unset': {'lease_token': '', 'lease_until': ''}})

    def _job_update(self, jid, **values):
        self.db.jobs.update_one({'_id': jid}, {'$set': {**values, 'updated_at': utcnow()}})

    def _duplicate(self, collection, document_id, source_hash, title, metadata):
        version = self.db.versions.find_one({'collection': collection, 'document_id': document_id,
            'source_hash': source_hash, 'published_at': {'$exists': True}}, sort=[('published_at', -1)])
        if version:
            # Metadata correction doesn't mutate the immutable text or vectors.
            updates = {'title': title}
            if metadata:
                merged = dict(version['metadata']); merged.update(metadata); updates['metadata'] = merged
            self.db.versions.update_one({'_id': version['_id']}, {'$set': updates})
            return {'snapshot_id': version['_id'], 'duplicate': True,
                    'status': self.snapshot(version['_id'], collection=collection)['status']}
        return None

    def ingest(self, content, *, uri, title, collection='library', metadata=None, document_id=None):
        if not content.strip() or not collection.strip():
            raise ValueError('nonempty content and collection required')
        document_id = document_id or uri
        with self._lease(collection, document_id) as lease:
            found = self._duplicate(collection, document_id, digest(content), title, metadata)
            if found:
                return found
            job = self._new_job(lease, collection, document_id, uri, title,
                                source_hash=digest(content), source_kind='parsed_text', metadata=metadata)
            return self._execute(job, content, lease)

    def _new_job(self, lease, collection, document_id, uri, title, **extra):
        jid, sid = uuid.uuid4().hex, uuid.uuid4().hex
        job = {'_id': jid, 'snapshot_id': sid, 'head_id': lease['head_id'], 'lease_token': lease['token'],
               'collection': collection, 'document_id': document_id, 'uri': uri, 'title': title,
               'status': 'pending', 'phase': 'created', 'created_at': utcnow(), 'updated_at': utcnow(),
               'mongo_ids': [], 'milvus_ids': [], **extra}
        self.db.jobs.insert_one(job)
        return job

    def _vector_collection_name(self, model, dimension, generation, context_fingerprint=None):
        identity = [model, dimension, self.contextual, 'structured-v1', generation]
        if context_fingerprint:
            identity.append(context_fingerprint)
        return 'rag_' + digest(json.dumps(identity))[:24]

    def _ensure_vector_collection(self, model, dimension, generation='shared', context_fingerprint=None):
        from pymilvus import DataType
        name = self._vector_collection_name(model, dimension, generation, context_fingerprint)
        if not self.vectors.has_collection(collection_name=name):
            schema = self.vectors.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=80)
            for field in ['snapshot', 'scope']:
                schema.add_field(field, DataType.VARCHAR, max_length=64)
            schema.add_field('status', DataType.VARCHAR, max_length=16)
            schema.add_field('vector', DataType.FLOAT_VECTOR, dim=dimension)
            params = self.vectors.prepare_index_params()
            params.add_index(field_name='vector', index_type='FLAT', metric_type='COSINE')
            try:
                self.vectors.create_collection(collection_name=name, schema=schema, index_params=params,
                                               consistency_level='Strong')
            except Exception:
                if not self.vectors.has_collection(collection_name=name):
                    raise
        return name

    def _execute(self, job, content, lease):
        sid, jid = job['snapshot_id'], job['_id']
        try:
            if len(content.encode('utf-8')) > 8*1024*1024:
                raise ValueError('parsed text exceeds local 8 MiB document limit')
            parts = structured_chunks(content)
            if not parts or len(parts) > 10000:
                raise ValueError('document must contain 1..10000 chunks')
            ids = [f'{sid}:{i}' for i in range(len(parts))]
            # Write-ahead intent covers even timeout/partial-success acknowledgements.
            job.update(mongo_ids=ids, milvus_ids=ids, phase='parsed')
            self._job_update(jid, mongo_ids=ids, milvus_ids=ids, phase='parsed')
            version = {'_id': sid, 'id': sid, 'collection': job['collection'], 'document_id': job['document_id'],
                       'head_id': job['head_id'], 'uri': job['uri'], 'title': job['title'], 'content': content,
                       'hash': digest(content), 'source_hash': job['source_hash'], 'state': 'pending',
                       'created_at': utcnow().isoformat(), 'metadata': job.get('metadata') or {},
                       'expected_chunks': len(parts), 'job_id': jid, 'supersedes': None}
            version.update(self._index_projection(content, parts))
            self.db.versions.insert_one(version)
            self._stage('snapshot_written', job)
            rows = [{'_id': cid, 'snapshot': sid, 'start': part['start'], 'end': part['end'],
                     'heading': part['path'], 'index_prefix': part['index_prefix']} for cid, part in zip(ids, parts, strict=True)]
            self.db.chunks.insert_many(rows, ordered=True)
            self._stage('mongo_chunks_written', job)
            with self.lock:
                vector_texts = [self._index_text(version, row) for row in rows]
                vectors = self.provider.encode_documents(vector_texts)
                validate_vectors(vectors, len(rows), self.provider.dimensions)
                if self.provider.degraded and not self.allow_degraded:
                    raise ValueError('semantic embedding unavailable; do not publish a hash index')
                model, dimensions = self.provider.model_id, self.provider.dimensions
            name = self._ensure_vector_collection(model, dimensions,
                context_fingerprint=version.get('context_manifest_sha256'))
            job['vector_collection'] = name
            self._job_update(jid, vector_collection=name, phase='writing_vectors')
            data = [{'id': cid, 'snapshot': sid, 'scope': digest(job['collection']),
                     'status': 'pending', 'vector': [float(v) for v in vector]} for cid, vector in zip(ids, vectors, strict=True)]
            for offset in range(0, len(data), 128):
                self.vectors.upsert(collection_name=name, data=data[offset:offset+128])
                self._stage('vector_batch_written', job)
            self.vectors.flush(collection_name=name)
            # Actual round-trip reads, not just successful return from an insert call.
            for offset in range(0, len(ids), 128):
                actual = self.vectors.get(collection_name=name, ids=ids[offset:offset+128],
                                          output_fields=['id', 'vector'], consistency_level='Strong')
                if {r['id'] for r in actual} != set(ids[offset:offset+128]):
                    raise ValueError('Milvus index is incomplete')
                validate_vectors([r['vector'] for r in actual], len(actual), dimensions)
            if self.db.chunks.count_documents({'snapshot': sid}) != len(ids):
                raise ValueError('Mongo chunk set is incomplete')
            self.db.versions.update_one({'_id': sid}, {'$set': {'state': 'ready', 'model': model,
                'degraded': self.provider.degraded, 'vector_collection': name, 'index_context': self.contextual}})
            self._job_update(jid, phase='ready')
            self._stage('ready', job)
            # Candidate status may lead or lag; head pointer always decides visibility.
            for offset in range(0, len(data), 128):
                active_rows = [{**row, 'status': 'active'} for row in data[offset:offset+128]]
                self.vectors.upsert(collection_name=name, data=active_rows)
            self.vectors.flush(collection_name=name)
            head = self.db.heads.find_one({'_id': job['head_id']})
            old = head.get('active')
            self.db.versions.update_one({'_id': sid}, {'$set': {'supersedes': old}})
            self._job_update(jid, phase='publishing', old_snapshot=old)
            if lease['lost'].is_set():
                raise IngestionBusy('lost document lease before publication')
            published_at = utcnow()
            result = self.db.heads.update_one({'_id': job['head_id'], 'lease_token': lease['token'],
                'lease_until': {'$gt': utcnow()}, 'active': old}, {
                    '$set': {'active': sid, 'published_at': published_at},
                    '$push': {'commits': {'snapshot': sid, 'at': published_at}}})
            if result.matched_count != 1:
                raise IngestionBusy('publication fenced by another writer')
            self._stage('head_published', job)
            self._finish_publication(job)
            return {'snapshot_id': sid, 'job_id': jid, 'duplicate': False, 'chunks': len(parts), 'degraded': self.provider.degraded}
        except Exception as exc:
            # An ambiguous publish acknowledgement must be resolved against Mongo.
            head = self.db.heads.find_one({'_id': job['head_id']})
            if head and (head.get('active') == sid or any(c['snapshot'] == sid for c in head.get('commits', []))):
                self._job_update(jid, phase='reconcile_publication', error=f'{type(exc).__name__}: {exc}')
                raise
            self._job_update(jid, error=f'{type(exc).__name__}: {exc}', phase='compensating')
            self._compensate(job)
            raise

    def _index_projection(self, content, parts):
        notes = self.document_context.projection(content, parts) if self.document_context else {}
        return {'index_context': self.contextual, 'retrieval_contexts': notes,
                'context_manifest_sha256': self.document_context.fingerprint if notes else None}

    def _index_text(self, version, row):
        text = version['content'][row['start']:row['end']]
        prefix = row.get('index_prefix', '')
        # Use the published projection, not a reader's current feature switches.
        if version.get('index_context', self.contextual):
            prefix = '\n'.join([version['title'][:200], ' > '.join(row['heading'])[:300], prefix])
        note = version.get('retrieval_contexts', {}).get(f"{row['start']}:{row['end']}")
        if note:
            if digest(text) != note['chunk_sha256']:
                raise ValueError('retrieval description source hash mismatch')
            prefix = '\n'.join(filter(None, [prefix, note['text']]))
        return prefix+'\n'+text if prefix else text

    def _finish_publication(self, job):
        sid = job['snapshot_id']
        version = self.db.versions.find_one({'_id': sid})
        head = self.db.heads.find_one({'_id': job['head_id']})
        commit = next((c for c in (head or {}).get('commits', []) if c['snapshot'] == sid), None)
        if not commit and (not head or head.get('active') != sid):
            raise IngestionBusy('snapshot has never been published')
        at = commit['at'] if commit else head['published_at']
        self.db.versions.update_one({'_id': sid, 'published_at': {'$exists': False}},
            {'$set': {'published_at': at, 'state': 'ready' if head.get('active') == sid else 'archived'}})
        if version.get('supersedes'):
            self.db.versions.update_one({'_id': version['supersedes']}, {'$set': {'state': 'archived'}})
            old = self.db.versions.find_one({'_id': version['supersedes']})
            if old and old.get('vector_collection'):
                ids = [r['_id'] for r in self.db.chunks.find({'snapshot': old['_id']}, {'_id': 1})]
                for offset in range(0, len(ids), 128):
                    candidates = self.vectors.get(collection_name=old['vector_collection'], ids=ids[offset:offset+128],
                        output_fields=['id', 'snapshot', 'scope', 'vector', 'status'], consistency_level='Strong')
                    if candidates:
                        self.vectors.upsert(collection_name=old['vector_collection'],
                            data=[{**r, 'status': 'archived'} for r in candidates])
                self.vectors.flush(collection_name=old['vector_collection'])
        self._stage('old_archived', job)
        self._job_update(job['_id'], status='completed', phase='published', error=None)

    def _compensate(self, job):
        sid = job['snapshot_id']
        # Never compensate a committed or historical version, even on delayed retries.
        version = self.db.versions.find_one({'_id': sid})
        head = self.db.heads.find_one({'_id': job['head_id']})
        committed = head and (head.get('active') == sid or any(c['snapshot'] == sid for c in head.get('commits', [])))
        if committed or (version and version.get('published_at')):
            raise ValueError('refuse to compensate a published snapshot')
        try:
            name = job.get('vector_collection')
            if name:
                ids = job.get('milvus_ids', [])
                for offset in range(0, len(ids), 128):
                    self.vectors.delete(collection_name=name, ids=ids[offset:offset+128])
            self._stage('vectors_compensated', job)
            self.db.chunks.delete_many({'snapshot': sid})
            self.db.versions.update_one({'_id': sid}, {'$set': {'state': 'pending'}})
            self._job_update(job['_id'], status='failed', phase='compensated')
        except Exception as exc:
            self._job_update(job['_id'], status='cleanup_failed', cleanup_error=f'{type(exc).__name__}: {exc}')
            raise

    def reconcile(self, *, collection='library'):
        """Recover interrupted jobs, gated by the same fenced document lease."""
        counts = {'published': 0, 'compensated': 0, 'busy': 0, 'errors': []}
        for job in self.db.jobs.find({'collection': collection, 'status': {'$in': ['pending', 'cleanup_failed']}}):
            try:
                with self._lease(collection, job['document_id']):
                    current = self.db.jobs.find_one({'_id': job['_id']})
                    if current['status'] not in ['pending', 'cleanup_failed']:
                        if current['status'] == 'completed' and job['status'] == 'pending':
                            counts['published'] += 1  # repaired while acquiring the lease
                        continue
                    if job.get('kind') == 'index_rebuild':
                        version = self.db.versions.find_one({'_id': job['snapshot_id']})
                        if version.get('vector_collection') == job['vector_collection']:
                            self._job_update(job['_id'], status='completed', phase='index_published')
                            counts['published'] += 1
                        else:
                            self._cleanup_rebuild(job); counts['compensated'] += 1
                        continue
                    version = self.db.versions.find_one({'_id': job['snapshot_id']})
                    head = self.db.heads.find_one({'_id': job['head_id']})
                    if head.get('active') == job['snapshot_id'] or any(c['snapshot'] == job['snapshot_id'] for c in head.get('commits', [])):
                        self._finish_publication(job); counts['published'] += 1
                    elif version and version.get('published_at'):
                        self._job_update(job['_id'], status='completed', phase='published'); counts['published'] += 1
                    else:
                        self._compensate(job); counts['compensated'] += 1
            except IngestionBusy:
                counts['busy'] += 1
            except Exception as exc:
                counts['errors'].append({'job_id': job['_id'], 'error': f'{type(exc).__name__}: {exc}'})
        return counts

    def retry(self, job_id):
        job = self.db.jobs.find_one({'_id': job_id})
        if not job or job['status'] != 'failed':
            raise ValueError('only compensated failed jobs can be retried')
        if job.get('kind') == 'index_rebuild':
            raise ValueError('retry an index rebuild using knowledge reindex')
        version = self.db.versions.find_one({'_id': job['snapshot_id']})
        if version:
            with self._lease(job['collection'], job['document_id']) as lease:
                found = self._duplicate(job['collection'], job['document_id'], job['source_hash'],
                                        job['title'], version['metadata'])
                if found:
                    return found
                attempt = self._new_job(lease, job['collection'], job['document_id'], job['uri'], job['title'],
                    source_hash=job['source_hash'], source_kind=job['source_kind'], metadata=version['metadata'],
                    retry_of=job_id)
                return self._execute(attempt, version['content'], lease)
        if job.get('source_path'):
            path = Path(job['source_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != job['source_hash']:
                raise ValueError('source file changed; retry requires the original upload')
            return self.ingest_file(path, collection=job['collection'], document_id=job['document_id'])
        raise ValueError('failed job has no recoverable source')

    def _parse(self, data, suffix):
        if self.parser == 'mineru' and suffix == '.pdf':
            from .parsing import mineru_parse
            return mineru_parse(data)
        if suffix == '.pdf':
            from pypdf import PdfReader
            pages = [p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages]
            offsets, position = [], 0
            for page in pages:
                offsets.append(position); position += len(page)+2
            return '\n\n'.join(pages), {'page_starts': offsets, 'parser': 'pypdf-text'}
        if suffix in ['.md', '.txt']:
            return data.decode('utf-8'), {'parser': 'utf8'}
        if suffix == '.docx':
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            blocks = []
            for block in Document(io.BytesIO(data)).iter_inner_content():
                if isinstance(block, Paragraph):
                    style = block.style.name or ''
                    match = re.match('Heading ([1-6])', style)
                    blocks.append(('#'*int(match[1])+' ' if match else '')+block.text)
                elif isinstance(block, Table):
                    rows = [[cell.text.replace('\n',' ').replace('|','\\|') for cell in row.cells] for row in block.rows]
                    if rows:
                        blocks.append('\n'.join(['| '+' | '.join(rows[0])+' |', '| '+' | '.join(['---']*len(rows[0]))+' |',
                                       *['| '+' | '.join(row)+' |' for row in rows[1:]]]))
            return '\n\n'.join(blocks), {'parser': 'python-docx'}
        raise ValueError('supported formats: .md, .txt, text .pdf, .docx; scan PDF requires MinerU')

    def ingest_file(self, path, *, collection='library', document_id=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        source_hash = hashlib.sha256(data).hexdigest()
        if len(data) > 32*1024*1024:
            raise ValueError('upload exceeds local 32 MiB limit')
        document_id = document_id or path.as_uri()
        with self._lease(collection, 'upload:'+source_hash), self._lease(collection, document_id) as lease:
            found = self._duplicate(collection, document_id, source_hash, path.name, None)
            if found:
                return found
            # Identical bytes at another path share the parsed snapshot and embeddings.
            upload = self.db.uploads.find_one({'collection': collection, 'file_hash': source_hash})
            if upload:
                existing = self.db.versions.find_one({'_id': upload['snapshot_id'], 'published_at': {'$exists': True}})
                if existing:
                    # Cross-source copies only alias the same logical document when no ID was forced.
                    # Explicit IDs must not silently collapse independently versioned documents.
                    if document_id == path.as_uri():
                        self.db.uploads.update_one({'_id': upload['_id']}, {'$addToSet': {'aliases': path.as_uri()}})
                        return {'snapshot_id': existing['_id'], 'duplicate': True,
                                'status': self.snapshot(existing['_id'], collection=collection)['status']}
            job = self._new_job(lease, collection, document_id, path.as_uri(), path.name,
                source_hash=source_hash, source_kind='upload_bytes', source_path=str(path))
            try:
                content, metadata = self._parse(data, path.suffix.lower())
                if not content.strip():
                    raise ValueError('empty parsed document; scans need OCR')
            except Exception as exc:
                self._job_update(job['_id'], status='failed', phase='parse_failed', error=f'{type(exc).__name__}: {exc}')
                raise
            job['metadata'] = metadata
            self._job_update(job['_id'], metadata=metadata)
            result = self._execute(job, content, lease)
            self.db.uploads.update_one({'_id': digest(collection+':'+source_hash)}, {'$set': {
                'collection': collection, 'file_hash': source_hash, 'snapshot_id': result['snapshot_id']},
                '$addToSet': {'aliases': path.as_uri()}}, upsert=True)
            return result

    def snapshot(self, sid, *, collection):
        row = self.db.versions.find_one({'_id': sid, 'collection': collection})
        if not row or digest(row['content']) != row['hash']:
            raise ValueError('missing snapshot or content hash mismatch')
        head = self.db.heads.find_one({'_id': row['head_id']})
        status = 'active' if head and head.get('active') == sid and row['state'] == 'ready' else (
            'archived' if row.get('published_at') else 'pending')
        return {**row, 'status': status, 'metadata': json.dumps(row['metadata'], ensure_ascii=False)}

    def _eligible(self, collection, snapshot_id, as_of):
        if snapshot_id:
            row = self.snapshot(snapshot_id, collection=collection)
            return [row] if row['status'] != 'pending' else []
        if as_of:
            cutoff = datetime.fromisoformat(as_of.replace('Z', '+00:00'))
            if cutoff.tzinfo is None:
                raise ValueError('as_of requires timezone')
            seen, result = set(), []
            for row in self.db.versions.find({'collection': collection, 'published_at': {'$lte': cutoff}}).sort('published_at', -1):
                if row['document_id'] not in seen:
                    seen.add(row['document_id']); result.append(self.snapshot(row['_id'], collection=collection))
            return result
        return [self.snapshot(head['active'], collection=collection)
                for head in self.db.heads.find({'collection': collection, 'active': {'$ne': None}})]

    def search(self, query, *, collection='library', snapshot_id=None, as_of=None, limit=5, mode='hybrid', rerank=True,
               candidate_policy=None, trace=None):
        if mode not in ['hybrid', 'bm25', 'vector']:
            raise ValueError('unknown retrieval mode')
        requested_policy = self.candidate_policy if candidate_policy is None else candidate_policy
        if requested_policy not in POLICIES:
            raise ValueError('unknown candidate policy')
        policy = requested_policy if mode == 'hybrid' else 'rrf'
        if not query.strip() or limit <= 0:
            return []
        versions = self._eligible(collection, snapshot_id, as_of)
        if not versions:
            return []
        by_sid = {v['_id']: v for v in versions}
        rows = list(self.db.chunks.find({'snapshot': {'$in': list(by_sid)}}).sort('_id', 1))
        # Stable ties across rebuilds/models; random staging IDs cannot decide relevance.
        def tie(cid):
            row = by_id[cid]
            version = by_sid[row['snapshot']]
            return version['document_id'], row['start'], row['end'], version['hash']
        by_id = {row['_id']: row for row in rows}
        lexical, semantic = [], []
        if mode != 'vector':
            counts = [Counter(features(self._index_text(by_sid[row['snapshot']], row))) for row in rows]
            terms = set(features(query)); df = Counter(t for c in counts for t in c if t in terms)
            lengths = [sum(c.values()) for c in counts]; avg = sum(lengths)/max(1,len(lengths)) or 1
            for row, freq, length in zip(rows, counts, lengths, strict=True):
                score = sum(math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*freq[t]*2.2 /
                    (freq[t]+1.2*(.25+.75*length/avg)) for t in terms if freq[t])
                if score > 0:
                    lexical.append((row['_id'], score))
            lexical.sort(key=lambda x: (-x[1], tie(x[0]))); lexical = lexical[:20]
        vector_status = 'disabled' if mode == 'bm25' else 'available'
        if mode != 'bm25':
            try:
                with self.lock:
                    vector = self.provider.encode_query(query)
                    validate_vectors([vector], 1, self.provider.dimensions)
                    if self.provider.degraded:
                        raise ValueError('semantic embedding unavailable')
                    model = self.provider.model_id
                names = {v['vector_collection'] for v in versions if v.get('model') == model}
                for name in names:
                    sids = [v['_id'] for v in versions if v.get('vector_collection') == name]
                    status_filter = 'status in ["active", "archived"]' if snapshot_id or as_of else 'status == "active"'
                    filt = 'scope == '+json.dumps(digest(collection))+' and '+status_filter+' and snapshot in '+json.dumps(sids)
                    hits = self.vectors.search(collection_name=name, data=[vector], filter=filt, limit=40,
                        output_fields=['snapshot'], consistency_level='Strong')[0]
                    for hit in hits:
                        if hit['id'] in by_id:
                            semantic.append((hit['id'], float(hit['distance'])))
                semantic.sort(key=lambda x: (-x[1], tie(x[0]))); semantic = semantic[:40]
            except Exception:
                vector_status = 'fallback_bm25'
                if mode == 'vector':
                    raise
        scores, route_ids = {}, []
        for route in ([lexical] if mode == 'bm25' else [semantic] if mode == 'vector' else [lexical, semantic]):
            seen, ids = set(), []
            for rank, (cid, _) in enumerate(route,1):
                row = by_id[cid]; version = by_sid[row['snapshot']]
                key = (row['snapshot'], digest(version['content'][row['start']:row['end']].strip()))
                if key in seen:
                    continue
                seen.add(key); scores[cid] = scores.get(cid,0)+1/(60+rank)
                ids.append(cid)
            route_ids.append(ids)
        budget = max(20,limit*2)
        fused = sorted(scores, key=lambda cid: (-scores[cid], tie(cid)))
        selected = select_candidates(route_ids[0] if mode != 'vector' else [],
            route_ids[-1] if mode != 'bm25' else [], fused, budget=budget, policy=policy)
        if trace is not None:
            lexical_target, semantic_target = quota_targets(budget, policy)
            trace.update(policy=policy, budget=budget, lexical_quota=lexical_target, semantic_quota=semantic_target,
                lexical=route_ids[0] if mode != 'vector' else [],
                semantic=route_ids[-1] if mode != 'bm25' else [], fused=fused, selected=selected,
                vector_status=vector_status)
        hits = []
        for cid in selected:
            row = by_id[cid]
            # Second gate: authoritative state and source bytes, never Milvus payload text.
            version = self.snapshot(row['snapshot'], collection=collection)
            if not snapshot_id and not as_of and version['status'] != 'active':
                continue
            text = version['content'][row['start']:row['end']]
            pages = json.loads(version['metadata']).get('page_starts', [])
            hits.append({'chunk_id': cid, 'snapshot_id': version['_id'], 'uri': version['uri'],
                'title': version['title'], 'content': text, 'start': row['start'], 'end': row['end'],
                'heading': row['heading'], 'table_header': row.get('index_prefix') or None,
                'page': sum(o<=row['start'] for o in pages) if pages else None, 'score': scores[cid],
                'degraded': bool(version.get('degraded')) or vector_status == 'fallback_bm25',
                'model_mismatch': version.get('model') != self.provider.model_id, 'index_context': bool(version.get('index_context')),
                'rerank_status': 'disabled', 'vector_status': vector_status})
        if trace is not None:
            trace['validated'] = [h['chunk_id'] for h in hits]
        if rerank and self.reranker and hits:
            try:
                ranked = self.reranker(query, self.prepare_rerank_hits(hits, collection=collection))
                if sorted(h['chunk_id'] for h in ranked) != sorted(h['chunk_id'] for h in hits):
                    raise ValueError('reranker must return a permutation')
                trusted = {h['chunk_id']: h for h in hits}; hits = [trusted[h['chunk_id']] for h in ranked]
                for hit in hits:
                    hit['rerank_status'] = 'applied'
            except Exception:
                for hit in hits:
                    hit['rerank_status'] = 'fallback'
        return hits[:limit]

    def prepare_rerank_hits(self, hits, *, collection='library'):
        """Bounded original neighbors for scoring only; never enlarge evidence spans."""
        inputs, snapshots = [], {}
        for hit in hits:
            value = dict(hit)
            value.pop('_rerank_neighbors', None)
            if self.reranker_neighbor_chars:
                sid = hit['snapshot_id']
                if sid not in snapshots:
                    snapshots[sid] = self.snapshot(sid, collection=collection)
                content = snapshots[sid]['content']
                a, b = hit['start'], hit['end']
                if content[a:b] != hit['content']:
                    raise ValueError('reranker target differs from immutable source')
                radius = self.reranker_neighbor_chars
                value['_rerank_neighbors'] = {'before': content[max(0,a-radius):a], 'after': content[b:b+radius]}
            inputs.append(value)
        return inputs

    windows = KnowledgeStore.windows

    def jobs(self, *, collection='library'):
        return [{k:v for k,v in row.items() if k not in ['mongo_ids','milvus_ids']}
                for row in self.db.jobs.find({'collection': collection}).sort('created_at',-1).limit(100)]

    def _cleanup_rebuild(self, job):
        current = self.db.versions.find_one({'_id': job['snapshot_id']})
        if current and current.get('vector_collection') == job.get('vector_collection'):
            raise ValueError('refuse to delete a published index projection')
        try:
            name = job.get('vector_collection')
            if name and self.vectors.has_collection(collection_name=name):
                self.vectors.drop_collection(collection_name=name)
            self._job_update(job['_id'], status='failed', phase='index_compensated')
        except Exception as exc:
            self._job_update(job['_id'], status='cleanup_failed', cleanup_error=f'{type(exc).__name__}: {exc}')
            raise

    def reindex(self, *, collection='library'):
        """Rebuild in disjoint generations, then publish each version's index pointer.

        Facts and source version IDs never change. This is per-version atomic
        projection switching, not a transaction across the whole collection.
        """
        count = 0
        for version in self.db.versions.find({'collection': collection, 'published_at': {'$exists': True}}):
            with self._lease(collection, version['document_id']) as lease:
                if digest(version['content']) != version['hash']:
                    raise ValueError('snapshot content hash mismatch')
                jid = uuid.uuid4().hex
                job = {'_id': jid, 'kind': 'index_rebuild', 'snapshot_id': version['_id'],
                       'collection': collection, 'head_id': version['head_id'], 'document_id': version['document_id'],
                       'status': 'pending', 'phase': 'index_encoding', 'created_at': utcnow(), 'updated_at': utcnow()}
                self.db.jobs.insert_one(job)
                try:
                    rows = list(self.db.chunks.find({'snapshot': version['_id']}).sort('_id',1))
                    if len(rows) != version['expected_chunks']:
                        raise ValueError('canonical chunk set is incomplete')
                    projection = self._index_projection(version['content'], rows)
                    projected = {**version, **projection}
                    with self.lock:
                        vectors = self.provider.encode_documents([self._index_text(projected,row) for row in rows])
                        validate_vectors(vectors,len(rows),self.provider.dimensions)
                        if self.provider.degraded and not self.allow_degraded:
                            raise ValueError('semantic embedding unavailable')
                        model, dimensions = self.provider.model_id, self.provider.dimensions
                        # Record the generated collection name BEFORE creating it.
                        name = self._vector_collection_name(model, dimensions, jid,
                            projection['context_manifest_sha256'])
                        job.update(vector_collection=name, milvus_ids=[row['_id'] for row in rows])
                        self._job_update(jid, vector_collection=name, milvus_ids=job['milvus_ids'])
                        self._ensure_vector_collection(model, dimensions, generation=jid,
                            context_fingerprint=projection['context_manifest_sha256'])
                    status = self.snapshot(version['_id'], collection=collection)['status']
                    for i in range(0,len(rows),128):
                        self.vectors.upsert(collection_name=name, data=[{'id':row['_id'], 'snapshot':version['_id'],
                            'scope':digest(collection), 'status':status, 'vector':[float(v) for v in vector]}
                            for row,vector in zip(rows[i:i+128],vectors[i:i+128],strict=True)])
                        self._stage('reindex_batch_written', job)
                    self.vectors.flush(collection_name=name)
                    for i in range(0,len(rows),128):
                        actual = self.vectors.get(collection_name=name, ids=job['milvus_ids'][i:i+128],
                            output_fields=['id','vector'], consistency_level='Strong')
                        if {r['id'] for r in actual} != set(job['milvus_ids'][i:i+128]):
                            raise ValueError('rebuilt index is incomplete')
                        validate_vectors([r['vector'] for r in actual],len(actual),dimensions)
                    if lease['lost'].is_set():
                        raise IngestionBusy('lost lease during index rebuild')
                    result = self.db.versions.update_one({'_id':version['_id'],
                        'vector_collection':version['vector_collection']},{'$set':{'vector_collection':name,
                        'model':model,'degraded':self.provider.degraded,**projection}})
                    if not result.matched_count:
                        raise IngestionBusy('index projection changed during rebuild')
                    self._job_update(jid, status='completed', phase='index_published')
                    count += 1
                except Exception as exc:
                    self._job_update(jid, error=f'{type(exc).__name__}: {exc}')
                    current = self.db.versions.find_one({'_id': version['_id']})
                    if current.get('vector_collection') != job.get('vector_collection'):
                        self._cleanup_rebuild(job)
                    raise
        return {'reindexed_snapshots':count,'model':self.provider.model_id,'degraded':self.provider.degraded}

"""Single-Milvus research library, including originals, versions and jobs.

Writers must run on ONE host and use the same lock directory. A process lock
serializes mutations; Milvus has no application-level compare-and-swap lease.
Readers use a small head record plus verified source manifests as visibility
boundary. This is recoverable publication, not a multi-record transaction.
"""
from __future__ import annotations
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from filelock import FileLock, Timeout
from .candidate_selection import POLICIES
from .milvus_records import MilvusRecords
from .retrieval import DocumentRetrievalMixin, structured_chunks, validate_vectors
from .store import KnowledgeStore, digest
from ..memory_system.embeddings import ResilientEmbeddingProvider

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


class IngestionBusy(RuntimeError):
    pass


class MilvusStore(DocumentRetrievalMixin):
    def __init__(self, *, milvus_uri='http://127.0.0.1:19530', milvus_token='',
                 namespace='deepresearch_knowledge', lock_dir='.runtime/knowledge-locks',
                 provider=None, reranker=None, contextual=False, parser='auto', parse_options=None,
                 allow_degraded=False, candidate_policy='rrf', document_context=None,
                 reranker_neighbor_chars=0, vector_client=None):
        if candidate_policy not in POLICIES:
            raise ValueError('unknown candidate policy')
        if not isinstance(reranker_neighbor_chars, int) or isinstance(reranker_neighbor_chars, bool) or not 0 <= reranker_neighbor_chars <= 300:
            raise ValueError('reranker_neighbor_chars must be an integer in 0..300')
        if not namespace.strip():
            raise ValueError('Milvus namespace must not be empty')
        from pymilvus import MilvusClient
        self.namespace = namespace
        self.vectors = vector_client if vector_client is not None else MilvusClient(uri=milvus_uri, token=milvus_token, timeout=30)
        self.provider = provider or ResilientEmbeddingProvider(fallback_enabled=False)
        self.reranker, self.contextual, self.parser = reranker, contextual, parser
        self.parse_options = parse_options
        self.allow_degraded, self.candidate_policy = allow_degraded, candidate_policy
        self.document_context, self.reranker_neighbor_chars = document_context, reranker_neighbor_chars
        self.lock = threading.RLock()
        # URI normalization makes the common localhost aliases use one host lock.
        endpoint = milvus_uri.replace('localhost', '127.0.0.1').rstrip('/')
        if '://' not in endpoint:
            endpoint = str(Path(endpoint).resolve())
        directory = Path(lock_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / (digest(endpoint+'\n'+namespace)+'.lock'))
        with _LOCKS_GUARD:
            self._file_lock = _LOCKS.setdefault(path, FileLock(path, timeout=0))
        try:
            with self._file_lock:
                self.records = MilvusRecords(self.vectors, namespace)
        except Timeout as exc:
            self.vectors.close()
            raise IngestionBusy('another local knowledge writer is active') from exc

    def close(self):
        self.vectors.close()

    def _stage(self, stage, job):
        """Fault injection after durable boundaries."""

    @staticmethod
    def _head_id(collection, document_id):
        return digest(json.dumps([collection, document_id], ensure_ascii=False))

    @contextmanager
    def _writer(self, *, repair=True):
        try:
            self._file_lock.acquire(timeout=0)
        except Timeout as exc:
            raise IngestionBusy('another local knowledge writer is active') from exc
        try:
            # Repair commits before a later writer can move the head again.
            if repair and self._file_lock.lock_counter == 1:
                for job in self.records.list('job'):
                    if job['status'] != 'pending':
                        continue
                    if job.get('kind') == 'index_rebuild' and job['phase'] == 'publishing_index':
                        version = self.records.get('version', job['snapshot_id'])
                        if version and version.get('vector_collection') == job.get('vector_collection'):
                            self._job_update(job, status='completed', phase='index_published')
                        else:
                            raise IngestionBusy('unconfirmed index publication; run knowledge reconcile first')
                    elif job.get('kind') != 'index_rebuild':
                        head = self.records.get('head', job['head_id'])
                        if head and head.get('active') == job['snapshot_id']:
                            self._finish_publication(job)
                        elif job['phase'] in ['publishing', 'publication_unknown', 'reconcile_publication']:
                            raise IngestionBusy('unconfirmed publication; run knowledge reconcile first')
            yield
        finally:
            self._file_lock.release()

    def _job_update(self, job, **values):
        job.update(values, updated_at=now())
        self.records.put('job', job['_id'], job)

    def _new_job(self, collection, document_id, uri, title, **extra):
        job = {'_id': uuid.uuid4().hex, 'snapshot_id': uuid.uuid4().hex,
               'head_id': self._head_id(collection, document_id), 'collection': collection,
               'document_id': document_id, 'uri': uri, 'title': title,
               'status': 'pending', 'phase': 'created', 'created_at': now(), **extra}
        self._job_update(job)
        return job

    def _duplicate(self, collection, document_id, source_hash, title, metadata):
        versions = [v for v in self.records.list('version', collection=collection, document_id=document_id)
                    if v.get('published_at') and v['source_hash'] == source_hash]
        if not versions:
            return None
        version = max(versions, key=lambda v: v['published_at'])
        version['title'] = title
        if metadata:
            version['metadata'].update(metadata)
        self.records.put('version', version['_id'], version)
        return {'snapshot_id': version['_id'], 'duplicate': True,
                'status': self.snapshot(version['_id'], collection=collection)['status']}

    def ingest(self, content, *, uri, title, collection='library', metadata=None, document_id=None, refresh_capture=False):
        if not content.strip() or not collection.strip():
            raise ValueError('nonempty content and collection required')
        document_id = document_id or uri
        with self._writer():
            if refresh_capture:
                from .research_sources import capture_is_older, merge_capture_metadata
                head = self.records.get('head', self._head_id(collection, document_id))
                current = self.records.get('version', head['active']) if head else None
                if current and capture_is_older(metadata or {}, current['metadata']):
                    return {'snapshot_id': current['_id'], 'superseded_capture': True}
                if current and current['hash'] == digest(content):
                    current['metadata'] = merge_capture_metadata(current['metadata'], metadata or {})
                    self.records.put('version', current['_id'], current)
                    return {'snapshot_id': current['_id'], 'duplicate': True, 'status': 'active'}
            else:
                found = self._duplicate(collection, document_id, digest(content), title, metadata)
                if found:
                    return found
            job = self._new_job(collection, document_id, uri, title, source_hash=digest(content),
                                source_kind='parsed_text', metadata=metadata or {})
            return self._execute(job, content)

    def ingest_file(self, path, *, collection='library', document_id=None, reparse=False):
        path = Path(path).resolve()
        if path.stat().st_size > 32*1024*1024:
            raise ValueError('upload exceeds local 32 MiB limit')
        data = path.read_bytes()
        if len(data) > 32*1024*1024:
            raise ValueError('upload exceeds local 32 MiB limit')
        if not collection.strip():
            raise ValueError('nonempty collection required')
        source_hash = hashlib.sha256(data).hexdigest()
        forced = document_id is not None
        document_id = document_id or path.as_uri()
        with self._writer():
            found = None if reparse else self._duplicate(collection, document_id, source_hash, path.name, None)
            if found:
                return found
            upload = self.records.get('upload', self._head_id(collection, source_hash))
            if upload and not forced and not reparse:
                version = self.records.get('version', upload['snapshot_id'])
                if version and version.get('published_at'):
                    upload['aliases'] = sorted(set(upload['aliases']+[path.as_uri()]))
                    self.records.put('upload', self._head_id(collection, source_hash), upload)
                    return {'snapshot_id': version['_id'], 'duplicate': True,
                            'status': self.snapshot(version['_id'], collection=collection)['status']}
            job = self._new_job(collection, document_id, path.as_uri(), path.name,
                                source_hash=source_hash, source_kind='upload_bytes', source_path=str(path), reparse=bool(reparse))
            try:
                content, metadata = self._parse(data, path.suffix.lower())
                if not content.strip():
                    raise ValueError('empty parsed document; scans need OCR')
            except Exception as exc:
                self._job_update(job, status='failed', phase='parse_failed', error=f'{type(exc).__name__}: {exc}')
                raise
            self._job_update(job, metadata=metadata)
            result = self._execute(job, content)
            self.records.put('upload', self._head_id(collection, source_hash), {
                'collection': collection, 'document_id': document_id, 'file_hash': source_hash,
                'snapshot_id': result['snapshot_id'], 'aliases': [path.as_uri()]})
            return result

    def _vector_collection_name(self, model, dimension, generation):
        return 'rag_vectors_'+digest(json.dumps([self.namespace, model, dimension, generation, 'single-v1']))[:24]

    def _ensure_index(self, name, dimensions):
        from pymilvus import DataType
        if self.vectors.has_collection(collection_name=name):
            return
        schema = self.vectors.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=80)
        for field in ['snapshot', 'scope']:
            schema.add_field(field, DataType.VARCHAR, max_length=64)
        schema.add_field('status', DataType.VARCHAR, max_length=16)
        schema.add_field('content', DataType.VARCHAR, max_length=8192)
        schema.add_field('start', DataType.INT64)
        schema.add_field('end', DataType.INT64)
        schema.add_field('vector', DataType.FLOAT_VECTOR, dim=dimensions)
        params = self.vectors.prepare_index_params()
        params.add_index(field_name='vector', index_type='FLAT', metric_type='COSINE')
        self.vectors.create_collection(collection_name=name, schema=schema, index_params=params, consistency_level='Strong')

    def _write_index(self, version, job):
        rows = version['chunks']
        with self.lock:
            vectors = self.provider.encode_documents([self._index_text(version, r) for r in rows])
            dimensions, model = self.provider.dimensions, self.provider.model_id
            validate_vectors(vectors, len(rows), dimensions)
            if self.provider.degraded and not self.allow_degraded:
                raise ValueError('semantic embedding unavailable; do not publish a hash index')
        # Share the initial index by model; rebuilds use a disjoint generation.
        generation = job['_id'] if job.get('kind') == 'index_rebuild' else 'shared'
        name = self._vector_collection_name(model, dimensions, generation)
        self._job_update(job, vector_collection=name, phase='writing_vectors')
        self._ensure_index(name, dimensions)
        data = [{'id': r['_id'], 'snapshot': version['_id'], 'scope': digest(version['collection']),
                 'status': 'active', 'content': version['content'][r['start']:r['end']],
                 'start': r['start'], 'end': r['end'], 'vector': [float(v) for v in vector]}
                for r, vector in zip(rows, vectors, strict=True)]
        for i in range(0, len(data), 128):
            self.vectors.upsert(collection_name=name, data=data[i:i+128])
            self._stage('reindex_batch_written' if job.get('kind') == 'index_rebuild' else 'vector_batch_written', job)
        self.vectors.flush(collection_name=name)
        for i in range(0, len(data), 128):
            expected = {r['id']: r for r in data[i:i+128]}
            actual = self.vectors.get(collection_name=name, ids=list(expected),
                output_fields=['id', 'content', 'start', 'end', 'vector'], consistency_level='Strong')
            # Lite can briefly lag an acknowledged flush even with Strong reads.
            # Confirm only by reading; do not repeat writes or relax validation.
            for _ in range(5):
                if {r['id'] for r in actual} == set(expected):
                    break
                time.sleep(.05)
                actual = self.vectors.get(collection_name=name, ids=list(expected),
                    output_fields=['id', 'content', 'start', 'end', 'vector'], consistency_level='Strong')
            if {r['id'] for r in actual} != set(expected):
                raise ValueError('Milvus index is incomplete')
            validate_vectors([r['vector'] for r in actual], len(actual), dimensions)
            if any(any(r[k] != expected[r['id']][k] for k in ['content', 'start', 'end']) for r in actual):
                raise ValueError('Milvus source projection differs from original')
        return {'vector_collection': name, 'model': model, 'degraded': self.provider.degraded}

    def _execute(self, job, content, *, imported=None):
        sid = job['snapshot_id']
        publication_attempted = False
        try:
            if len(content.encode('utf-8')) > 8*1024*1024:
                raise ValueError('parsed text exceeds local 8 MiB document limit')
            parts = structured_chunks(content)
            if not parts or len(parts) > 10000:
                raise ValueError('document must contain 1..10000 chunks')
            rows = [{'_id': f'{sid}:{i}', 'snapshot': sid, 'start': p['start'], 'end': p['end'],
                     'heading': p['path'], 'index_prefix': p['index_prefix']} for i, p in enumerate(parts)]
            version = {'_id': sid, 'id': sid, 'collection': job['collection'], 'document_id': job['document_id'],
                       'head_id': job['head_id'], 'uri': job['uri'], 'title': job['title'],
                       'content': content, 'hash': digest(content), 'source_hash': job['source_hash'],
                       'state': 'pending', 'created_at': now(), 'metadata': job.get('metadata') or {},
                       'expected_chunks': len(rows), 'chunks': rows, 'job_id': job['_id'], 'supersedes': None}
            version.update(self._index_projection(content, parts))
            if imported:
                # Keep exact old spans and index projection for persisted citations.
                version.update({k: imported[k] for k in ['chunks', 'expected_chunks', 'index_context',
                    'retrieval_contexts', 'context_manifest_sha256'] if k in imported})
            self.records.put('version', sid, version)
            self._stage('snapshot_written', job)
            version.update(self._write_index(version, job), state='ready')
            self.records.put('version', sid, version)
            self._job_update(job, phase='ready')
            self._stage('ready', job)
            head = self.records.get('head', job['head_id'])
            version['supersedes'] = head.get('active') if head else None
            self.records.put('version', sid, version)
            at = imported['published_at'] if imported else now()
            self._job_update(job, phase='publishing', published_at=at, old_snapshot=version['supersedes'])
            publication_attempted = True
            self.records.put('head', job['head_id'], {'collection': job['collection'],
                'document_id': job['document_id'], 'active': sid, 'published_at': at})
            self._stage('head_published', job)
            self._finish_publication(job)
            return {'snapshot_id': sid, 'job_id': job['_id'], 'duplicate': False,
                    'chunks': len(version['chunks']), 'degraded': version['degraded']}
        except Exception as exc:
            # An unavailable/ambiguous head read MUST NOT trigger destructive cleanup.
            try:
                head = self.records.get('head', job['head_id'])
            except Exception:
                raise exc
            if head and head.get('active') == sid:
                self._job_update(job, phase='reconcile_publication', error=f'{type(exc).__name__}: {exc}')
            elif publication_attempted:
                # A timed-out publication may still reach the server. Keep the
                # staged data for explicit reconciliation, rather than delete it.
                self._job_update(job, phase='publication_unknown', error=f'{type(exc).__name__}: {exc}')
            else:
                self._job_update(job, phase='compensating', error=f'{type(exc).__name__}: {exc}')
                self._compensate(job)
            raise

    def _finish_publication(self, job):
        version = self.records.get('version', job['snapshot_id'])
        head = self.records.get('head', job['head_id'])
        if not head or head['active'] != version['_id']:
            if version.get('published_at'):
                self._job_update(job, status='completed', phase='published')
                return
            raise ValueError('snapshot has not been published')
        version.update(published_at=head['published_at'], state='ready')
        self.records.put('version', version['_id'], version)
        if version.get('supersedes'):
            old = self.records.get('version', version['supersedes'])
            old['state'] = 'archived'
            self.records.put('version', old['_id'], old)
        # ANN status is deliberately not authority; head + snapshot select versions.
        self._stage('old_archived', job)
        self._job_update(job, status='completed', phase='published', error=None)

    def _compensate(self, job):
        version = self.records.get('version', job['snapshot_id'])
        head = self.records.get('head', job['head_id'])
        if (head and head.get('active') == job['snapshot_id']) or (version and version.get('published_at')):
            raise ValueError('refuse to compensate a published snapshot')
        try:
            name = job.get('vector_collection')
            if name and self.vectors.has_collection(collection_name=name):
                ids = [r['_id'] for r in version['chunks']] if version else []
                for i in range(0, len(ids), 128):
                    self.vectors.delete(collection_name=name, ids=ids[i:i+128])
                self.vectors.flush(collection_name=name)
            self._stage('vectors_compensated', job)
            self._job_update(job, status='failed', phase='compensated')
        except Exception as exc:
            self._job_update(job, status='cleanup_failed', cleanup_error=f'{type(exc).__name__}: {exc}')
            raise

    def snapshot(self, sid, *, collection):
        version = self.records.get('version', sid)
        if not version or version['collection'] != collection or digest(version['content']) != version['hash']:
            raise ValueError('missing snapshot or content hash mismatch')
        head = self.records.get('head', version['head_id'])
        active = head and head.get('active') == sid and version['state'] == 'ready'
        status = 'active' if active else 'archived' if version.get('published_at') else 'pending'
        # Include an acknowledged head commit even if finalization is interrupted.
        if active and not version.get('published_at'):
            version['published_at'] = head['published_at']
        return {**version, 'status': status, 'metadata': json.dumps(version['metadata'], ensure_ascii=False)}

    def _eligible(self, collection, snapshot_id, as_of):
        if snapshot_id:
            version = self.snapshot(snapshot_id, collection=collection)
            return [version] if version['status'] != 'pending' else []
        heads = self.records.list('head', collection=collection)
        if not as_of:
            return [self.snapshot(h['active'], collection=collection) for h in heads]
        cutoff = datetime.fromisoformat(as_of.replace('Z', '+00:00'))
        if cutoff.tzinfo is None:
            raise ValueError('as_of requires timezone')
        versions = {v['_id']: v for v in self.records.list('version', collection=collection) if v.get('published_at')}
        for head in heads:
            if head['active'] not in versions:
                value = self.snapshot(head['active'], collection=collection)
                versions[value['_id']] = value
        selected = {}
        for version in versions.values():
            at = datetime.fromisoformat(version['published_at'])
            if at <= cutoff and (version['document_id'] not in selected or
                at > datetime.fromisoformat(selected[version['document_id']]['published_at'])):
                selected[version['document_id']] = version
        return [self.snapshot(v['_id'], collection=collection) for v in selected.values()]

    def _chunk_rows(self, versions):
        return sorted([r for v in versions for r in v['chunks']], key=lambda r: r['_id'])

    windows = KnowledgeStore.windows

    def jobs(self, *, collection='library'):
        return sorted(self.records.list('job', collection=collection), key=lambda r: r['created_at'], reverse=True)[:100]

    def reconcile(self, *, collection='library'):
        counts = {'published': 0, 'compensated': 0, 'busy': 0, 'errors': []}
        # The writer context first finalizes previously committed heads.
        try:
            with self._writer(repair=False):
                for job in self.records.list('job', collection=collection):
                    if job['status'] not in ['pending', 'cleanup_failed']:
                        continue
                    try:
                        version = self.records.get('version', job['snapshot_id'])
                        if job.get('kind') == 'index_rebuild':
                            if version.get('vector_collection') == job.get('vector_collection'):
                                self._job_update(job, status='completed', phase='index_published')
                                counts['published'] += 1
                            else:
                                self._cleanup_rebuild(job); counts['compensated'] += 1
                        else:
                            head = self.records.get('head', job['head_id'])
                            if (head and head['active'] == job['snapshot_id']) or (version and version.get('published_at')):
                                self._finish_publication(job); counts['published'] += 1
                            else:
                                self._compensate(job); counts['compensated'] += 1
                    except Exception as exc:
                        counts['errors'].append({'job_id': job['_id'], 'error': f'{type(exc).__name__}: {exc}'})
        except IngestionBusy:
            counts['busy'] += 1
        return counts

    def retry(self, job_id):
        with self._writer():
            job = self.records.get('job', job_id)
            if not job or job['status'] != 'failed':
                raise ValueError('only compensated failed jobs can be retried')
            if job.get('kind') == 'index_rebuild':
                raise ValueError('retry an index rebuild using knowledge reindex')
            version = self.records.get('version', job['snapshot_id'])
            if version:
                found = None if job.get('reparse') else self._duplicate(
                    job['collection'], job['document_id'], job['source_hash'], job['title'], version['metadata'])
                if found:
                    return found
                attempt = self._new_job(job['collection'], job['document_id'], job['uri'], job['title'],
                    source_hash=job['source_hash'], source_kind=job['source_kind'], metadata=version['metadata'], retry_of=job_id, reparse=job.get('reparse', False))
                return self._execute(attempt, version['content'])
            if job.get('source_path'):
                path = Path(job['source_path'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != job['source_hash']:
                    raise ValueError('source file changed; retry requires the original upload')
                return self.ingest_file(path, collection=job['collection'], document_id=job['document_id'],
                                    reparse=job.get('reparse', False))
            raise ValueError('failed job has no recoverable source')

    def _cleanup_rebuild(self, job):
        version = self.records.get('version', job['snapshot_id'])
        if version and version.get('vector_collection') == job.get('vector_collection'):
            raise ValueError('refuse to delete a published index projection')
        name = job.get('vector_collection')
        try:
            if name and self.vectors.has_collection(collection_name=name):
                self.vectors.drop_collection(collection_name=name)
            self._job_update(job, status='failed', phase='index_compensated')
        except Exception as exc:
            self._job_update(job, status='cleanup_failed', cleanup_error=str(exc))
            raise

    def reindex(self, *, collection='library'):
        count = 0
        with self._writer():
            for version in self.records.list('version', collection=collection):
                if not version.get('published_at'):
                    continue
                if digest(version['content']) != version['hash']:
                    raise ValueError('snapshot content hash mismatch')
                if len(version['chunks']) != version['expected_chunks']:
                    raise ValueError('canonical chunk set is incomplete')
                job = self._new_job(collection, version['document_id'], version['uri'], version['title'],
                                    kind='index_rebuild', snapshot_id=version['_id'])
                try:
                    projected = {**version, **self._index_projection(version['content'], version['chunks'])}
                    projected.update(self._write_index(projected, job))
                    self._job_update(job, phase='publishing_index')
                    self.records.put('version', version['_id'], projected)
                    self._stage('index_published', job)
                    self._job_update(job, status='completed', phase='index_published')
                    count += 1
                except Exception as exc:
                    self._job_update(job, error=f'{type(exc).__name__}: {exc}')
                    current = self.records.get('version', version['_id'])
                    if job['phase'] != 'publishing_index' and current.get('vector_collection') != job.get('vector_collection'):
                        self._cleanup_rebuild(job)
                    raise
        return {'reindexed_snapshots': count, 'model': self.provider.model_id, 'degraded': self.provider.degraded}

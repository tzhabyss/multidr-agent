"""Explicit, non-destructive Mongo/Milvus -> single-Milvus migration."""
from datetime import timezone
from .store import digest


def _timestamp(value):
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def migrate_collection(source, target, *, collection='library'):
    """Keep immutable snapshot/chunk IDs and capture times; re-encode vectors.

    Pause legacy writers for migration. Source records and collections are never
    deleted. A changed source head is detected and reported, not silently accepted.
    A failed run can be resumed; imported snapshots are verified before skipping.
    """
    heads = {h['_id']: h.get('active') for h in source.db.heads.find({'collection': collection})}
    versions = list(source.db.versions.find({'collection': collection, 'published_at': {'$exists': True}}))
    versions.sort(key=lambda v: _timestamp(v['published_at']))
    ids = {v['_id'] for v in versions}
    if any(sid and sid not in ids for sid in heads.values()):
        raise ValueError('legacy publication is incomplete; reconcile the source first')
    migrated, skipped = 0, 0
    with target._writer():
        for head in target.records.list('head', collection=collection):
            if head['active'] not in ids:
                raise ValueError('target contains independent versions; use an empty namespace/collection')
        for version in versions:
            content = version['content']
            if digest(content) != version['hash']:
                raise ValueError('legacy source hash mismatch')
            rows = list(source.db.chunks.find({'snapshot': version['_id']}).sort('start', 1))
            if len(rows) != version['expected_chunks'] or any(
                r['snapshot'] != version['_id'] or not 0 <= r['start'] < r['end'] <= len(content) for r in rows):
                raise ValueError('legacy chunk set is incomplete')
            existing = target.records.get('version', version['_id'])
            if existing and existing.get('published_at'):
                target.snapshot(existing['_id'], collection=collection)
                if (existing['hash'] != version['hash'] or existing['chunks'] != rows or
                    existing['document_id'] != version['document_id']):
                    raise ValueError('target snapshot conflicts with legacy source')
                skipped += 1
                continue
            imported = {**version, 'chunks': rows, 'published_at': _timestamp(version['published_at'])}
            job = target._new_job(collection, version['document_id'], version['uri'], version['title'],
                source_hash=version['source_hash'], source_kind='legacy_migration', metadata=version['metadata'],
                snapshot_id=version['_id'])
            target._execute(job, content, imported=imported)
            migrated += 1
        after = {h['_id']: h.get('active') for h in source.db.heads.find({'collection': collection})}
        if heads != after:
            raise ValueError('legacy source changed during migration; pause legacy writers and rerun')
        for hid, sid in heads.items():
            head = target.records.get('head', hid)
            if sid and (not head or head['active'] != sid):
                raise ValueError('migrated active version differs from legacy source')
        for upload in source.db.uploads.find({'collection': collection}):
            if upload['snapshot_id'] in ids:
                row = {k: v for k, v in upload.items() if k != '_id'}
                row.setdefault('document_id', '')
                target.records.put('upload', target._head_id(collection, row['file_hash']), row)
    return {'migrated_snapshots': migrated, 'verified_existing': skipped,
            'collection': collection, 'preserved_snapshot_ids': True}

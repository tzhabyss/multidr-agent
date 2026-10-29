"""Durable source/control records in Milvus 2.5 (no MongoDB or SQLite).

Large UTF-8 JSON values are copy-on-write pages. The small manifest is replaced
only after every page has been read back. Previous generations are retained so
an in-flight reader can finish. This is not a multi-row database transaction.
"""
from __future__ import annotations
import json
import uuid
import time
from .store import digest


class MilvusRecords:
    def __init__(self, client, namespace):
        from pymilvus import DataType
        self.client = client
        self.name = 'rag_records_' + digest(namespace)[:20]
        if not client.has_collection(collection_name=self.name):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=80)
            for field in ['kind', 'scope', 'document']:
                schema.add_field(field, DataType.VARCHAR, max_length=64)
            schema.add_field('payload', DataType.VARCHAR, max_length=65535)
            # Milvus 2.5 requires a vector field; control rows never use ANN.
            schema.add_field('reserved', DataType.FLOAT_VECTOR, dim=2)
            params = client.prepare_index_params()
            params.add_index(field_name='reserved', index_type='FLAT', metric_type='COSINE')
            client.create_collection(collection_name=self.name, schema=schema, index_params=params,
                                     consistency_level='Strong')

    @staticmethod
    def key(kind, key):
        return digest(json.dumps([kind, key], ensure_ascii=False))

    def _rows(self, ids):
        rows = []
        for i in range(0, len(ids), 128):
            rows.extend(self.client.get(collection_name=self.name, ids=ids[i:i+128],
                output_fields=['id', 'payload'], consistency_level='Strong'))
        return {row['id']: row['payload'] for row in rows}

    def _decode(self, envelope):
        header = json.loads(envelope)
        if 'value' in header:
            text = header['value']
        else:
            ids = header['pages']
            values = self._rows(ids)
            if set(values) != set(ids):
                raise ValueError('Milvus source record is incomplete')
            text = ''.join(values[key] for key in ids)
        if digest(text) != header['sha256']:
            raise ValueError('Milvus source record hash mismatch')
        return json.loads(text)

    def get(self, kind, key):
        rows = self._rows([self.key(kind, key)])
        return self._decode(next(iter(rows.values()))) if rows else None

    def put(self, kind, key, value):
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        rid = self.key(kind, key)
        scope = digest(value.get('collection', ''))
        document = digest(value.get('document_id', ''))
        def row(key, category, payload):
            return {'id': key, 'kind': category, 'scope': scope, 'document': document,
                    'payload': payload, 'reserved': [1., 0.]}
        header = {'sha256': digest(text), 'value': text}
        envelope = json.dumps(header, ensure_ascii=False, separators=(',', ':'))
        if len(envelope.encode('utf-8')) > 60000:
            # A Unicode code point uses at most 4 UTF-8 bytes: <=48k/page.
            generation = uuid.uuid4().hex
            pages = [row(f'{generation}:{i//12000}', 'blob', text[i:i+12000])
                     for i in range(0, len(text), 12000)]
            ids = [r['id'] for r in pages]
            for i in range(0, len(pages), 128):
                self.client.upsert(collection_name=self.name, data=pages[i:i+128])
            self.client.flush(collection_name=self.name)
            actual = self._rows(ids)
            if set(actual) != set(ids) or digest(''.join(actual[k] for k in ids)) != digest(text):
                raise ValueError('Milvus source pages failed round-trip validation')
            envelope = json.dumps({'sha256': digest(text), 'pages': ids}, separators=(',', ':'))
        if len(envelope.encode('utf-8')) > 65535:
            raise ValueError('Milvus source manifest exceeds the record limit')
        self.client.upsert(collection_name=self.name, data=[row(rid, kind, envelope)])
        self.client.flush(collection_name=self.name)
        actual = self._rows([rid])
        # Lite may briefly lag an acknowledged upsert even with Strong reads.
        # Retry confirmation only; never blindly repeat a publication write.
        for _ in range(5):
            if actual.get(rid) == envelope:
                break
            time.sleep(.02)
            actual = self._rows([rid])
        if actual.get(rid) != envelope:
            raise ValueError('Milvus record failed round-trip validation')

    def list(self, kind, *, collection=None, document_id=None):
        filters = ['kind == '+json.dumps(kind)]
        if collection is not None:
            filters.append('scope == '+json.dumps(digest(collection)))
        if document_id is not None:
            filters.append('document == '+json.dumps(digest(document_id)))
        iterator = self.client.query_iterator(collection_name=self.name, batch_size=128,
            filter=' and '.join(filters), output_fields=['id', 'payload'], consistency_level='Strong')
        try:
            result = []
            while True:
                batch = iterator.next()
                if not batch:
                    return result
                result.extend(self._decode(row['payload']) for row in batch)
        finally:
            iterator.close()

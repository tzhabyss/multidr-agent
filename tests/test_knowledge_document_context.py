import asyncio
import json

import pytest

from deepresearch.knowledge.document_context import DocumentContextIndex, describe_chunk, sha
from deepresearch.knowledge.dual_store import structured_chunks
from deepresearch.llm import LLMResponse


def context_cache(content, note='检索说明', *, parts=None):
    parts = parts or structured_chunks(content)
    return {'protocol': {'model': 'fixture', 'prompt_sha256': 'fixture'}, 'chunks': {
        str(i): {'document_sha256': sha(content), 'chunk_sha256': sha(content[p['start']:p['end']]),
                 'start': p['start'], 'end': p['end'], 'context': note, 'status': 'generated'}
        for i, p in enumerate(parts)}}


def test_cache_rejects_changed_boundaries_and_source_bytes():
    content = '冻结的原文。'
    cache = context_cache(content)
    index = DocumentContextIndex(cache)
    assert index.projection('其他版本。', structured_chunks('其他版本。')) == {}
    with pytest.raises(ValueError, match='boundaries'):
        index.projection(content, [{'start': 1, 'end': len(content)}])
    cache['chunks']['0']['chunk_sha256'] = sha('其他版本。')
    with pytest.raises(ValueError, match='hash mismatch'):
        DocumentContextIndex(cache).projection(content, structured_chunks(content))


def test_generator_sees_document_only_and_preserves_usage_on_invalid():
    class Fake:
        async def complete_json(self, system, user, *, max_tokens):
            self.value = json.loads(user)
            return {'context': 'x'*181}, LLMResponse('', input_tokens=10, output_tokens=20, finish_reason='stop')
    fake = Fake()
    content = '命令示例。'
    result = asyncio.run(describe_chunk(fake, content, structured_chunks(content)[0], title='说明'))
    assert set(fake.value) == {'title', 'document', 'heading', 'chunk'}
    assert result['status'] == 'invalid' and result['context'] == ''
    assert result['input_tokens'] == 10 and result['output_tokens'] == 20


def test_duplicate_chunks_and_oversized_descriptions_rejected():
    data = context_cache('原文。')
    data['chunks']['duplicate'] = dict(data['chunks']['0'])
    with pytest.raises(ValueError, match='duplicate'):
        DocumentContextIndex(data)
    with pytest.raises(ValueError, match='invalid retrieval'):
        DocumentContextIndex(context_cache('原文。', 'x'*181))


def test_factory_missing_optional_cache_uses_original_index(tmp_path, monkeypatch, caplog):
    pytest.importorskip('filelock')
    from deepresearch.knowledge import create_store
    from deepresearch.knowledge import milvus_store
    from deepresearch.memory_system.embeddings import HashEmbeddingProvider
    from deepresearch.runtime_config import KnowledgeRuntimeConfig
    monkeypatch.setattr(milvus_store, 'MilvusStore', lambda **kwargs: kwargs)
    config = KnowledgeRuntimeConfig(document_context_path=str(tmp_path/'not-published.json'))
    result = create_store(config, provider=HashEmbeddingProvider())
    assert result['document_context'] is None
    assert 'using original-text indexing' in caplog.text


def test_factory_does_not_hide_invalid_description_cache(tmp_path, monkeypatch):
    pytest.importorskip('filelock')
    from deepresearch.knowledge import create_store
    from deepresearch.knowledge import milvus_store
    from deepresearch.memory_system.embeddings import HashEmbeddingProvider
    from deepresearch.runtime_config import KnowledgeRuntimeConfig
    def forbidden(**kwargs):
        pytest.fail('invalid descriptions must fail before opening storage')
    monkeypatch.setattr(milvus_store, 'MilvusStore', forbidden)
    path = tmp_path/'invalid.json'; path.write_text('{invalid JSON')
    with pytest.raises(json.JSONDecodeError):
        create_store(KnowledgeRuntimeConfig(document_context_path=str(path)),
                     provider=HashEmbeddingProvider())

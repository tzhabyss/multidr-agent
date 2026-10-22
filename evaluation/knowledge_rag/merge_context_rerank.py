"""Combine the preserved three-route run with the explicitly adaptive fourth route."""
import hashlib
import json
from pathlib import Path

from .compare_context_rerank import ROUTES, summarize

OUT = Path(__file__).parent/'context_rerank'


def merge():
    original = OUT/'phase1-results.json'
    current = json.loads((OUT/'results.json').read_text())
    if 'phases' not in current['protocol']:
        original.write_bytes((OUT/'results.json').read_bytes())
        (OUT/'phase1-protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
    elif not original.exists():
        raise ValueError('preserved phase1 results are missing')
    first = json.loads(original.read_text())
    second_path = OUT/'followup/results.json'
    second = json.loads(second_path.read_text())
    selection = json.loads((OUT/'followup/selection-protocol.json').read_text())
    snapshot = OUT/'followup/selection-snapshot.json'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == selection['partial_results_sha256_at_selection']
    assert list(first['protocol']['routes']) == list(ROUTES)[:3]
    assert second['protocol']['routes'] == {'bge_neighbors': ROUTES['bge_neighbors']}
    for key in ['dataset_manifest', 'contexts_sha256', 'embedding', 'embedding_revision', 'model_revisions',
                'max_reranker_tokens', 'reranker_timeout_seconds', 'candidate_policy']:
        assert first['protocol'][key] == second['protocol'][key]
    assert len(first['cases']) == len(second['cases']) == 124
    for row, follow in zip(first['cases'], second['cases']):
        assert all(row[k] == follow[k] for k in ['id', 'query', 'category', 'group'])
        value = follow['routes']['bge_neighbors']
        neighbor = row['routes']['mini_neighbors']
        assert value['channels'] == neighbor['channels']
        assert [(c['query'], c['input'], c['scoring_text_sha256']) for c in value['calls']] == [
            (c['query'], c['input'], c['scoring_text_sha256']) for c in neighbor['calls']]
        row['routes']['bge_neighbors'] = value
    first['protocol']['routes'] = ROUTES
    first['protocol']['order'] = list(ROUTES)
    first['protocol']['phases'] = {
        'initial': {'routes': list(ROUTES)[:3], 'results_sha256': hashlib.sha256(original.read_bytes()).hexdigest()},
        'adaptive_followup': {'routes': ['bge_neighbors'], 'selection': selection,
                              'results_sha256': hashlib.sha256(second_path.read_bytes()).hexdigest()}}
    first['summary'] = {g: summarize([row for row in first['cases'] if row['group'] == g]) for g in first['summary']}
    first['followup_model_load_seconds'] = second['model_load_seconds']
    (OUT/'results.json').write_text(json.dumps(first, ensure_ascii=False, indent=2)+'\n')
    (OUT/'protocol.json').write_text(json.dumps(first['protocol'], ensure_ascii=False, indent=2)+'\n')
    print('Merged four full124-case routes, with phase1 and follow-up results preserved.')


if __name__ == '__main__':
    merge()

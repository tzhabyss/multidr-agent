"""Evaluation correctness tests; never tune retrieval against the benchmark."""
import json
from collections import Counter

import pytest

from evaluation.knowledge_rag.metrics_v3 import score
from evaluation.knowledge_rag.run_v3 import DATA, load_dataset


def span(document,start,end):
    return {'document':document,'start':start,'end':end}


def test_alternative_valid_passage_is_not_a_false_negative():
    groups=[[span('doc',0,10),span('doc',100,110)]]
    result=score(groups,[span('doc',100,120)])
    assert result['hit_at_5']==1
    assert result['mrr_at_5']==1


def test_known_evidence_can_span_adjacent_retrieved_chunks():
    assert score([[span('doc',5,15)]],[span('doc',0,10),span('doc',10,20)])['hit_at_5']==1
    assert score([[span('doc',5,15)]],[span('doc',0,9),span('doc',10,20)])['hit_at_5']==0


def test_multi_fact_question_is_not_complete_after_one_source():
    result=score([[span('a',0,10)],[span('b',0,10)]],[span('a',0,20)])
    assert result['hit_at_5']==1
    assert result['evidence_group_coverage_at_5']==.5
    assert result['all_groups_at_5']==0


def test_wrong_source_or_rank_beyond_five_is_not_support():
    assert score([[span('a',0,10)]],[span('b',0,20)])['hit_at_5']==0
    hits=[span('b',0,20)]*5+[span('a',0,20)]
    assert score([[span('a',0,10)]],hits)['hit_at_5']==0


def test_no_answer_case_does_not_get_a_fake_perfect_score():
    assert score([],[])=={'candidate_nonempty':0}
    assert score([],[span('a',0,20)])=={'candidate_nonempty':1}


def test_frozen_dataset_has_more_than_a_hundred_unique_questions():
    manifest,docs,cases=load_dataset()
    assert len(cases)==124
    assert Counter(c['group'] for c in cases)=={'repository':72,'synthetic':32,'multi_source':8,'negative':12}
    assert len(docs)==26
    assert all(c.get('as_of_snapshot') in {d['id'] for d in docs} for c in cases if c.get('as_of_snapshot'))
    deficit=next(c for c in cases if c['query']=='一轮研究发现资料不足时怎么办？')
    assert len(deficit['evidence_groups'][0])>=2


def test_dataset_hash_rejects_silent_relabelling(tmp_path):
    for name in ['corpus.json','cases.jsonl','manifest.json']:
        (tmp_path/name).write_bytes((DATA/name).read_bytes())
    with (tmp_path/'cases.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_dataset(tmp_path)

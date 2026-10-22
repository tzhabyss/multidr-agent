import pytest

from deepresearch.knowledge.candidate_selection import quota_targets, select_candidates


def test_quotas_keep_both_channels_within_same_budget():
    lexical = [f'l{i}' for i in range(20)]
    semantic = [f's{i}' for i in range(40)]
    fused = lexical+semantic
    assert select_candidates(lexical, semantic, fused, budget=20)==lexical
    selected = select_candidates(lexical, semantic, fused, budget=20, policy='quota_equal')
    assert selected==lexical[:10]+semantic[:10]
    assert len(selected)==20


def test_shared_candidates_count_for_both_reservations_and_slots_are_refilled():
    common = [f'c{i}' for i in range(8)]
    lexical = common+[f'l{i}' for i in range(12)]
    semantic = common+[f's{i}' for i in range(32)]
    fused = list(dict.fromkeys(lexical+semantic))
    selected = select_candidates(lexical, semantic, fused, budget=20, policy='quota_equal')
    assert len(selected)==len(set(selected))==20
    assert set(lexical[:10]+semantic[:10]).issubset(selected)
    assert selected==[cid for cid in fused if cid in set(selected)]


@pytest.mark.parametrize('policy', ['quota_equal','quota_dense','quota_lexical'])
def test_exhausted_channel_and_small_corpus_do_not_leave_avoidable_empty_slots(policy):
    semantic = [f's{i}' for i in range(40)]
    assert select_candidates([], semantic, semantic, budget=20, policy=policy)==semantic[:20]
    assert select_candidates(['a'], ['a','b'], ['a','b'], budget=20, policy=policy)==['a','b']
    assert select_candidates(['a'], ['a','b'], ['a','b'], budget=0, policy=policy)==[]


def test_invalid_policy_or_outside_candidate_is_rejected():
    with pytest.raises(ValueError):
        select_candidates([], [], [], budget=20, policy='unknown')
    with pytest.raises(ValueError):
        select_candidates(['invented'], [], ['real'], budget=20, policy='quota_equal')
    with pytest.raises(ValueError):
        quota_targets(-1, 'quota_equal')


def test_reservation_ratios_and_worker_pool_targets_are_explicit():
    assert quota_targets(20, 'quota_equal')==(10,10)
    assert quota_targets(20, 'quota_dense')==(5,15)
    assert quota_targets(20, 'quota_lexical')==(15,5)
    assert quota_targets(40, 'quota_equal')==(20,20)
    assert quota_targets(40, 'quota_lexical')==(30,10)


def test_unsupported_legacy_backend_rejects_quotas_instead_of_ignoring_them():
    from deepresearch.knowledge import create_store
    from deepresearch.runtime_config import KnowledgeRuntimeConfig
    config = KnowledgeRuntimeConfig(backend='sqlite', candidate_policy='quota_equal')
    with pytest.raises(ValueError, match='mongo_milvus'):
        create_store(config)

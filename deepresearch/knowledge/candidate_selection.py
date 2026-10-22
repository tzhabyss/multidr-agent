"""Reserve per-channel candidates while keeping a fixed reranking budget."""

POLICIES = {'rrf': None, 'quota_equal': (1, 2),
            'quota_dense': (1, 4), 'quota_lexical': (3, 4)}


def quota_targets(budget, policy):
    if policy not in POLICIES:
        raise ValueError('unknown candidate policy')
    if budget < 0:
        raise ValueError('candidate budget cannot be negative')
    ratio = POLICIES[policy]
    if ratio is None:
        return 0, 0
    lexical = budget*ratio[0]//ratio[1]
    return lexical, budget-lexical


def select_candidates(lexical, semantic, fused, *, budget, policy='rrf'):
    """Reserve route prefixes, deduplicate, fill by RRF, preserve RRF order.

    A shared candidate satisfies both route-prefix reservations. An exhausted
    route does not consume empty slots. No source/rank labels are consulted.
    """
    lexical_quota, semantic_quota = quota_targets(budget, policy)
    fused = list(dict.fromkeys(fused))
    if policy == 'rrf':
        return fused[:budget]
    allowed = set(fused)
    lexical, semantic = list(dict.fromkeys(lexical)), list(dict.fromkeys(semantic))
    if not set([*lexical, *semantic]).issubset(allowed):
        raise ValueError('route contains a candidate outside the fused set')
    chosen = set([*lexical[:lexical_quota], *semantic[:semantic_quota]])
    for cid in fused:
        if len(chosen) >= budget:
            break
        chosen.add(cid)
    return [cid for cid in fused if cid in chosen]

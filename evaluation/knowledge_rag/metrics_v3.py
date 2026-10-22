"""Scoring against known alternative evidence spans; no claim of exhaustive relevance."""
from __future__ import annotations


def covered(alternative, hits):
    spans=sorted((h['start'],h['end']) for h in hits if h['document']==alternative['document'])
    cursor=alternative['start']
    for start,end in spans:
        if end<=cursor:continue
        if start>cursor:break
        cursor=max(cursor,end)
        if cursor>=alternative['end']:return True
    return False


def score(groups, hits, k=5):
    if not groups:
        return {'candidate_nonempty':int(bool(hits[:k]))}
    selected=hits[:k]
    satisfied=[any(covered(a,selected) for a in group) for group in groups]
    first=next((rank for rank in range(1,len(selected)+1)
                if any(any(covered(a,selected[:rank]) for a in group) for group in groups)),None)
    return {'hit_at_5':int(any(satisfied)),
            'mrr_at_5':1/first if first else 0,
            'evidence_group_coverage_at_5':sum(satisfied)/len(groups),
            'all_groups_at_5':int(all(satisfied))}

"""Fresh authoritative checks at the final citation boundary."""
from .store import digest


def invalid_library_sources(store, sources, snapshots):
    by_source = {s.source_id: s for s in snapshots}
    invalid = set()
    for source in sources:
        reference = source.provenance.get('knowledge_ref')
        if not reference:
            continue
        try:
            current = store.snapshot(reference['snapshot_id'], collection=reference['collection'])
            original = by_source[source.source_id]
            if current['status'] != 'active' or digest(original.content) != current['hash']:
                invalid.add(source.source_id)
        except Exception:
            invalid.add(source.source_id)
    return invalid


def exclude_stale_citations(output, stale):
    sections = []
    for section in output.sections:
        paragraphs = [p for p in section.paragraphs if not stale.intersection(p.citation_ids)]
        tables = []
        for table in section.tables:
            rows = [row for row in table.rows if not any(stale.intersection(cell.citation_ids) for cell in row)]
            tables.append(table.model_copy(update={'rows': rows}))
        sections.append(section.model_copy(update={'paragraphs': paragraphs, 'tables': tables}))
    return output.model_copy(update={'sections': sections})

"""Source-linked claim construction and report rendering."""

from __future__ import annotations

from urllib.parse import urlsplit

from .ids import make_id, normalize_text, stable_hash
from .retrieval import independent_domain
from .schemas import (
    CitationRecord,
    ClaimEvidenceLink,
    ClaimRecord,
    EvidenceRecord,
    FetchedDocument,
    GateResult,
    OutputParagraph,
    OutputSection,
    OutputTable,
    OutputTableCell,
    ResearchMode,
    RunStatus,
    SourceRecord,
    SourceSnapshot,
    StructuredOutput,
)


def quality_for_url(url: str) -> tuple[int, str]:
    host = (urlsplit(url).hostname or "").lower()
    path = urlsplit(url).path.lower()
    if host.endswith((".gov", ".gov.cn", ".edu", ".edu.cn", ".int")):
        return 1, "official or institutional domain"
    if any(marker in host for marker in ("wikipedia.org", "github.com", "arxiv.org")):
        return 2, "identified public knowledge or primary project source"
    low_control_hosts = (
        "bilibili.com",
        "douyin.com",
        "youtube.com",
        "youtu.be",
        "reddit.com",
        "quora.com",
        "zhihu.com",
        "weibo.com",
        "x.com",
        "twitter.com",
        "instagram.com",
        "facebook.com",
        "fb.com",
        "linkedin.com",
        "medium.com",
        "pinterest.com",
        "slideshare.net",
        "scribd.com",
    )
    if any(host == item or host.endswith("." + item) for item in low_control_hosts):
        return 4, "social, video, or community-published page"
    if host.startswith(("bbs.", "forum.")) or any(
        marker in path for marker in ("/bbs/", "/forum/", "/thread/")
    ):
        return 4, "forum or community discussion page"
    if "/search" in path or "/activity/" in path:
        return 4, "search or user-submitted activity page"
    return 3, "public web source with fetched full text"


def records_from_document(
    document: FetchedDocument,
    provider: str,
    extracted_claims: list[dict],
    requirement_ids: list[str] | None = None,
    published_at: str | None = None,
) -> tuple[SourceRecord, SourceSnapshot, list[EvidenceRecord], list[ClaimRecord], list[ClaimEvidenceLink]]:
    content_hash = stable_hash(document.content)
    is_local = urlsplit(document.final_url).scheme == "file"
    source_kind = "local_document" if is_local else "web_page"
    source_id = make_id("src", source_kind, document.final_url, content_hash)
    tier, reason = (
        (3, "user-provided local document with extracted text")
        if is_local
        else quality_for_url(document.final_url)
    )
    source = SourceRecord(
        source_id=source_id,
        source_kind=source_kind,
        title=document.title,
        canonical_uri=document.final_url,
        provider=provider,
        published_at=published_at,
        fetched_at=document.fetched_at,
        fetch_status="full",
        http_status=document.http_status,
        mime_type=document.content_type,
        content_hash=content_hash,
        quality_tier=tier,
        quality_reason=reason,
        provenance={
            "redirect_chain": document.redirect_chain,
            "independent_domain": independent_domain(document.final_url),
            "local_path": urlsplit(document.final_url).path if is_local else "",
            "retrieval_method": document.retrieval_method,
            **({"knowledge_ref": document.knowledge_ref} if document.knowledge_ref else {}),
        },
    )
    snapshot = SourceSnapshot(source_id=source_id, content=document.content, content_hash=content_hash)
    evidence: list[EvidenceRecord] = []
    claims: list[ClaimRecord] = []
    links: list[ClaimEvidenceLink] = []
    allowed_requirement_ids = set(requirement_ids or [])
    accepted_claims, _ = filter_extracted_claims(document, extracted_claims)

    for index, item in enumerate(accepted_claims):
        text = normalize_text(str(item.get("text") or ""))
        quote = normalize_text(str(item.get("quote") or ""))
        item_requirement_ids = [
            str(value)
            for value in item.get("requirement_ids", [])
            if str(value) in allowed_requirement_ids
        ]
        if not item_requirement_ids and len(allowed_requirement_ids) == 1:
            item_requirement_ids = sorted(allowed_requirement_ids)
        location_kind = "local_text" if is_local else "web_text"
        evidence_id = make_id("ev", source_id, stable_hash(quote), f"{location_kind}:{index}")
        claim_id = make_id("clm", normalize_text(text).lower())
        link_id = make_id("lnk", claim_id, evidence_id, "supports")
        evidence.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=source_id,
                quote=quote,
                quote_hash=stable_hash(quote),
                location={"kind": location_kind, "locator": f"claim-{index + 1}"},
                requirement_ids=item_requirement_ids,
            )
        )
        claims.append(
            ClaimRecord(
                claim_id=claim_id,
                text=text,
                impact=item.get("impact", "medium"),
                claim_type=item.get("claim_type", "factual"),
                status="supported",
                requirement_ids=item_requirement_ids,
            )
        )
        links.append(
            ClaimEvidenceLink(
                link_id=link_id,
                claim_id=claim_id,
                evidence_id=evidence_id,
                relation="supports",
                entailment_status="verified",
                verification_method="reviewer",
            )
        )
    return source, snapshot, evidence, claims, links


def filter_extracted_claims(
    document: FetchedDocument,
    extracted_claims: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Normalize extraction output without applying semantic hard gates.

    Missing text/passage is malformed output and is dropped. Passage mismatch
    and number/date normalization are retained as review warnings rather than
    silently deleting a useful claim.
    """
    normalized_content = normalize_text(document.content)
    accepted: list[dict] = []
    rejected: list[dict] = []
    for index, item in enumerate(extracted_claims):
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "candidate_is_not_an_object"})
            continue
        text = normalize_text(str(item.get("text") or ""))
        quote = normalize_text(str(item.get("quote") or ""))
        if not text:
            rejected.append({"index": index, "reason": "missing_claim_text"})
            continue
        if not quote:
            rejected.append(
                {"index": index, "reason": "missing_quote", "text": text[:500]}
            )
            continue
        warning: dict[str, object] | None = None
        if quote not in normalized_content:
            warning = {
                "index": index,
                "reason": "supporting_context_not_exact_substring",
                "text": text[:500],
                "quote": quote[:1000],
            }
        if warning:
            rejected.append(
                {**warning, "action": "review_warning"}
            )
        accepted.append(
            {
                "text": text,
                "quote": quote,
                "impact": item.get("impact", "medium"),
                "claim_type": item.get("claim_type", "factual"),
                "requirement_ids": [
                    str(value)
                    for value in item.get("requirement_ids", [])
                    if str(value).strip()
                ],
            }
        )
    return accepted, rejected


def build_structured_output(
    *,
    run_id: str,
    mode: ResearchMode,
    question: str,
    claims: list[ClaimRecord],
    links: list[ClaimEvidenceLink],
    gaps: list[str],
    assumptions: list[str],
    preliminary_status: RunStatus,
) -> tuple[StructuredOutput, list[CitationRecord]]:
    link_by_claim = {
        link.claim_id: link
        for link in links
        if link.relation == "supports" and link.entailment_status == "verified"
    }
    citations: list[CitationRecord] = []
    paragraphs: list[OutputParagraph] = []
    for index, claim in enumerate(claims):
        link = link_by_claim.get(claim.claim_id)
        if link is None:
            continue
        section_id = "findings"
        citation_id = make_id("cit", section_id, claim.claim_id, link.link_id)
        citations.append(
            CitationRecord(
                citation_id=citation_id,
                section_id=section_id,
                claim_id=claim.claim_id,
                link_id=link.link_id,
            )
        )
        paragraphs.append(
            OutputParagraph(
                paragraph_id=make_id("par", run_id, section_id, index),
                text=claim.text,
                claim_ids=[claim.claim_id],
                citation_ids=[citation_id],
                requirement_ids=claim.requirement_ids,
                paragraph_type="factual",
            )
        )

    direct_answer = None
    sections = [OutputSection(section_id="findings", title="核心发现", paragraphs=paragraphs)]

    if gaps:
        sections.append(
            OutputSection(
                section_id="uncertainty",
                title="不确定性与待验证项",
                paragraphs=[
                    OutputParagraph(
                        paragraph_id=make_id("par", run_id, "gap", index),
                        text=gap,
                        paragraph_type="uncertainty",
                    )
                    for index, gap in enumerate(dict.fromkeys(gaps))
                ],
            )
        )

    return (
        StructuredOutput(
            run_id=run_id,
            mode=mode,
            title=question,
            direct_answer=direct_answer,
            sections=sections,
            unresolved_gaps=list(dict.fromkeys(gaps)),
            assumptions=assumptions,
            run_status=preliminary_status,
        ),
        citations,
    )


def validate_citations(
    output: StructuredOutput,
    sources: list[SourceRecord],
    snapshots: list[SourceSnapshot],
    evidence: list[EvidenceRecord],
    claims: list[ClaimRecord],
    links: list[ClaimEvidenceLink],
    citations: list[CitationRecord],
    *,
    require_all_factual_citations: bool = True,
) -> GateResult:
    source_map = {item.source_id: item for item in sources}
    snapshot_map = {item.source_id: item for item in snapshots}
    evidence_map = {item.evidence_id: item for item in evidence}
    claim_map = {item.claim_id: item for item in claims}
    link_map = {item.link_id: item for item in links}
    citation_map = {item.citation_id: item for item in citations}
    violations: list[str] = []
    unsupported: set[str] = set()
    grounded: list[str] = []

    acceptable_claims: set[str] = set()
    for link in links:
        if link.relation != "supports" or link.entailment_status != "verified":
            continue
        claim = claim_map.get(link.claim_id)
        item = evidence_map.get(link.evidence_id)
        if claim is None or item is None:
            violations.append(f"Broken link {link.link_id}")
            continue
        source = source_map.get(item.source_id)
        snapshot = snapshot_map.get(item.source_id)
        if source is None or snapshot is None:
            violations.append(f"Evidence {item.evidence_id} has no source snapshot")
            unsupported.add(claim.claim_id)
            continue
        acceptable_claims.add(claim.claim_id)

    for section in output.sections:
        for paragraph in section.paragraphs:
            if paragraph.paragraph_type != "factual":
                continue
            if not paragraph.claim_ids and require_all_factual_citations:
                violations.append(f"Factual paragraph has no claims: {paragraph.paragraph_id}")
            if not paragraph.citation_ids and (
                require_all_factual_citations or paragraph.claim_ids
            ):
                violations.append(f"Factual paragraph has no citations: {paragraph.paragraph_id}")
            for claim_id in paragraph.claim_ids:
                if claim_id not in acceptable_claims:
                    unsupported.add(claim_id)
                    violations.append(f"Unsupported factual claim: {claim_id}")
            cited_claims: set[str] = set()
            for citation_id in paragraph.citation_ids:
                citation = citation_map.get(citation_id)
                if citation is None:
                    violations.append(f"Unknown citation: {citation_id}")
                    continue
                link = link_map.get(citation.link_id)
                if (
                    link is None
                    or citation.claim_id not in paragraph.claim_ids
                    or link.claim_id != citation.claim_id
                    or citation.section_id != section.section_id
                ):
                    violations.append(f"Citation does not resolve: {citation_id}")
                    continue
                if citation.claim_id in acceptable_claims:
                    grounded.append(citation_id)
                    cited_claims.add(citation.claim_id)
            for claim_id in paragraph.claim_ids:
                if claim_id not in cited_claims:
                    unsupported.add(claim_id)
                    violations.append(f"Factual claim has no resolving citation: {claim_id}")
        for table in section.tables:
            if not table.columns:
                violations.append(f"Table has no columns: {table.table_id}")
            for row_index, row in enumerate(table.rows):
                if len(row) != len(table.columns):
                    violations.append(
                        f"Table row width mismatch: {table.table_id} row {row_index + 1}"
                    )
                    continue
                for cell_index, cell in enumerate(row):
                    cell_label = cell.cell_id or f"{table.table_id}:{row_index}:{cell_index}"
                    if cell.cell_type == "uncertainty":
                        if cell.claim_ids or cell.citation_ids:
                            violations.append(
                                f"Uncertainty table cell has evidence references: {cell_label}"
                            )
                        continue
                    if not cell.claim_ids and require_all_factual_citations:
                        violations.append(f"Factual table cell has no claims: {cell_label}")
                    if not cell.citation_ids and (
                        require_all_factual_citations or cell.claim_ids
                    ):
                        violations.append(f"Factual table cell has no citations: {cell_label}")
                    cited_claims: set[str] = set()
                    for claim_id in cell.claim_ids:
                        if claim_id not in acceptable_claims:
                            unsupported.add(claim_id)
                            violations.append(f"Unsupported factual claim: {claim_id}")
                    for citation_id in cell.citation_ids:
                        citation = citation_map.get(citation_id)
                        if citation is None:
                            violations.append(f"Unknown citation: {citation_id}")
                            continue
                        link = link_map.get(citation.link_id)
                        if (
                            link is None
                            or citation.claim_id not in cell.claim_ids
                            or link.claim_id != citation.claim_id
                            or citation.section_id != section.section_id
                        ):
                            violations.append(f"Citation does not resolve: {citation_id}")
                            continue
                        if citation.claim_id in acceptable_claims:
                            grounded.append(citation_id)
                            cited_claims.add(citation.claim_id)
                    for claim_id in cell.claim_ids:
                        if claim_id not in cited_claims:
                            unsupported.add(claim_id)
                            violations.append(
                                f"Factual table claim has no resolving citation: {claim_id}"
                            )

    for claim in claims:
        if claim.impact == "high" and claim.claim_id not in acceptable_claims:
            unsupported.add(claim.claim_id)
            violations.append(f"High-impact claim lacks verified evidence: {claim.claim_id}")

    return GateResult(
        passed=(
            not violations
            and (bool(grounded) if require_all_factual_citations else True)
        ),
        violations=list(dict.fromkeys(violations)),
        unsupported_claim_ids=sorted(unsupported),
        grounded_citation_ids=list(dict.fromkeys(grounded)),
    )


def apply_citation_gate(output: StructuredOutput, gate: GateResult) -> StructuredOutput:
    """Compatibility helper for callers that explicitly request strict filtering."""
    grounded = set(gate.grounded_citation_ids)
    sections: list[OutputSection] = []
    for section in output.sections:
        paragraphs: list[OutputParagraph] = []
        tables: list[OutputTable] = []
        for paragraph in section.paragraphs:
            if paragraph.paragraph_type != "factual":
                paragraphs.append(paragraph)
                continue
            citation_ids = [item for item in paragraph.citation_ids if item in grounded]
            if (
                not paragraph.claim_ids
                or any(item in gate.unsupported_claim_ids for item in paragraph.claim_ids)
                or len(citation_ids) != len(paragraph.citation_ids)
                or not citation_ids
            ):
                continue
            paragraphs.append(paragraph.model_copy(update={"citation_ids": citation_ids}))
        for table in section.tables:
            rows: list[list[OutputTableCell]] = []
            for row in table.rows:
                output_row: list[OutputTableCell] = []
                for cell in row:
                    if cell.cell_type == "uncertainty":
                        if cell.claim_ids or cell.citation_ids:
                            output_row = []
                            break
                        output_row.append(cell)
                        continue
                    citation_ids = [item for item in cell.citation_ids if item in grounded]
                    if (
                        not cell.claim_ids
                        or any(item in gate.unsupported_claim_ids for item in cell.claim_ids)
                        or len(citation_ids) != len(cell.citation_ids)
                        or not citation_ids
                    ):
                        output_row = []
                        break
                    output_row.append(cell.model_copy(update={"citation_ids": citation_ids}))
                if output_row and len(output_row) == len(table.columns):
                    rows.append(output_row)
            if rows:
                tables.append(table.model_copy(update={"rows": rows}))
        if paragraphs or tables:
            sections.append(
                section.model_copy(update={"paragraphs": paragraphs, "tables": tables})
            )

    factual_text = [
        paragraph.text
        for section in sections
        for paragraph in section.paragraphs
        if paragraph.paragraph_type == "factual"
    ]
    return output.model_copy(update={"sections": sections, "direct_answer": None})


def render_markdown(
    output: StructuredOutput,
    citations: list[CitationRecord],
    links: list[ClaimEvidenceLink],
    evidence: list[EvidenceRecord],
    sources: list[SourceRecord],
    *,
    include_audit_links: bool = True,
) -> str:
    citation_map = {item.citation_id: item for item in citations}
    link_map = {item.link_id: item for item in links}
    evidence_map = {item.evidence_id: item for item in evidence}
    source_map = {item.source_id: item for item in sources}
    ordered_source_uris: list[str] = []
    source_by_uri: dict[str, SourceRecord] = {}

    def marker(citation_id: str) -> str | None:
        citation = citation_map.get(citation_id)
        if citation is None:
            return None
        link = link_map.get(citation.link_id)
        if link is None:
            return None
        item = evidence_map.get(link.evidence_id)
        if item is None:
            return None
        source = source_map.get(item.source_id)
        if source is None:
            return None
        uri = source.canonical_uri
        source_by_uri.setdefault(uri, source)
        if uri not in ordered_source_uris:
            ordered_source_uris.append(uri)
        return f"[{ordered_source_uris.index(uri) + 1}]"

    def markers_for(citation_ids: list[str]) -> list[str]:
        resolved = (marker(item) for item in citation_ids)
        return list(dict.fromkeys(item for item in resolved if item is not None))

    lines = [f"# {output.title}", "", f"运行状态：`{output.run_status.value}`", ""]
    if output.direct_answer:
        lines.extend(["## 直接答案", "", output.direct_answer, ""])
    for section in output.sections:
        # Evidence limits stay in the structured output and audit artifacts. They
        # are operational metadata, not part of the reader-facing report body.
        if section.section_id == "evidence_limits":
            continue
        lines.extend([f"## {section.title}", ""])
        for paragraph in section.paragraphs:
            markers = markers_for(paragraph.citation_ids)
            suffix = "".join(markers)
            lines.extend([paragraph.text + (f" {suffix}" if suffix else ""), ""])
        for table in section.tables:
            if table.title:
                lines.extend([f"### {table.title}", ""])
            escaped_columns = [item.replace("|", "\\|").replace("\n", "<br>") for item in table.columns]
            lines.append("| " + " | ".join(escaped_columns) + " |")
            lines.append("| " + " | ".join("---" for _ in escaped_columns) + " |")
            for row in table.rows:
                rendered_cells: list[str] = []
                for cell in row:
                    markers = markers_for(cell.citation_ids)
                    text = cell.text.replace("|", "\\|").replace("\n", "<br>")
                    suffix = "".join(markers)
                    rendered_cells.append(text + (f" {suffix}" if suffix else ""))
                lines.append("| " + " | ".join(rendered_cells) + " |")
            lines.append("")
    lines.extend(["## Sources", ""])
    for index, uri in enumerate(ordered_source_uris, 1):
        source = source_by_uri[uri]
        lines.append(f"[{index}] [{source.title}]({source.canonical_uri})，抓取时间：{source.fetched_at}")
    if include_audit_links:
        lines.extend(
            [
                "",
                "## 研究审计",
                "",
                "[研究审计报告](details/audit/audit-report.md) · [结构化审计日志](details/audit/audit-log.json)",
            ]
        )
    return "\n".join(lines).strip() + "\n"

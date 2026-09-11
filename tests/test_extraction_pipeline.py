from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.evidence import filter_extracted_claims
from deepresearch.schemas import EvidenceExtractionResult, FetchedDocument, RunStatus, SearchHit
from deepresearch.service import ResearchService

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


def test_extraction_schema_accepts_common_provider_aliases():
    result = EvidenceExtractionResult.model_validate(
        {
            "factual_claims": [
                {
                    "statement": "The verified fact is 42 units.",
                    "source_quote": "The verified fact is 42 units.",
                    "impact": "HIGH",
                }
            ],
            "gaps": "No remaining gap",
            "provider_comment": "ignored",
        }
    )

    assert result.claims[0].text == "The verified fact is 42 units."
    assert result.claims[0].quote == "The verified fact is 42 units."
    assert result.claims[0].impact == "high"
    assert result.knowledge_gaps == ["No remaining gap"]


def test_extraction_schema_keeps_claims_when_strategy_contains_artifact_role():
    result = EvidenceExtractionResult.model_validate(
        {
            "claims": [
                {
                    "text": "Python 3.13.0 was released on October 7, 2024.",
                    "quote": "October 7, 2024",
                    "impact": "high",
                }
            ],
            "artifact_role": "source_document",
            "extraction_strategy": "field_evidence",
        }
    )

    assert len(result.claims) == 1
    assert result.extraction_strategy == "none"


@pytest.mark.parametrize("scope_status", ["", None, "UNKNOWN_SCOPE"])
def test_extraction_schema_normalizes_invalid_entity_scope_without_losing_claim(
    scope_status,
):
    result = EvidenceExtractionResult.model_validate(
        {
            "claims": [
                {
                    "text": "The verified fact is 42 units.",
                    "quote": "The verified fact is 42 units.",
                    "entity_scope_status": scope_status,
                }
            ]
        }
    )

    assert len(result.claims) == 1
    assert result.claims[0].entity_scope_status == "uncertain"


def test_numeric_date_format_difference_is_not_a_hard_rejection():
    document = FetchedDocument(
        requested_url="https://example.com/source",
        final_url="https://example.com/source",
        title="Source",
        content="Python 3.13 adds a Just-In-Time compiler.",
        content_type="text/plain",
        http_status=200,
    )

    accepted, rejected = filter_extracted_claims(
        document,
        [
            {
                "text": "Python 3.13 adds a JIT compiler.",
                "quote": "a Just-In-Time compiler",
                "impact": "high",
            }
        ],
    )

    assert accepted[0]["text"] == "Python 3.13 adds a JIT compiler."
    assert rejected == []


class LongDocumentFetcher(FakeFetcher):
    async def fetch(self, url: str):
        self.calls.append(url)
        content = "The verified fact is 42 units.\n" + ("Additional background context. " * 2_000)
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title="Long verified source",
            content=content,
            content_type="text/plain",
            http_status=200,
        )


class DynamicRetryFetcher(FakeFetcher):
    async def fetch(self, url: str, render: bool = False):
        self.calls.append((url, render))
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title="Dynamic source",
            content=(
                "The verified fact is 42 units. Additional rendered context."
                if render
                else "Loading application data..."
            ),
            content_type="text/html",
            http_status=200,
            retrieval_method="scrapling_dynamic" if render else "direct",
        )


class MultiHitSearchBackend:
    name = "multi_hit_search"

    async def search(self, query: str, max_results: int = 5):
        return [
            SearchHit(
                title=f"Source {index}",
                url=f"https://source-{index}.example/research",
                snippet="The verified fact is 42 units.",
                provider=self.name,
                rank=index,
            )
            for index in range(4)
        ]


class LargeMultiHitSearchBackend:
    name = "large_multi_hit_search"

    async def search(self, query: str, max_results: int = 5):
        return [
            SearchHit(
                title=f"Source {index}",
                url=f"https://source-{index}.example/research",
                snippet="The verified fact is 42 units.",
                provider=self.name,
                rank=index,
            )
            for index in range(12)
        ]


class ConcurrentFetcher(FakeFetcher):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.peak = 0
        self.two_started = asyncio.Event()

    async def fetch(self, url: str):
        self.calls.append(url)
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.peak >= 2:
            self.two_started.set()
        try:
            await asyncio.wait_for(self.two_started.wait(), timeout=0.5)
            return FetchedDocument(
                requested_url=url,
                final_url=url,
                title="Concurrent verified source",
                content="The verified fact is 42 units. Additional context is available.",
                content_type="text/plain",
                http_status=200,
            )
        finally:
            self.active -= 1


class PipelineFetcher(FakeFetcher):
    def __init__(self):
        super().__init__()
        self.next_batch_started = asyncio.Event()

    async def fetch(self, url: str):
        self.calls.append(url)
        if "source-8.example" in url:
            self.next_batch_started.set()
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title="Pipelined verified source",
            content="The verified fact is 42 units. Additional context is available.",
            content_type="text/plain",
            http_status=200,
        )


class PipelineLLM(FakeLLM):
    def __init__(self, next_batch_started: asyncio.Event):
        super().__init__()
        self.next_batch_started = next_batch_started
        self.first_extraction = True
        self.next_batch_started_before_first_extraction = False

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        payload = json.loads(user)
        if "untrusted_content" in payload and self.first_extraction:
            self.first_extraction = False
            self.next_batch_started_before_first_extraction = (
                self.next_batch_started.is_set()
            )
        return await super().complete_json(system, user, max_tokens=max_tokens)


@pytest.mark.asyncio
async def test_worker_prefetches_two_candidates_through_shared_queue(tmp_path: Path):
    fetcher = ConcurrentFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=MultiHitSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )

    async with service:
        await service.run(
            {
                "question": "What is the verified fact?",
                "mode": "standard",
                "run_id": "concurrent-fetch-run",
                "options": {"max_rounds": 1, "max_workers": 1},
            }
        )

    assert fetcher.peak >= 2


@pytest.mark.asyncio
async def test_worker_delays_next_prefetch_batch_until_current_batch_is_consumed(
    tmp_path: Path,
):
    fetcher = PipelineFetcher()
    llm = PipelineLLM(fetcher.next_batch_started)
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=LargeMultiHitSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )

    async with service:
        result = await service.run(
            {
                "question": "What is the verified fact?",
                "mode": "standard",
                "run_id": "pipeline-fetch-run",
                "options": {"max_rounds": 1, "max_workers": 1},
            }
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert llm.next_batch_started_before_first_extraction is False
    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "fetch.json").read_text(
            encoding="utf-8"
        )
    )
    unused = [trace for trace in traces if trace["status"] == "prefetched_unused"]
    assert len(unused) < 8


@pytest.mark.asyncio
async def test_long_document_is_chunked_instead_of_rejected_as_budget_exhausted(tmp_path: Path):
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=LongDocumentFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "What is the verified fact?",
                "mode": "deep_search",
                "run_id": "long-document-run",
            }
        )

    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "extraction.json").read_text(encoding="utf-8")
    )
    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert result.gate.passed
    assert "42 units" in result.markdown
    assert len(traces) == 1
    assert traces[0]["accepted_count"] == 1
    assert traces[0]["content_truncated"]
    assert not any("Source is too large" in gap for gap in result.gate.violations)


@pytest.mark.asyncio
async def test_zero_claim_static_html_does_not_force_dynamic_fetch(tmp_path: Path):
    fetcher = DynamicRetryFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "What is the verified fact?",
                "run_id": "dynamic-zero-claim-retry",
                "options": {"max_rounds": 1, "max_workers": 1},
            }
        )

    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert fetcher.calls == [("https://example.com/research", False)]
    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "fetch.json").read_text(
            encoding="utf-8"
        )
    )
    assert [trace["status"] for trace in traces] == ["no_evidence"]
    assert traces[0]["duration_seconds"] >= 0
    assert traces[0]["content_chars"] == len("Loading application data...")


class BadQuoteLLM(FakeLLM):
    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        if "research planner" in system:
            return await super().complete_json(system, user, max_tokens=max_tokens)
        return (
            {
                "claims": [
                    {
                        "text": "The verified fact is 42 units.",
                        "quote": "This quote was not in the source.",
                        "impact": "high",
                        "claim_type": "factual",
                    }
                ],
                "knowledge_gaps": [],
            },
            LLMResponse("{}", input_tokens=30, output_tokens=15),
        )


@pytest.mark.asyncio
async def test_non_exact_supporting_context_is_a_review_warning(tmp_path: Path):
    service = ResearchService.in_memory(
        llm=BadQuoteLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "Reject unsupported evidence",
                "mode": "deep_search",
                "run_id": "rejected-quote-run",
            }
        )

    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "extraction.json").read_text(encoding="utf-8")
    )
    reasons = {
        item["reason"]
        for trace in traces
        for item in trace["rejected_candidates"]
    }
    # This fixture intentionally returns extraction-shaped JSON for the
    # synthesizer too, so semantic assessment falls back to partial.
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert "supporting_context_not_exact_substring" in reasons
    assert all(
        item.get("action") == "review_warning"
        for trace in traces
        for item in trace["rejected_candidates"]
    )
    assert len(traces) == 1
    fetch_traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "fetch.json").read_text(encoding="utf-8")
    )
    assert len(fetch_traces) == 1


@pytest.mark.asyncio
async def test_extractor_accepts_equivalent_numeric_context_without_retry(tmp_path: Path):
    class RepairingExtractorLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.extraction_calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "You extract evidence" in system:
                self.extraction_calls += 1
                payload = json.loads(user)
                if "repair" in payload:
                    claim = {
                        "text": "The verified fact is 42 units.",
                        "quote": "The verified fact is 42 units.",
                        "impact": "high",
                        "claim_type": "fact",
                    }
                else:
                    claim = {
                        "text": "In 2026, the verified fact is 42 units.",
                        "quote": "The verified fact is 42 units.",
                        "impact": "high",
                        "claim_type": "fact",
                    }
                return (
                    {"claims": [claim], "knowledge_gaps": []},
                    LLMResponse("{}", input_tokens=30, output_tokens=15),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = RepairingExtractorLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "Repair mechanically invalid evidence",
                "run_id": "repair-extraction-run",
            }
        )

    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "extraction.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert llm.extraction_calls == 1
    assert traces[0]["accepted_count"] == 1
    assert traces[0]["rejected_candidates"] == []


@pytest.mark.asyncio
async def test_provider_http_response_does_not_mask_extraction_error(tmp_path: Path):
    class ProviderFailureLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "You extract evidence" in system:
                error = RuntimeError("provider unavailable")
                error.response = object()
                raise error
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=ProviderFailureLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "Handle an extraction provider failure",
                "run_id": "provider-response-regression",
                "options": {"max_rounds": 1},
            }
        )

    traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "extraction.json").read_text(
            encoding="utf-8"
        )
    )
    # Evidence-assisted profiles still let the Writer produce a useful report
    # when extraction is unavailable; the provider failure remains observable
    # in the extraction trace below.
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert traces[0]["error"] == "RuntimeError: provider unavailable"
    assert "input_tokens" not in traces[0]["error"]

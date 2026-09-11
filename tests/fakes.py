from __future__ import annotations

import json

from deepresearch.llm import LLMResponse
from deepresearch.schemas import FetchedDocument, SearchHit


class FakeLLM:
    model_id = "fake-research-model"

    def __init__(self, planner_task_limit: int | None = None):
        self.planner_task_limit = planner_task_limit
        self.calls = []

    async def complete_text(self, system: str, user: str, *, max_tokens: int = 2000):
        self.calls.append(("text", system, user))
        return LLMResponse("fake", input_tokens=10, output_tokens=5)

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        self.calls.append(("json", system, user))
        payload = json.loads(user)
        if "research planner" in system:
            gap_dimensions = [
                item.get("dimension")
                for item in payload.get("actionable_gaps_from_synthesizer", [])
                if item.get("dimension")
            ]
            dimensions = (
                gap_dimensions
                if payload.get("round_no", 1) > 1 and gap_dimensions
                else payload["required_dimensions"] or ["核心答案"]
            )
            limit = self.planner_task_limit or min(
                len(dimensions), payload["rules"]["max_tasks"]
            )
            tasks = [
                {
                    "objective": f"Investigate {dimension}",
                    "research_dimension": dimension,
                    "queries": [f"{payload['question']} {dimension}"],
                    "key_questions": [f"What establishes {dimension}?"],
                    "preferred_source_types": ["primary", "independent"],
                    "verification_strategy": "verify from fetched full text",
                    "expected_output": "validated claims",
                    "sufficiency": {
                        "min_queries": 1,
                        "min_query_coverage": 1,
                        "min_successful_extractions": 1,
                        "min_effective_sources": 1,
                        "min_independent_domains": 1,
                        "min_evidence": 1,
                        "max_source_quality_tier": 3,
                    },
                }
                for dimension in dimensions[:limit]
            ]
            return {
                "query_type": "straightforward" if len(tasks) == 1 else "breadth_first",
                "research_strategy": "test strategy",
                "answer_requirements": dimensions[:limit],
                "tasks": tasks,
            }, LLMResponse("{}", input_tokens=20, output_tokens=10)

        if "research worker subagent" in system:
            if payload.get("phase") == "strategy":
                return {
                    "research_assessment": "execute the delegated fixture task",
                    "queries": payload["task_contract"]["queries"],
                    "source_targets": ["fixture source"],
                    "risks": [],
                }, LLMResponse("{}", input_tokens=15, output_tokens=8)
            if payload.get("phase") == "candidate_selection":
                return {
                    "selected_candidate_ids": [
                        item["candidate_id"] for item in payload.get("candidates", [])
                    ],
                    "next_action": "fetch_selected",
                    "rationale": "fixture selects every coarse-filtered candidate",
                }, LLMResponse("{}", input_tokens=15, output_tokens=8)
            claims = payload.get("validated_claims", [])
            return {
                "result_brief": "；".join(item.get("text", "") for item in claims),
                "knowledge_gaps": [],
                "conflicts": [],
            }, LLMResponse("{}", input_tokens=15, output_tokens=8)

        if "research synthesizer" in system and "final research writer" not in system:
            context = payload["source_context"]
            missing_dimensions = context.get("missing_explicit_dimensions") or []
            worker_sufficient = all(
                item.get("passed") for item in context.get("current_worker_sufficiency", [])
            )
            semantically_ready = (
                context.get("has_source_linked_claims")
                and not missing_dimensions
                and worker_sufficient
            )
            decision = "ready" if semantically_ready else "need_more" if payload["can_continue"] else "partial"
            missing = []
            if decision == "need_more":
                dimensions = missing_dimensions or ["核心答案"]
                missing = [
                    {
                        "gap_id": f"fixture-gap-{dimension}",
                        "dimension": dimension,
                        "missing_information": f"More verified evidence is required for {dimension}",
                        "why_needed": "The hard gate is not satisfied",
                        "preferred_source_types": ["primary"],
                        "verification_requirement": "one verified source",
                        "priority": "high",
                    }
                    for dimension in dimensions
                ]
            return {
                "decision": decision,
                "reasoning": ["fixture synthesis"],
                "coverage": [],
                "missing_information": missing,
            }, LLMResponse("{}", input_tokens=20, output_tokens=10)

        if "final research writer" in system:
            claims = payload["claim_ledger"]
            paragraphs = [
                {"text": claim["text"], "claim_ids": [claim["claim_id"]]}
                for claim in claims
            ]
            return {
                "sections": [{"title": "核心发现", "paragraphs": paragraphs}],
                "missing_dimensions": [],
            }, LLMResponse("{}", input_tokens=40, output_tokens=20)

        untrusted = payload["untrusted_content"]
        if "verified fact is 42 units" in untrusted:
            claims = [
                {
                    "text": "The verified fact is 42 units.",
                    "quote": "The verified fact is 42 units.",
                    "impact": "high",
                    "claim_type": "factual",
                }
            ]
            source_uri = payload.get("source_uri", "")
            if "independent.example.net" in source_uri or "source-3.example" in source_uri:
                claims.append(
                    {
                        "text": "The independently verified value is 43 units.",
                        "quote": "The independently verified value is 43 units.",
                        "impact": "medium",
                        "claim_type": "factual",
                    }
                )
        else:
            claims = []
        return {
            "claims": claims,
            "knowledge_gaps": [],
        }, LLMResponse("{}", input_tokens=30, output_tokens=15)


class FakeSearchBackend:
    name = "fake_search"

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls = []

    async def search(self, query: str, max_results: int = 5):
        self.calls.append(query)
        if self.fail_first and len(self.calls) == 1:
            return []
        dimension = next(
            (item for item in ("dimension-a", "dimension-b", "dimension-c") if item in query),
            None,
        )
        if dimension:
            url = f"https://{dimension}.example/research"
        elif "官方" in query and "第一手" in query:
            url = "https://official.example.org/research"
        elif "独立" in query and "核查" in query:
            url = "https://independent.example.net/research"
        else:
            url = "https://example.com/research"
        return [
            SearchHit(
                title="Verified source",
                url=url,
                snippet="The verified fact is 42 units.",
                provider=self.name,
                rank=1,
            )
        ]


class FakeFetcher:
    def __init__(self):
        self.calls = []

    async def fetch(self, url: str):
        self.calls.append(url)
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title="Verified source",
            content=(
                "The verified fact is 42 units. "
                "The independently verified value is 43 units. Additional context is available."
            ),
            content_type="text/plain",
            http_status=200,
        )


class TwoHitSearchBackend:
    name = "two_hit_search"

    async def search(self, query: str, max_results: int = 5):
        return [
            SearchHit(
                title="Broken source",
                url="https://broken.example.com/research",
                provider=self.name,
                rank=1,
            ),
            SearchHit(
                title="Verified source",
                url="https://example.com/research",
                provider=self.name,
                rank=2,
            ),
        ]


class FailOneFetcher(FakeFetcher):
    async def fetch(self, url: str):
        self.calls.append(url)
        if "broken.example.com" in url:
            raise RuntimeError("injected fetch failure")
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title="Verified source",
            content=(
                "The verified fact is 42 units. "
                "The independently verified value is 43 units. Additional context is available."
            ),
            content_type="text/plain",
            http_status=200,
        )

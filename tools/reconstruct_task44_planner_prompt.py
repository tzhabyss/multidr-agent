"""Reconstruct the exact application-level Round-1 Planner messages for task44."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from deepresearch.agents import Planner
from deepresearch.llm import LLMResponse
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import ResearchScope, TaskBudget
from deepresearch.skill_loader import load_skill_registry


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    ROOT
    / "artifacts"
    / "evaluations"
    / "rerun-task44-20260812-122024"
    / "samples"
    / "eval-task44-r1-4ae49666"
)
OUTPUT = ROOT / "docs" / "task44-craft-planner-prompt-20260812.md"
SYSTEM_OUTPUT = ROOT / "docs" / "task44-craft-planner-system-20260812.txt"
USER_OUTPUT = ROOT / "docs" / "task44-craft-planner-user-20260812.json"


class CaptureLLM:
    model_id = "gpt-5.6-terra"
    provider_id = "default"

    def __init__(self) -> None:
        self.system = ""
        self.user = ""
        self.max_tokens = 0

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> tuple[dict[str, Any], LLMResponse]:
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        value = {
            "query_type": "straightforward",
            "research_strategy": "capture only",
            "delivery_contract_required": True,
            "answer_requirements": ["capture"],
            "requirements": [
                {
                    "requirement_id": "req-001",
                    "text": "capture",
                    "kind": "fact",
                    "importance": "required",
                }
            ],
            "tasks": [
                {
                    "task_key": "capture",
                    "objective": "capture",
                    "research_dimension": "capture",
                    "research_object": "capture",
                    "task_type": "web",
                    "task_kind": "gather",
                    "depends_on": [],
                    "consumes_requirement_ids": [],
                    "produces_requirement_ids": ["req-001"],
                    "capability": "web_research",
                    "expected_value": 0.5,
                    "source_origin": False,
                    "key_questions": ["capture"],
                    "queries": ["capture"],
                    "preferred_source_types": ["official"],
                    "verification_strategy": "capture",
                    "expected_output": "capture",
                    "requirement_ids": ["req-001"],
                    "sufficiency": {
                        "min_queries": 1,
                        "min_query_coverage": 1,
                        "min_successful_extractions": 1,
                        "min_effective_sources": 1,
                        "min_independent_domains": 1,
                        "min_evidence": 1,
                    },
                }
            ],
        }
        text = json.dumps(value, ensure_ascii=False)
        return value, LLMResponse(
            content=text,
            input_tokens=0,
            output_tokens=0,
            finish_reason="stop",
            reasoning_chars=0,
        )


async def main() -> None:
    config = load_runtime_config(ROOT / "config" / "runtime.yaml")
    scope_data = json.loads(
        (SAMPLE / "details" / "planning" / "research-plan.json").read_text(
            encoding="utf-8"
        )
    )["scope"]
    scope = ResearchScope.model_validate(scope_data)
    capture = CaptureLLM()
    skills = load_skill_registry(
        ROOT,
        executable_roles=config.tool_permissions.executable_roles,
        profile_skill_fallbacks={
            profile_id: profile.skill_fallbacks
            for profile_id, profile in config.agent_profiles.items()
            if profile.skill_fallbacks
        },
        profile_skill_provider_modes={
            profile_id: profile.skill_provider_modes
            for profile_id, profile in config.agent_profiles.items()
            if profile.skill_provider_modes
        },
    )
    planner = Planner(capture, skills, config)
    await planner.plan(
        "eval-task44-r1-4ae49666",
        scope,
        TaskBudget(max_searches=200, max_fetches=336, max_wall_time_seconds=4080),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={
            "source_count": 0,
            "claim_count": 0,
            "candidate_leads": [],
            "prior_plans": [],
            "existing_claims": [],
        },
        memory_packet={"role": "planner"},
        max_tasks_override=12,
    )
    document = f"""# Task44 craft Planner prompt（2026-08-12 首轮重建）

## 调用参数

- model: `gpt-5.6-terra`
- temperature: `0.1`
- max output tokens: `{capture.max_tokens}`
- response format: JSON object（项目调用 `complete_json`）
- application timeout: craft 1200 秒；800 秒时并发启动 default hedge，default 最长 400 秒
- system/user 两段消息需分别发送，不要把 Markdown 围栏一起发送。

## System message

```text
{capture.system}
```

## User message

```json
{capture.user}
```
"""
    OUTPUT.write_text(document, encoding="utf-8")
    SYSTEM_OUTPUT.write_text(capture.system, encoding="utf-8")
    USER_OUTPUT.write_text(capture.user, encoding="utf-8")
    print(OUTPUT)
    print(SYSTEM_OUTPUT)
    print(USER_OUTPUT)
    print(f"system_chars={len(capture.system)}")
    print(f"user_chars={len(capture.user)}")
    print(f"max_tokens={capture.max_tokens}")


if __name__ == "__main__":
    asyncio.run(main())

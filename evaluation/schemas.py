"""Evaluation contracts that keep agent inputs separate from references."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deepresearch.schemas import ResearchConstraints, ResearchMode, ResearchRequest


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    question: str
    mode: ResearchMode = ResearchMode.RESEARCH
    profile_id: str = "general_research"
    source_urls: list[str] = Field(default_factory=list)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    expected_answer: str | None = None
    reference_source_ids: list[str] = Field(default_factory=list)
    atomic_rubrics: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("case_id", "question")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    def agent_request(
        self,
        run_id: str,
        *,
        mode: ResearchMode | str | None = None,
        profile_id: str | None = None,
    ) -> ResearchRequest:
        """Whitelist only agent-visible fields; references never cross this boundary."""
        return ResearchRequest(
            question=self.question,
            mode=ResearchMode(mode) if mode is not None else self.mode,
            profile_id=profile_id or self.profile_id,
            source_urls=self.source_urls,
            constraints=self.constraints,
            run_id=run_id,
        )

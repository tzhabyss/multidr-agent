from __future__ import annotations

import pytest

from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import ResearchMode, ResearchRequest
from deepresearch.service import ResearchService
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchBackend


@pytest.mark.asyncio
async def test_deep_mode_reaches_scope_budget_and_output(tmp_path):
    config = load_runtime_config().model_copy(
        update={
            "memory": load_runtime_config().memory.model_copy(
                update={
                    "local_db_path": str(tmp_path / "memory.sqlite"),
                    "automatic_capture_enabled": False,
                }
            )
        }
    )
    async with ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    ) as service:
        result = await service.run(
            ResearchRequest(question="What is the verified fact?", mode=ResearchMode.DEEP)
        )
        assert service._graph is not None
        state = await service._graph.graph.aget_state(
            {"configurable": {"thread_id": result.run_id}}
        )

    assert result.mode is ResearchMode.DEEP
    assert state.values["scope"]["mode"] == "deep"
    options = state.values["request"]["options"]
    assert options["max_rounds"] == 4
    assert options["max_worker_iterations"] == 6
    assert options["max_workers"] == 10
    assert state.values["run_budget"]["max_total_searches"] == 200
    assert state.values["structured_output"]["mode"] == "deep"

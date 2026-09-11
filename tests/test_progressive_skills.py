from pathlib import Path

from deepresearch.skill_loader import load_skill_registry


ROOT = Path(__file__).resolve().parent.parent
LAYER_SETTINGS = {
    "orchestrator": {"enabled": True, "skills": [], "injection": "none"},
    "strategy": {"enabled": True, "skills": [], "injection": "selected"},
    "access": {"enabled": True, "skills": [], "injection": "tools"},
}


def layered_registry():
    return load_skill_registry(
        ROOT,
        skill_mode="layered",
        layer_settings=LAYER_SETTINGS,
    )


def test_layered_loading_is_metadata_then_body():
    registry = layered_registry()
    strategy = registry.get("research-worker")
    assert strategy is not None
    assert strategy.instructions_loaded is False
    catalog = registry.prompt_instructions(
        role="worker", profile_id="general_research", phase="worker_strategy"
    )
    assert "research-worker" in catalog
    assert "# Research Worker" not in catalog
    assert strategy.instructions_loaded is False
    body = registry.prompt_instructions(
        role="worker",
        profile_id="general_research",
        phase="worker_loop",
        selected_skills=("research-worker",),
    )
    assert "# Research Worker" in body
    assert strategy.instructions_loaded is True


def test_layered_responsibility_boundaries():
    registry = layered_registry()
    assert registry.prompt_instructions(
        role="planner", profile_id="general_research", phase="planner"
    ) == ""
    assert registry.prompt_instructions(
        role="synthesizer", profile_id="general_research", phase="synthesizer"
    ) == ""
    tools = registry.to_openai_tools(
        role="worker", profile_id="general_research", any_tags=("search", "file")
    )
    assert tools
    assert all(
        registry.get(item["function"]["name"]).skill_layer == "access"
        for item in tools
    )


def test_legacy_mode_keeps_full_prompt_behavior():
    registry = load_skill_registry(ROOT, skill_mode="legacy")
    prompt = registry.prompt_instructions(
        role="worker", profile_id="general_research"
    )
    assert "# Research Worker" in prompt

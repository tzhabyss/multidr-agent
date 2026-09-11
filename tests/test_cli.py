from __future__ import annotations

from deepresearch.cli import _parser, build_request, main
from deepresearch.schemas import ResearchMode


def test_cli_builds_request_with_scope_constraints():
    args = _parser().parse_args(
        [
            "run",
            "Compare two policies",
            "--source-url",
            "https://example.com/a",
            "--required-dimension",
            "cost",
            "--allow-domain",
            "example.com",
            "--block-domain",
            "private.example.com",
            "--run-id",
            "cli-test-run",
            "--min-evidence",
            "6",
            "--max-source-quality-tier",
            "2",
            "--max-rounds",
            "3",
            "--mode",
            "deep",
        ]
    )
    request = build_request(args)

    assert request.mode == ResearchMode.DEEP
    assert request.source_urls == ["https://example.com/a"]
    assert request.constraints.required_dimensions == ["cost"]
    assert request.constraints.allowed_domains == ["example.com"]
    assert request.constraints.blocked_domains == ["private.example.com"]
    assert request.run_id == "cli-test-run"
    assert request.options.task_sufficiency.min_evidence == 6
    assert request.options.task_sufficiency.max_source_quality_tier == 2
    assert request.options.task_sufficiency.min_effective_sources is None
    assert request.options.max_rounds == 3


def test_doctor_reports_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["doctor"]) == 1
    assert "[FAIL] OPENAI_API_KEY: missing" in capsys.readouterr().out


def test_doctor_does_not_require_an_unused_provider_key(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENLUX_API_KEY", raising=False)

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "[OK] OPENAI_API_KEY: configured" in output
    assert "OPENLUX_API_KEY" not in output


def test_cli_accepts_explicit_runtime_config_path(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    args = _parser().parse_args(
        ["run", "Research question", "--config", str(config_path)]
    )

    assert args.config == config_path

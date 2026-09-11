from __future__ import annotations

from pathlib import Path

from deepresearch.runtime_config import load_runtime_config


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_uses_environment_variables_for_project_credentials():
    config = load_runtime_config(ROOT / "config" / "runtime.yaml")

    assert config.memory.automatic_capture_enabled is True
    assert not hasattr(config.memory, "mem0_api_key_env")
    assert config.mcp.servers["deepseek-web-search"].env_files == {}
    assert config.mcp.servers["anysearch"].env_files == {}
    assert (
        config.mcp.servers["deepseek-web-search"].env["DEEPSEEK_API_KEY"]
        == "${DEEPSEEK_API_KEY}"
    )
    assert (
        config.mcp.servers["anysearch"].headers["Authorization"]
        == "Bearer ${ANYSEARCH_API_KEY}"
    )


def test_plaintext_secret_files_are_not_present_and_dotenv_is_ignored():
    forbidden = {
        "mem0key.txt",
        "deepseek apikey.txt",
        "anysearchkey.txt",
        "dee.txt",
        "2x.txt",
    }

    assert not [name for name in forbidden if (ROOT / name).exists()]
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".env" in ignored
    assert "!.env.example" in ignored

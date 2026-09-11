from deepresearch.prompts import REPORT_SYSTEM_PROMPT


NORMALIZED_REPORT_PROMPT = " ".join(REPORT_SYSTEM_PROMPT.split())


def test_report_prompt_prioritizes_analysis_over_evidence_volume() -> None:
    assert "not the number or ordering of retrieved claims" in NORMALIZED_REPORT_PROMPT
    assert "mechanisms, causes, consequences" in NORMALIZED_REPORT_PROMPT
    assert (
        "property or observation -> underlying mechanism or cause"
        in NORMALIZED_REPORT_PROMPT
    )
    assert "is not comparative analysis" in NORMALIZED_REPORT_PROMPT
    assert (
        "must not crowd out another explicit user requirement"
        in NORMALIZED_REPORT_PROMPT
    )
    assert "Do not compress away a requested explanation" in NORMALIZED_REPORT_PROMPT


def test_report_prompt_preserves_explicit_scope_and_reasoning_bridges() -> None:
    assert "For every material why/how request" in NORMALIZED_REPORT_PROMPT
    assert "For every material comparison" in NORMALIZED_REPORT_PROMPT
    assert "For every conclusion" in NORMALIZED_REPORT_PROMPT
    assert "follow the original user constraint" in NORMALIZED_REPORT_PROMPT
    assert (
        "do not silently expand them to a full calendar period"
        in NORMALIZED_REPORT_PROMPT
    )


def test_report_prompt_preserves_verified_rows_with_unavailable_cells():
    assert "retain one row for every entity that is verified anywhere" in NORMALIZED_REPORT_PROMPT
    assert "do not drop the whole entity row" in NORMALIZED_REPORT_PROMPT
    assert "Candidate-only" in NORMALIZED_REPORT_PROMPT

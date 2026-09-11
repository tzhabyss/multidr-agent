from tools.audit_skill_evolution_oracle_guards import audit


def test_audit_confirms_target_regressions_blocked_and_guards_can_pass():
    result = audit()

    assert result["mixed_target_case"]["blind_spot_present"] is False
    assert result["mixed_target_case"]["regression_blocked"] is True
    assert result["mixed_target_case"]["checks"]["target:evidence_gain"] == "pass"
    assert (
        result["mixed_target_case"]["checks"]["target:zero_gain_searches"]
        == "fail"
    )
    assert result["optimal_guard_case"]["guard_role_missing"] is False
    assert result["optimal_guard_case"]["non_regressing_guard_passed"] is True
    assert result["skill_candidate_can_self_fix"] is False

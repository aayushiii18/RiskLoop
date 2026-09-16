"""Automated tests for Phase 5C Frozen 12-Run Decision Analysis & Representative Selection.

SRD Traceability: FR-29..FR-37, NFR-01..NFR-04, FR-38
"""

import json
import pytest
from pathlib import Path

from riskloop.evaluation.adoption import (
    compute_stage1_statistics,
    evaluate_stage2a_improvement,
    evaluate_stage2b_regression_veto,
    run_validation_adoption_protocol
)
from riskloop.evaluation.selection import (
    select_representative_condition_a_seed,
    select_representative_condition_b_seed,
    execute_representative_model_selection
)
from riskloop.evaluation.decision import (
    generate_official_decision_analysis,
    save_decision_analysis_artifact,
    OFFICIAL_12_RUN_SCORES
)


def test_median_calculations():
    """Verify median and descriptive statistics calculation for 3 validation runs (FR-30)."""
    runs = [0.781893, 0.812785, 0.791837]
    stats = compute_stage1_statistics(runs)
    
    assert stats["median"] == pytest.approx(0.791837)
    assert stats["min"] == pytest.approx(0.781893)
    assert stats["max"] == pytest.approx(0.812785)
    assert stats["mean"] == pytest.approx((0.781893 + 0.812785 + 0.791837) / 3)


def test_stage2a_adoption_rule():
    """Verify Stage 2A adoption rule requiring Median(B) - Median(A) > 0.02 AND >= 2 B runs > Max(A)."""
    # Case 1: Both conditions pass
    stats_a = compute_stage1_statistics([0.70, 0.71, 0.72]) # Median 0.71, Max 0.72
    stats_b = compute_stage1_statistics([0.75, 0.76, 0.77]) # Median 0.76 (diff 0.05 > 0.02), 3 B runs > 0.72
    adopted, details = evaluate_stage2a_improvement(stats_a, stats_b, margin=0.02)
    assert adopted is True

    # Case 2: Median diff <= 0.02 fails
    stats_b_close = compute_stage1_statistics([0.72, 0.725, 0.73]) # Median 0.725 (diff 0.015 <= 0.02)
    adopted_close, _ = evaluate_stage2a_improvement(stats_a, stats_b_close, margin=0.02)
    assert adopted_close is False

    # Case 3: Extreme count < 2 fails despite median diff > 0.02
    stats_a_outlier = compute_stage1_statistics([0.70, 0.71, 0.85]) # Median 0.71, Max 0.85
    stats_b_high_med = compute_stage1_statistics([0.75, 0.76, 0.86]) # Median 0.76 (diff 0.05), but only 1 B run (0.86) > 0.85
    adopted_extreme_fail, details = evaluate_stage2a_improvement(stats_a_outlier, stats_b_high_med, margin=0.02)
    assert adopted_extreme_fail is False
    assert details["extreme_condition_met"] is False


def test_stage2b_veto_rule():
    """Verify Stage 2B global veto and ambiguity rule logic (FR-32..FR-34)."""
    # Case 1: Veto triggered (regression > 0.005 AND >= 2 B runs strictly below Min(A))
    stats_a = compute_stage1_statistics([0.781893, 0.812785, 0.791837]) # Median 0.791837, Min 0.781893
    stats_b = compute_stage1_statistics([0.715328, 0.707581, 0.638158]) # Median 0.707581 (regression 0.084 > 0.005), 3 B runs < 0.781893
    is_vetoed, is_ambiguous, details = evaluate_stage2b_regression_veto(stats_a, stats_b, veto_margin=0.005)
    assert is_vetoed is True
    assert is_ambiguous is False

    # Case 2: Ambiguous (regression > 0.005 BUT < 2 B runs strictly below Min(A))
    stats_a_wide = compute_stage1_statistics([0.70, 0.80, 0.82]) # Median 0.80, Min 0.70
    stats_b_mod = compute_stage1_statistics([0.75, 0.76, 0.77]) # Median 0.76 (regression 0.04 > 0.005), 0 B runs < 0.70
    is_vetoed_amb, is_ambiguous_amb, details_amb = evaluate_stage2b_regression_veto(stats_a_wide, stats_b_mod, veto_margin=0.005)
    assert is_vetoed_amb is False
    assert is_ambiguous_amb is True


def test_strict_inequality_boundaries():
    """Verify strict inequality boundaries (>, <) for margins and extreme run comparisons."""
    # Boundary 1: Median diff exactly 0.02 (must NOT pass, requires strict > 0.02)
    stats_a = compute_stage1_statistics([0.80, 0.80, 0.80]) # Median 0.80, Max 0.80
    stats_b_exact = compute_stage1_statistics([0.82, 0.82, 0.82]) # Median 0.82 (diff <= 0.02)
    adopted, details = evaluate_stage2a_improvement(stats_a, stats_b_exact, margin=0.02)
    assert adopted is False
    assert details["median_condition_met"] is False

    # Boundary 2: B run exactly equal to Max(A) (must NOT count as exceeding)
    stats_a_max = compute_stage1_statistics([0.65, 0.70, 0.75]) # Median 0.70, Max 0.75
    stats_b_equal_max = compute_stage1_statistics([0.74, 0.75, 0.75]) # 2 B runs equal to 0.75 (NOT > 0.75)
    adopted_max, details_max = evaluate_stage2a_improvement(stats_a_max, stats_b_equal_max, margin=0.02)
    assert details_max["b_runs_exceeding_max_a"] == 0
    assert adopted_max is False

    # Boundary 3: Median regression exactly 0.005 (must NOT trigger regression flag, requires strict > 0.005)
    stats_a_reg = compute_stage1_statistics([0.80, 0.80, 0.80]) # Median 0.80
    stats_b_reg_exact = compute_stage1_statistics([0.795, 0.795, 0.795]) # Median 0.795 (regression exactly 0.005)
    is_vetoed, is_ambiguous, details_reg = evaluate_stage2b_regression_veto(stats_a_reg, stats_b_reg_exact, veto_margin=0.005)
    assert details_reg["regression_flagged"] is False
    assert is_vetoed is False
    assert is_ambiguous is False

    # Boundary 4: B run exactly equal to Min(A) (must NOT count as strictly below Min(A))
    stats_a_min = compute_stage1_statistics([0.70, 0.80, 0.85]) # Min 0.70, Median 0.80
    stats_b_equal_min = compute_stage1_statistics([0.70, 0.70, 0.71]) # 2 B runs equal 0.70 (NOT < 0.70)
    is_vetoed_min, _, details_min = evaluate_stage2b_regression_veto(stats_a_min, stats_b_equal_min, veto_margin=0.005)
    assert details_min["b_runs_below_min_a"] == 0
    assert is_vetoed_min is False


def test_representative_a_selection():
    """Verify selection of median seed for Condition A with tie-breaking rule (FR-36)."""
    # Exact median selection
    scores_cap = {42: 0.781893, 43: 0.812785, 44: 0.791837} # Median 0.791837 -> Seed 44
    assert select_representative_condition_a_seed(scores_cap) == 44

    scores_anti = {42: 0.771831, 43: 0.804665, 44: 0.741333} # Median 0.771831 -> Seed 42
    assert select_representative_condition_a_seed(scores_anti) == 42

    # Tie-breaker: lowest seed ID when scores are identical
    scores_tie = {42: 0.80, 43: 0.80, 44: 0.80}
    assert select_representative_condition_a_seed(scores_tie) == 42


def test_b_rank_sum_selection():
    """Verify Condition B joint representative seed selection by minimum sum-of-ranks (FR-37)."""
    val_seed_map_b = {
        "Cap On Liability": {42: 0.715328, 43: 0.707581, 44: 0.638158},            # Ranks: 42=1, 43=2, 44=3
        "Anti-Assignment": {42: 0.710997, 43: 0.788406, 44: 0.715736},             # Ranks: 43=1, 44=2, 42=3
        "Termination For Convenience": {42: 0.580247, 43: 0.583851, 44: 0.617284} # Ranks: 44=1, 43=2, 42=3
    }
    # Sum of ranks:
    # Seed 42: 1 + 3 + 3 = 7
    # Seed 43: 2 + 1 + 2 = 5
    # Seed 44: 3 + 2 + 1 = 6
    selected_seed, total_ranks, task_ranks = select_representative_condition_b_seed(val_seed_map_b)
    assert selected_seed == 43
    assert total_ranks[43] == 5
    assert total_ranks[44] == 6
    assert total_ranks[42] == 7


def test_no_accidental_test_set_access():
    """Verify decision analysis runs without reading, loading, or opening test set artifacts (FR-38)."""
    # Execute official decision analysis function directly
    artifact = generate_official_decision_analysis()
    
    # Assert explicit test isolation statement is present in artifact
    assert "test_set_isolation_statement" in artifact
    assert "test_chunks.json" in artifact["test_set_isolation_statement"]
    assert "NOT been opened, loaded, evaluated, or accessed" in artifact["test_set_isolation_statement"]


def test_official_12_run_decision_analysis_artifact():
    """Verify complete official 12-run decision analysis artifact output and veto outcome."""
    artifact = generate_official_decision_analysis()

    # Verify scores count (9 Condition A, 3 Condition B)
    scores = artifact["official_12_run_scores"]
    assert len(scores) == 12

    # Verify global veto triggered on all 3 tasks
    assert artifact["joint_architecture_vetoed"] is True
    assert len(artifact["veto_reasons"]) == 3

    # Verify final task adoptions default to Condition A across all tasks
    adoptions = artifact["final_task_adoptions"]
    assert adoptions["Cap On Liability"] == "Condition_A"
    assert adoptions["Anti-Assignment"] == "Condition_A"
    assert adoptions["Termination For Convenience"] == "Condition_A"

    # Verify representative A models match expected runs
    reps = artifact["representative_models"]
    assert reps["Cap On Liability"]["run_name"] == "run_03"
    assert reps["Cap On Liability"]["seed"] == 44
    assert reps["Cap On Liability"]["validation_score"] == pytest.approx(0.791837)

    assert reps["Anti-Assignment"]["run_name"] == "run_04"
    assert reps["Anti-Assignment"]["seed"] == 42
    assert reps["Anti-Assignment"]["validation_score"] == pytest.approx(0.771831)

    assert reps["Termination For Convenience"]["run_name"] == "run_07"
    assert reps["Termination For Convenience"]["seed"] == 42
    assert reps["Termination For Convenience"]["validation_score"] == pytest.approx(0.666667)

    # Verify Condition B representative seed selection
    b_selection = artifact["condition_b_representative_joint_seed_selection"]
    assert b_selection["selected_b_seed"] == 43
    total_ranks = b_selection["b_seed_total_ranks"]
    assert total_ranks.get(43, total_ranks.get("43")) == 5

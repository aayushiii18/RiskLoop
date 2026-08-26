"""Tests for Validation-Only Architecture Adoption Decision Protocol.

SRD Traceability: FR-29, FR-30, FR-31, FR-32, FR-33, FR-34, NFR-01, NFR-02, NFR-03, NFR-04
"""

import pytest
from riskloop.evaluation.adoption import (
    compute_stage1_statistics,
    evaluate_stage2a_improvement,
    evaluate_stage2b_regression_veto,
    run_validation_adoption_protocol
)

def test_stage1_statistics():
    """Verify calculation of Stage 1 descriptive statistics."""
    runs = [0.81, 0.85, 0.83]
    stats = compute_stage1_statistics(runs)
    
    assert stats["mean"] == pytest.approx(0.83)
    assert stats["median"] == pytest.approx(0.83)
    assert stats["min"] == pytest.approx(0.81)
    assert stats["max"] == pytest.approx(0.85)
    assert stats["range"] == pytest.approx(0.04)


def test_stage2a_improvement_adopted():
    """Verify Condition B adopted when median diff > 2% AND >= 2 B runs exceed Max(A)."""
    stats_a = compute_stage1_statistics([0.80, 0.81, 0.82]) # Median 0.81, Max 0.82
    stats_b = compute_stage1_statistics([0.84, 0.85, 0.86]) # Median 0.85, 3 B runs > 0.82
    
    adopted, details = evaluate_stage2a_improvement(stats_a, stats_b, margin=0.02)
    assert adopted is True
    assert details["median_diff"] == pytest.approx(0.04)


def test_stage2a_improvement_rejected_by_extreme_rule():
    """Verify Condition B rejected if median diff > 2% BUT extreme rule (<2 B runs > Max(A)) fails."""
    stats_a = compute_stage1_statistics([0.70, 0.71, 0.85]) # Median 0.71, Max 0.85
    stats_b = compute_stage1_statistics([0.75, 0.76, 0.86]) # Median 0.76 (diff 5%), but only 1 B run (0.86) > Max(A) (0.85)
    
    adopted, details = evaluate_stage2a_improvement(stats_a, stats_b, margin=0.02)
    assert adopted is False
    assert details["extreme_condition_met"] is False


def test_stage2b_global_veto_triggered():
    """Verify Condition B global veto triggered when regression > 0.5% AND >= 2 B runs strictly below Min(A)."""
    stats_a = compute_stage1_statistics([0.85, 0.86, 0.87]) # Median 0.86, Min 0.85
    stats_b = compute_stage1_statistics([0.80, 0.81, 0.84]) # Median 0.81 (regression 5%), 2 B runs (0.80, 0.81) < 0.85
    
    is_vetoed, is_ambiguous, details = evaluate_stage2b_regression_veto(stats_a, stats_b, veto_margin=0.005)
    assert is_vetoed is True
    assert is_ambiguous is False


def test_stage2b_ambiguity_disclosed_not_vetoed():
    """Verify regression > 0.5% without extreme consistency is flagged as ambiguous, NOT vetoed."""
    stats_a = compute_stage1_statistics([0.80, 0.85, 0.86]) # Median 0.85, Min 0.80
    stats_b = compute_stage1_statistics([0.81, 0.82, 0.83]) # Median 0.82 (regression 3%), but 0 B runs < Min(A) (0.80)
    
    is_vetoed, is_ambiguous, details = evaluate_stage2b_regression_veto(stats_a, stats_b, veto_margin=0.005)
    assert is_vetoed is False
    assert is_ambiguous is True


def test_run_validation_adoption_protocol_complete():
    """Integration test for complete validation adoption protocol."""
    val_a = {
        "Task_A": [0.80, 0.81, 0.82],
        "Task_B": [0.85, 0.86, 0.87]
    }
    val_b = {
        "Task_A": [0.84, 0.85, 0.86], # Stage 2A adopted!
        "Task_B": [0.84, 0.85, 0.86]  # Equal performance, defaults to A
    }
    
    decision = run_validation_adoption_protocol(val_a, val_b)
    
    assert decision["joint_architecture_vetoed"] is False
    assert decision["final_task_adoptions"]["Task_A"] == "Condition_B"
    assert decision["final_task_adoptions"]["Task_B"] == "Condition_A"

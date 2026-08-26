"""Tests for Representative Model Selection.

SRD Traceability: FR-35, FR-36, FR-37
"""

import pytest
from riskloop.evaluation.selection import (
    select_representative_condition_a_seed,
    select_representative_condition_b_seed,
    execute_representative_model_selection
)

def test_select_condition_a_median_seed():
    """Verify Condition A selects the seed corresponding to the median validation score."""
    seed_scores = {42: 0.81, 43: 0.85, 44: 0.83}
    # Scores: 0.81, 0.83, 0.85 -> Median is 0.83 -> Seed 44
    selected = select_representative_condition_a_seed(seed_scores)
    assert selected == 44


def test_select_condition_a_tie_breaker():
    """Verify lowest seed tie-breaker for Condition A when scores are identical."""
    seed_scores = {42: 0.85, 43: 0.85, 44: 0.85}
    selected = select_representative_condition_a_seed(seed_scores)
    assert selected == 42 # Lowest seed number


def test_select_condition_b_joint_sum_of_ranks():
    """Verify Condition B selects the single seed with lowest sum of ranks across tasks."""
    task_scores = {
        "Task_A": {42: 0.90, 43: 0.80, 44: 0.70}, # Ranks: 42=1, 43=2, 44=3
        "Task_B": {42: 0.85, 43: 0.80, 44: 0.75}, # Ranks: 42=1, 43=2, 44=3
        "Task_C": {42: 0.70, 43: 0.90, 44: 0.80}  # Ranks: 42=3, 43=1, 44=2
    }
    # Sum of ranks:
    # Seed 42: 1 + 1 + 3 = 5
    # Seed 43: 2 + 2 + 1 = 5
    # Seed 44: 3 + 3 + 2 = 8
    # Tie between 42 and 43 (total rank 5). Tie-breaker: lowest seed -> 42
    
    selected_seed, total_ranks, task_ranks = select_representative_condition_b_seed(task_scores)
    assert selected_seed == 42
    assert total_ranks[42] == 5
    assert total_ranks[43] == 5


def test_execute_representative_model_selection_integration():
    """Integration test for post-freeze model selection."""
    adoption_decision = {
        "final_task_adoptions": {
            "Task_A": "Condition_B",
            "Task_B": "Condition_A"
        }
    }
    val_a = {
        "Task_A": {42: 0.80, 43: 0.81, 44: 0.82},
        "Task_B": {42: 0.85, 43: 0.86, 44: 0.87}
    }
    val_b = {
        "Task_A": {42: 0.90, 43: 0.85, 44: 0.80},
        "Task_B": {42: 0.88, 43: 0.84, 44: 0.80}
    }
    
    res = execute_representative_model_selection(adoption_decision, val_a, val_b)
    
    selected = res["selected_models"]
    assert selected["Task_A"]["condition"] == "Condition_B"
    assert selected["Task_B"]["condition"] == "Condition_A"
    assert selected["Task_B"]["seed"] == 43 # Median seed for Val A Task B (0.86)

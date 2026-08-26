"""Tests for 12-run experiment matrix integrity and condition isolation.

SRD Traceability: FR-21, FR-22, FR-23, FR-24
"""

import pytest
from riskloop.experiment.runner import generate_12_run_matrix, validate_experiment_isolation

def test_12_run_matrix_structure():
    """Verify that generate_12_run_matrix creates exactly 12 runs for 3 tasks and 3 seeds."""
    tasks = ["Task_1", "Task_2", "Task_3"]
    seeds = [42, 43, 44]
    
    matrix = generate_12_run_matrix(tasks, seeds)
    assert len(matrix) == 12
    
    cond_a_runs = [r for r in matrix if r["condition"] == "Condition_A"]
    cond_b_runs = [r for r in matrix if r["condition"] == "Condition_B"]
    
    assert len(cond_a_runs) == 9 # 3 tasks x 3 seeds
    assert len(cond_b_runs) == 3 # 1 joint model x 3 seeds
    
    # Check that Condition B runs include all tasks in scope
    for r in cond_b_runs:
        assert sorted(r["task_scope"]) == sorted(tasks)


def test_validate_isolation_success(mock_config_dicts):
    """Verify that validate_experiment_isolation passes for valid 12-run matrix and matching configs."""
    tasks = ["Task_1", "Task_2", "Task_3"]
    seeds = [42, 43, 44]
    matrix = generate_12_run_matrix(tasks, seeds)
    
    # Should not raise exception
    validate_experiment_isolation(mock_config_dicts["config_a"], mock_config_dicts["config_b"], matrix)

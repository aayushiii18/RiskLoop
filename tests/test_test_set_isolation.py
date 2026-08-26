"""Tests for strict test-set isolation and single-pass evaluation.

SRD Traceability: FR-38, FR-39, FR-40
"""

import pytest
from riskloop.evaluation.test_evaluator import TestSetEvaluator, TestSetIsolationException

def test_test_evaluator_unfrozen_fails():
    """Verify that TestSetEvaluator refuses to run if selection artifact is missing or invalid."""
    with pytest.raises(TestSetIsolationException) as exc:
        TestSetEvaluator({})
    assert "TEST ISOLATION VIOLATION" in str(exc.value)


def test_test_evaluator_single_pass_enforced():
    """Verify that TestSetEvaluator executes once successfully and blocks second execution."""
    selection_artifact = {
        "selected_models": {
            "Task_A": {"condition": "Condition_A", "seed": 42}
        }
    }
    evaluator = TestSetEvaluator(selection_artifact)
    
    test_data = {
        "Task_A": {
            "y_true": [0, 1, 0, 1],
            "y_score": [0.1, 0.9, 0.2, 0.8]
        }
    }
    
    # First execution succeeds
    res = evaluator.evaluate_test_set(test_data)
    assert "Task_A" in res["test_results"]
    assert res["test_results"]["Task_A"]["test_metrics"]["macro_f1"] == pytest.approx(1.0)
    
    # Second execution MUST fail
    with pytest.raises(TestSetIsolationException) as exc:
        evaluator.evaluate_test_set(test_data)
    assert "Single-execution rule violated" in str(exc.value)

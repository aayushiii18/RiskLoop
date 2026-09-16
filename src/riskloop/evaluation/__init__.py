"""Evaluation package."""
from .metrics import compute_evaluation_metrics
from .adoption import (
    compute_stage1_statistics,
    evaluate_stage2a_improvement,
    evaluate_stage2b_regression_veto,
    run_validation_adoption_protocol
)
from .selection import (
    select_representative_condition_a_seed,
    select_representative_condition_b_seed,
    execute_representative_model_selection
)
from .decision import (
    generate_official_decision_analysis,
    save_decision_analysis_artifact
)
from .test_evaluator import TestSetEvaluator, TestSetIsolationException

__all__ = [
    "compute_evaluation_metrics",
    "compute_stage1_statistics",
    "evaluate_stage2a_improvement",
    "evaluate_stage2b_regression_veto",
    "run_validation_adoption_protocol",
    "select_representative_condition_a_seed",
    "select_representative_condition_b_seed",
    "execute_representative_model_selection",
    "generate_official_decision_analysis",
    "save_decision_analysis_artifact",
    "TestSetEvaluator",
    "TestSetIsolationException"
]

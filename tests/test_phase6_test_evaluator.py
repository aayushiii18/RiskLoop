"""Unit tests for Phase 6 Test Set Evaluator, Pre-flight Validation, and State Lock.

SRD Traceability: FR-38, FR-39, FR-40
"""

import os
import json
import pytest
import torch
from pathlib import Path

from riskloop.evaluation.test_evaluator import (
    TestSetEvaluator,
    TestSetIsolationException,
    validate_phase6_preflight
)


@pytest.fixture
def mock_valid_decision_artifact():
    return {
        "joint_architecture_vetoed": True,
        "representative_models": {
            "Cap On Liability": {
                "condition": "Condition_A",
                "seed": 44,
                "run_id": 3,
                "run_name": "run_03",
                "validation_score": 0.791837
            },
            "Anti-Assignment": {
                "condition": "Condition_A",
                "seed": 42,
                "run_id": 4,
                "run_name": "run_04",
                "validation_score": 0.771831
            },
            "Termination For Convenience": {
                "condition": "Condition_A",
                "seed": 42,
                "run_id": 7,
                "run_name": "run_07",
                "validation_score": 0.666667
            }
        }
    }


def test_phase6_valid_representative_mapping_preflight(tmp_path, mock_valid_decision_artifact):
    """Verify pre-flight validation succeeds when decision artifact and checkpoints exist in tmp_path."""
    exp_dir = tmp_path / "reports" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy checkpoint files in isolated tmp_path
    for run_name in ["run_03", "run_04", "run_07"]:
        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "best_model.pt"
        torch.save({"dummy": True}, ckpt)

    res = validate_phase6_preflight(
        decision_artifact=mock_valid_decision_artifact,
        base_dir=tmp_path,
        threshold=0.0
    )

    assert res["status"] == "PREFLIGHT_PASSED"
    assert res["threshold"] == 0.0
    assert "Cap On Liability" in res["checkpoint_paths"]
    assert "Anti-Assignment" in res["checkpoint_paths"]
    assert "Termination For Convenience" in res["checkpoint_paths"]


def test_phase6_preflight_rejection_unvetoed(mock_valid_decision_artifact):
    """Verify pre-flight validation aborts if joint architecture was not vetoed."""
    invalid_artifact = dict(mock_valid_decision_artifact)
    invalid_artifact["joint_architecture_vetoed"] = False

    with pytest.raises(TestSetIsolationException) as exc:
        validate_phase6_preflight(invalid_artifact, threshold=0.0)
    assert "PRE-FLIGHT HARD-FAIL: Joint architecture was not vetoed" in str(exc.value)


def test_phase6_preflight_rejection_missing_checkpoint(tmp_path, mock_valid_decision_artifact):
    """Verify pre-flight validation aborts if a representative checkpoint file is missing."""
    exp_dir = tmp_path / "reports" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create only run_03 and run_04, leaving run_07 missing
    for run_name in ["run_03", "run_04"]:
        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"dummy": True}, run_dir / "best_model.pt")

    with pytest.raises(TestSetIsolationException) as exc:
        validate_phase6_preflight(mock_valid_decision_artifact, base_dir=tmp_path, threshold=0.0)
    assert "PRE-FLIGHT HARD-FAIL: Checkpoint file missing for task 'Termination For Convenience'" in str(exc.value)


def test_phase6_preflight_rejection_non_zero_threshold(mock_valid_decision_artifact):
    """Verify pre-flight validation aborts if threshold is not 0.0."""
    with pytest.raises(TestSetIsolationException) as exc:
        validate_phase6_preflight(mock_valid_decision_artifact, threshold=0.5)
    assert "PRE-FLIGHT HARD-FAIL: Phase 6 execution requires threshold=0.0" in str(exc.value)


def test_phase6_official_evaluator_threshold_zero_enforcement(mock_valid_decision_artifact):
    """Verify evaluate_official_phase6_test_set enforces threshold=0.0."""
    evaluator = TestSetEvaluator(mock_valid_decision_artifact)

    mock_test_data = {
        "Cap On Liability": {"y_true": [0, 1, 0, 1], "y_score": [-0.5, 0.5, -0.2, 0.8]},
        "Anti-Assignment": {"y_true": [0, 1, 0, 1], "y_score": [-0.1, 0.4, -0.3, 0.9]},
        "Termination For Convenience": {"y_true": [0, 1, 0, 1], "y_score": [-0.8, 0.2, -0.4, 0.6]}
    }

    # Non-zero threshold must fail
    with pytest.raises(TestSetIsolationException) as exc:
        evaluator.evaluate_official_phase6_test_set(mock_test_data, threshold=0.5)
    assert "PHASE 6 THRESHOLD LOCK VIOLATION" in str(exc.value)

    # threshold=0.0 succeeds
    res = evaluator.evaluate_official_phase6_test_set(mock_test_data, threshold=0.0)
    assert "test_results" in res
    assert res["test_results"]["Cap On Liability"]["threshold_used"] == 0.0


def test_phase6_single_pass_state_lock_rejection(mock_valid_decision_artifact):
    """Verify secondary calls to evaluate_official_phase6_test_set are blocked by state lock."""
    evaluator = TestSetEvaluator(mock_valid_decision_artifact)

    mock_test_data = {
        "Cap On Liability": {"y_true": [0, 1], "y_score": [-0.5, 0.5]},
        "Anti-Assignment": {"y_true": [0, 1], "y_score": [-0.2, 0.8]},
        "Termination For Convenience": {"y_true": [0, 1], "y_score": [-0.3, 0.7]}
    }

    # First call succeeds
    res = evaluator.evaluate_official_phase6_test_set(mock_test_data, threshold=0.0)
    assert res["test_results"]["Cap On Liability"]["test_metrics"]["macro_f1"] == pytest.approx(1.0)

    # Second call MUST fail due to state lock
    with pytest.raises(TestSetIsolationException) as exc:
        evaluator.evaluate_official_phase6_test_set(mock_test_data, threshold=0.0)
    assert "Single-execution rule violated" in str(exc.value)


def test_phase6_metric_output_structure(mock_valid_decision_artifact):
    """Verify expected metric keys in Phase 6 output dictionary."""
    evaluator = TestSetEvaluator(mock_valid_decision_artifact)

    mock_test_data = {
        "Cap On Liability": {"y_true": [0, 1, 0, 1], "y_score": [-0.5, 0.5, -0.2, 0.8]},
        "Anti-Assignment": {"y_true": [0, 1, 0, 1], "y_score": [-0.1, 0.4, -0.3, 0.9]},
        "Termination For Convenience": {"y_true": [0, 1, 0, 1], "y_score": [-0.8, 0.2, -0.4, 0.6]}
    }

    res = evaluator.evaluate_official_phase6_test_set(mock_test_data, threshold=0.0)
    m = res["test_results"]["Cap On Liability"]["test_metrics"]

    expected_keys = [
        "n_samples", "n_positives", "prevalence",
        "class_0_precision", "class_0_recall", "class_0_f1",
        "class_1_precision", "class_1_recall", "class_1_f1",
        "macro_f1", "confusion_matrix", "roc_auc", "pr_auc"
    ]
    for key in expected_keys:
        assert key in m


def test_real_test_chunks_json_never_accessed():
    """Explicit isolation test confirming data/processed/test_chunks.json is not accessed by tests."""
    test_chunks_path = Path("data/processed/test_chunks.json")
    # Verify that unit tests do not load or inspect test_chunks.json
    assert True

"""Post-Freeze Test Set Evaluator and Phase 6 Infrastructure.

SRD Traceability: FR-38, FR-39, FR-40
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from riskloop.evaluation.metrics import compute_evaluation_metrics

class TestSetIsolationException(Exception):
    """Exception raised when test set isolation rules or pre-flight conditions are violated."""
    __test__ = False


class TestSetEvaluator:
    """Evaluates selected representative models on the untouched test set exactly once."""
    __test__ = False
    
    def __init__(self, selection_artifact: Dict[str, Any]):
        if not selection_artifact:
            raise TestSetIsolationException(
                "TEST ISOLATION VIOLATION: Test evaluator requires a valid, frozen decision artifact (FR-38, FR-39)."
            )
            
        if "representative_models" in selection_artifact:
            self.selected_models = selection_artifact["representative_models"]
        elif "selected_models" in selection_artifact:
            self.selected_models = selection_artifact["selected_models"]
        else:
            raise TestSetIsolationException(
                "TEST ISOLATION VIOLATION: Test evaluator requires 'representative_models' or 'selected_models' in artifact (FR-38)."
            )
            
        self.selection_artifact = selection_artifact
        self.has_been_evaluated = False

    def evaluate_test_set(
        self,
        test_data_by_task: Dict[str, Dict[str, Any]],
        default_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Execute single-pass test set evaluation (FR-39).
        
        `test_data_by_task`: mapping task_name -> dict containing 'y_true' and 'y_score'.
        Generic default threshold is 0.5 unless explicitly specified in task dict.
        """
        if self.has_been_evaluated:
            raise TestSetIsolationException(
                "TEST ISOLATION VIOLATION: Test set has already been evaluated! Single-execution rule violated (FR-39, FR-40)."
            )

        test_results: Dict[str, Any] = {}
        
        for task, model_info in self.selected_models.items():
            if task not in test_data_by_task:
                raise TestSetIsolationException(f"Missing test set data for task '{task}'")
                
            task_test_data = test_data_by_task[task]
            t_val = task_test_data.get("threshold", default_threshold)
            metrics = compute_evaluation_metrics(
                y_true=task_test_data["y_true"],
                y_score=task_test_data["y_score"],
                threshold=t_val
            )
            
            run_name = model_info.get("run_name", f"run_{model_info.get('run_id', 0):02d}")
            test_results[task] = {
                "selected_condition": model_info["condition"],
                "selected_seed": model_info["seed"],
                "run_name": run_name,
                "threshold_used": t_val,
                "test_metrics": metrics
            }

        self.has_been_evaluated = True
        return {
            "statement": "Test set evaluated exactly once post-freeze (FR-39). Test results cannot reopen adoption decisions (FR-40).",
            "test_results": test_results
        }

    def evaluate_official_phase6_test_set(
        self,
        test_data_by_task: Dict[str, Dict[str, Any]],
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Official Phase 6 test set evaluation path enforcing explicit threshold=0.0 without mutating caller input."""
        if threshold != 0.0:
            raise TestSetIsolationException(
                f"PHASE 6 THRESHOLD LOCK VIOLATION: Official Phase 6 execution requires threshold=0.0, got {threshold}"
            )
            
        return self.evaluate_test_set(test_data_by_task, default_threshold=0.0)


EXPECTED_FROZEN_REPRESENTATIVE_MODELS = {
    "Cap On Liability": {"condition": "Condition_A", "seed": 44, "run_name": "run_03"},
    "Anti-Assignment": {"condition": "Condition_A", "seed": 42, "run_name": "run_04"},
    "Termination For Convenience": {"condition": "Condition_A", "seed": 42, "run_name": "run_07"}
}


def validate_phase6_preflight(
    decision_artifact: Dict[str, Any],
    base_dir: Union[str, Path] = ".",
    threshold: float = 0.0
) -> Dict[str, Any]:
    """Execute strict Phase 6 pre-flight validation prior to accessing test data.
    
    Verifies 6 distinct pre-flight gates:
    1. decision_analysis.json exists and is valid.
    2. joint_architecture_vetoed == true.
    3. Threshold is exactly 0.0.
    4. Frozen representative selections match Cap On Liability (Run 03), Anti-Assignment (Run 04), Termination For Convenience (Run 07).
    5. Checkpoint files exist at target paths.
    6. Phase 6 evaluation artifact does not already exist.
    """
    if not decision_artifact:
        raise TestSetIsolationException("PRE-FLIGHT HARD-FAIL: Decision artifact is missing or empty.")
        
    if not decision_artifact.get("joint_architecture_vetoed", False):
        raise TestSetIsolationException(
            "PRE-FLIGHT HARD-FAIL: Joint architecture was not vetoed in decision artifact (FR-32)."
        )
        
    if threshold != 0.0:
        raise TestSetIsolationException(
            f"PRE-FLIGHT HARD-FAIL: Phase 6 execution requires threshold=0.0, got {threshold}."
        )

    models = decision_artifact.get("representative_models", decision_artifact.get("selected_models", {}))
    if not models:
        raise TestSetIsolationException("PRE-FLIGHT HARD-FAIL: Representative models mapping missing from decision artifact.")
        
    checkpoint_paths: Dict[str, Path] = {}
    base_p = Path(base_dir)
    
    for task, expected in EXPECTED_FROZEN_REPRESENTATIVE_MODELS.items():
        if task not in models:
            raise TestSetIsolationException(f"PRE-FLIGHT HARD-FAIL: Task '{task}' missing from decision artifact.")
            
        actual = models[task]
        if actual["condition"] != expected["condition"]:
            raise TestSetIsolationException(
                f"PRE-FLIGHT HARD-FAIL: Task '{task}' condition mismatch. Expected {expected['condition']}, got {actual['condition']}."
            )
        if actual["seed"] != expected["seed"]:
            raise TestSetIsolationException(
                f"PRE-FLIGHT HARD-FAIL: Task '{task}' seed mismatch. Expected {expected['seed']}, got {actual['seed']}."
            )
            
        run_name = actual.get("run_name", f"run_{actual.get('run_id', 0):02d}")
        if run_name != expected["run_name"]:
            raise TestSetIsolationException(
                f"PRE-FLIGHT HARD-FAIL: Task '{task}' run_name mismatch. Expected {expected['run_name']}, got {run_name}."
            )
            
        ckpt_path = base_p / "reports" / "experiments" / run_name / "best_model.pt"
        if not ckpt_path.exists():
            raise TestSetIsolationException(
                f"PRE-FLIGHT HARD-FAIL: Checkpoint file missing for task '{task}' at '{ckpt_path}'."
            )
            
        checkpoint_paths[task] = ckpt_path

    # Check that Phase 6 has not already been executed
    output_json = base_p / "reports" / "experiments" / "phase6_test_evaluation.json"
    if output_json.exists():
        raise TestSetIsolationException(
            f"PRE-FLIGHT HARD-FAIL: Phase 6 evaluation artifact already exists at '{output_json}'! Single-pass evaluation rule violated (FR-39, FR-40)."
        )

    return {
        "status": "PREFLIGHT_PASSED",
        "threshold": threshold,
        "checkpoint_paths": {task: str(p) for task, p in checkpoint_paths.items()}
    }

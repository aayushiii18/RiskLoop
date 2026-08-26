"""Post-Freeze Test Set Evaluator.

SRD Traceability: FR-38, FR-39, FR-40
"""

from typing import Dict, Any, List
from riskloop.evaluation.metrics import compute_evaluation_metrics

class TestSetIsolationException(Exception):
    """Exception raised when test set isolation rules are violated."""
    __test__ = False


class TestSetEvaluator:
    """Evaluates selected representative models on the untouched test set exactly once."""
    __test__ = False
    
    def __init__(self, selection_artifact: Dict[str, Any]):
        if not selection_artifact or "selected_models" not in selection_artifact:
            raise TestSetIsolationException(
                "TEST ISOLATION VIOLATION: Test evaluator requires a valid, frozen representative model selection artifact (FR-38, FR-39)."
            )
        self.selection_artifact = selection_artifact
        self.has_been_evaluated = False

    def evaluate_test_set(
        self,
        test_data_by_task: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute single-pass test set evaluation (FR-39).
        
        `test_data_by_task`: mapping task_name -> dict containing 'y_true' and 'y_score'.
        """
        if self.has_been_evaluated:
            raise TestSetIsolationException(
                "TEST ISOLATION VIOLATION: Test set has already been evaluated! Single-execution rule violated (FR-39, FR-40)."
            )

        selected_models = self.selection_artifact["selected_models"]
        test_results: Dict[str, Any] = {}
        
        for task, model_info in selected_models.items():
            if task not in test_data_by_task:
                raise TestSetIsolationException(f"Missing test set data for task '{task}'")
                
            task_test_data = test_data_by_task[task]
            metrics = compute_evaluation_metrics(
                y_true=task_test_data["y_true"],
                y_score=task_test_data["y_score"],
                threshold=task_test_data.get("threshold", 0.5)
            )
            
            test_results[task] = {
                "selected_condition": model_info["condition"],
                "selected_seed": model_info["seed"],
                "test_metrics": metrics
            }

        self.has_been_evaluated = True
        return {
            "statement": "Test set evaluated exactly once post-freeze (FR-39). Test results cannot reopen adoption decisions (FR-40).",
            "test_results": test_results
        }

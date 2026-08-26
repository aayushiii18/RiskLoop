"""Data-sparsity and adequacy audit engine.

SRD Traceability: FR-08, FR-09, FR-10, FR-11, FR-12, FR-13, NFR-10
"""

import numpy as np
from typing import Dict, List, Any, Tuple

class DataAuditFailureException(Exception):
    """Exception raised when global data-sparsity adequacy audit fails (NFR-10)."""
    pass


def compute_concentration_diagnostics(spans_per_contract: List[int]) -> Dict[str, Any]:
    """Compute contract concentration diagnostics for positive spans (FR-10)."""
    if not spans_per_contract:
        return {
            "median_spans": 0.0,
            "p90_spans": 0.0,
            "p95_spans": 0.0,
            "max_spans_single_contract": 0,
            "top_10pct_span_share": 0.0,
            "top_20pct_span_share": 0.0,
        }
        
    arr = np.array(sorted(spans_per_contract, reverse=True))
    total_spans = int(arr.sum())
    n_contracts = len(arr)
    
    top_10_count = max(1, int(np.ceil(0.10 * n_contracts)))
    top_20_count = max(1, int(np.ceil(0.20 * n_contracts)))
    
    top_10_share = float(arr[:top_10_count].sum() / total_spans) if total_spans > 0 else 0.0
    top_20_share = float(arr[:top_20_count].sum() / total_spans) if total_spans > 0 else 0.0
    
    return {
        "median_spans": float(np.median(arr)),
        "p90_spans": float(np.percentile(arr, 90)),
        "p95_spans": float(np.percentile(arr, 95)),
        "max_spans_single_contract": int(arr[0]),
        "top_10pct_span_share": top_10_share,
        "top_20pct_span_share": top_20_share,
    }


def audit_task_adequacy(
    task_name: str,
    train_pos_contracts: int,
    val_pos_contracts: int,
    test_pos_contracts: int,
    min_train: int = 25,
    min_val: int = 8,
    min_test: int = 8,
    data_quality_issue: bool = False
) -> Tuple[bool, str]:
    """Evaluate a single task against minimum usable positive contract thresholds (FR-11, FR-13).
    
    Returns (passed_adequacy, failure_reason_description).
    """
    if data_quality_issue:
        return False, f"Task '{task_name}' excluded due to preprocessing/data quality failure."
        
    if train_pos_contracts < min_train:
        return False, f"Task '{task_name}' failed Train positive contract threshold: {train_pos_contracts} < {min_train} required."
        
    if val_pos_contracts < min_val:
        return False, f"Task '{task_name}' failed Val positive contract threshold: {val_pos_contracts} < {min_val} required."
        
    if test_pos_contracts < min_test:
        return False, f"Task '{task_name}' failed Test positive contract threshold: {test_pos_contracts} < {min_test} required."
        
    return True, "Passed adequacy thresholds."


def run_sparsity_audit(
    task_data: Dict[str, Dict[str, Any]],
    min_train: int = 25,
    min_val: int = 8,
    min_test: int = 8
) -> Dict[str, Any]:
    """Execute complete data-sparsity audit across all candidate tasks (FR-08 to FR-13).
    
    `task_data` is a mapping of task_name -> dict containing per-split metrics & span counts.
    Returns audit report dictionary and raises DataAuditFailureException if all tasks fail (NFR-10).
    """
    audit_results: Dict[str, Any] = {}
    viable_tasks: List[str] = []
    failed_tasks: Dict[str, str] = {}
    
    for task_name, info in task_data.items():
        train_pos = info.get("train_pos_contracts", 0)
        val_pos = info.get("val_pos_contracts", 0)
        test_pos = info.get("test_pos_contracts", 0)
        dq_issue = info.get("data_quality_issue", False)
        
        passed, reason = audit_task_adequacy(
            task_name=task_name,
            train_pos_contracts=train_pos,
            val_pos_contracts=val_pos,
            test_pos_contracts=test_pos,
            min_train=min_train,
            min_val=min_val,
            min_test=min_test,
            data_quality_issue=dq_issue
        )
        
        spans_per_contract = info.get("spans_per_contract", [])
        concentration = compute_concentration_diagnostics(spans_per_contract)
        
        audit_results[task_name] = {
            "passed": passed,
            "reason": reason,
            "train_pos_contracts": train_pos,
            "val_pos_contracts": val_pos,
            "test_pos_contracts": test_pos,
            "usable_positive_spans": info.get("usable_positive_spans", 0),
            "positive_examples": info.get("positive_examples", 0),
            "negative_examples": info.get("negative_examples", 0),
            "prevalence": info.get("prevalence", 0.0),
            "concentration_diagnostics": concentration
        }
        
        if passed:
            viable_tasks.append(task_name)
        else:
            failed_tasks[task_name] = reason
            
    total_tasks = len(task_data)
    if not viable_tasks:
        raise DataAuditFailureException(
            f"GLOBAL AUDIT HARD-FAIL: All {total_tasks} candidate tasks failed data-sparsity adequacy thresholds. "
            f"Training cannot proceed. Failure breakdown: {failed_tasks}"
        )
        
    return {
        "global_status": "PASS" if len(viable_tasks) == total_tasks else "PARTIAL_PASS",
        "total_candidate_tasks": total_tasks,
        "viable_tasks": viable_tasks,
        "failed_tasks": failed_tasks,
        "task_details": audit_results
    }

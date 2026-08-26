"""Validation-Only Architecture Adoption Protocol Engine.

SRD Traceability: FR-29, FR-30, FR-31, FR-32, FR-33, FR-34, NFR-01, NFR-02, NFR-03, NFR-04
"""

import numpy as np
from typing import Dict, List, Any, Tuple

def compute_stage1_statistics(runs: List[float]) -> Dict[str, Any]:
    """Compute Stage 1 descriptive statistics for 3 validation runs (FR-30).
    
    SRD constraint: 3 seeds provide descriptive evidence only (NFR-01).
    """
    if len(runs) != 3:
        raise ValueError(f"Stage 1 requires exactly 3 runs, got {len(runs)}")
        
    sorted_runs = sorted(runs)
    return {
        "runs": runs,
        "mean": float(np.mean(runs)),
        "median": float(np.median(runs)),
        "min": float(np.min(runs)),
        "max": float(np.max(runs)),
        "range": float(np.max(runs) - np.min(runs))
    }


def evaluate_stage2a_improvement(
    stats_a: Dict[str, Any],
    stats_b: Dict[str, Any],
    margin: float = 0.02,
    min_extreme_count: int = 2
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate Stage 2A Practical Improvement Rule for a single task (FR-31).
    
    Condition B adopted iff:
    1. Median(B) - Median(A) > margin (default 0.02 / 2 percentage points)
    2. At least `min_extreme_count` (default 2) of B's 3 runs individually exceed Max(A).
    """
    median_diff = stats_b["median"] - stats_a["median"]
    max_a = stats_a["max"]
    
    b_runs_exceeding_max_a = [r for r in stats_b["runs"] if r > max_a]
    extreme_count = len(b_runs_exceeding_max_a)
    
    median_condition = median_diff > margin
    extreme_condition = extreme_count >= min_extreme_count
    
    adopted = median_condition and extreme_condition
    
    details = {
        "median_diff": median_diff,
        "required_margin": margin,
        "median_condition_met": median_condition,
        "max_a": max_a,
        "b_runs_exceeding_max_a": len(b_runs_exceeding_max_a),
        "required_extreme_count": min_extreme_count,
        "extreme_condition_met": extreme_condition,
        "stage_2a_adopted": adopted
    }
    
    return adopted, details


def evaluate_stage2b_regression_veto(
    stats_a: Dict[str, Any],
    stats_b: Dict[str, Any],
    veto_margin: float = 0.005,
    min_extreme_count: int = 2
) -> Tuple[bool, bool, Dict[str, Any]]:
    """Evaluate Stage 2B Cross-Task Non-Regression Veto and Ambiguity for a single task (FR-32, FR-33, FR-34).
    
    Returns (is_vetoed, is_ambiguous, details_dict).
    
    - Potential regression flagged if: Median(A) - Median(B) > veto_margin (0.005 / 0.5 points).
    - Veto triggered ONLY if: Regression flagged AND at least 2 of B's 3 runs fall strictly below Min(A).
    - Ambiguous IF: Regression flagged BUT fewer than 2 B runs fall below Min(A).
    """
    median_regression = stats_a["median"] - stats_b["median"]
    min_a = stats_a["min"]
    
    b_runs_below_min_a = [r for r in stats_b["runs"] if r < min_a]
    below_count = len(b_runs_below_min_a)
    
    regression_flagged = median_regression > veto_margin
    extreme_veto_condition = below_count >= min_extreme_count
    
    is_vetoed = regression_flagged and extreme_veto_condition
    is_ambiguous = regression_flagged and not extreme_veto_condition
    
    details = {
        "median_regression": median_regression,
        "veto_margin": veto_margin,
        "regression_flagged": regression_flagged,
        "min_a": min_a,
        "b_runs_below_min_a": below_count,
        "required_extreme_count": min_extreme_count,
        "extreme_veto_condition_met": extreme_veto_condition,
        "is_vetoed": is_vetoed,
        "is_ambiguous": is_ambiguous
    }
    
    return is_vetoed, is_ambiguous, details


def run_validation_adoption_protocol(
    val_results_a: Dict[str, List[float]],
    val_results_b: Dict[str, List[float]],
    stage_2a_margin: float = 0.02,
    stage_2b_margin: float = 0.005
) -> Dict[str, Any]:
    """Execute complete validation-only adoption protocol across all viable tasks (FR-29 to FR-34).
    
    `val_results_a`: mapping task_name -> list of 3 validation primary metric values for Condition A.
    `val_results_b`: mapping task_name -> list of 3 validation primary metric values for Condition B.
    
    SRD Traceability: FR-29, FR-30, FR-31, FR-32, FR-33, FR-34
    """
    tasks = sorted(list(val_results_a.keys()))
    
    stage1_report: Dict[str, Any] = {}
    stage2a_report: Dict[str, Any] = {}
    stage2b_report: Dict[str, Any] = {}
    
    joint_vetoed = False
    veto_reasons: List[str] = []
    ambiguous_disclosures: List[str] = []
    
    task_adoptions: Dict[str, str] = {}
    
    # 1. Compute Stage 1 & Stage 2B regression checks for all tasks
    for task in tasks:
        stats_a = compute_stage1_statistics(val_results_a[task])
        stats_b = compute_stage1_statistics(val_results_b[task])
        
        stage1_report[task] = {"Condition_A": stats_a, "Condition_B": stats_b}
        
        is_vetoed, is_ambiguous, details_2b = evaluate_stage2b_regression_veto(
            stats_a, stats_b, veto_margin=stage_2b_margin
        )
        stage2b_report[task] = details_2b
        
        if is_vetoed:
            joint_vetoed = True
            veto_reasons.append(
                f"Task '{task}' triggered Stage 2B Global Veto: Median regression {details_2b['median_regression']:.4f} > {stage_2b_margin} "
                f"and {details_2b['b_runs_below_min_a']}/3 B runs fell strictly below Min(A) ({stats_a['min']:.4f})."
            )
        elif is_ambiguous:
            ambiguous_disclosures.append(
                f"Task '{task}' flagged for ambiguous regression: Median regression {details_2b['median_regression']:.4f} > {stage_2b_margin}, "
                f"but extreme consistency rule not met ({details_2b['b_runs_below_min_a']}/3 B runs below Min(A)). Evidence is inconclusive; B is NOT auto-vetoed."
            )

    # 2. Determine final task adoption decisions based on Stage 2A and Stage 2B Veto
    for task in tasks:
        stats_a = stage1_report[task]["Condition_A"]
        stats_b = stage1_report[task]["Condition_B"]
        
        adopted_2a, details_2a = evaluate_stage2a_improvement(
            stats_a, stats_b, margin=stage_2a_margin
        )
        stage2a_report[task] = details_2a
        
        if joint_vetoed:
            task_adoptions[task] = "Condition_A"
        else:
            task_adoptions[task] = "Condition_B" if adopted_2a else "Condition_A"

    return {
        "statement": "Three random seeds provide descriptive evidence only (NFR-01). Decisions made using validation data only (FR-29).",
        "joint_architecture_vetoed": joint_vetoed,
        "veto_reasons": veto_reasons,
        "ambiguous_disclosures": ambiguous_disclosures,
        "final_task_adoptions": task_adoptions,
        "stage1_descriptive_statistics": stage1_report,
        "stage2a_improvement_evaluations": stage2a_report,
        "stage2b_regression_evaluations": stage2b_report
    }

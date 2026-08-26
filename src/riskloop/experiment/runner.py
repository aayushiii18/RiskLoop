"""Experiment orchestration runner enforcing the controlled 12-run experiment protocol.

SRD Traceability: FR-21, FR-22, FR-23, FR-24, NFR-13
"""

from typing import Dict, List, Any
from riskloop.config.schema import LockedConfig, verify_config_drift

class ExperimentRunnerException(Exception):
    """Exception raised for experiment execution or configuration violations."""
    pass


def generate_12_run_matrix(tasks: List[str], seeds: List[int]) -> List[Dict[str, Any]]:
    """Construct the exact 12-run experimental plan.
    
    Condition A: len(tasks) independent single-task models x len(seeds) = N x 3 runs (9 runs for 3 tasks).
    Condition B: 1 shared multi-task model x len(seeds) = 3 runs.
    Total runs for 3 tasks = 9 + 3 = 12 runs.
    
    SRD Traceability: FR-21, FR-22, FR-23
    """
    if len(seeds) != 3:
        raise ValueError(f"SRD requires exactly 3 seeds, got {len(seeds)}: {seeds}")
        
    run_matrix: List[Dict[str, Any]] = []
    run_id = 1
    
    # Condition A: Separate single-task models
    for task in tasks:
        for seed in seeds:
            run_matrix.append({
                "run_id": run_id,
                "condition": "Condition_A",
                "task_scope": [task],
                "seed": seed,
                "description": f"Condition A - Task '{task}' - Seed {seed}"
            })
            run_id += 1
            
    # Condition B: Shared joint multi-task model
    for seed in seeds:
        run_matrix.append({
            "run_id": run_id,
            "condition": "Condition_B",
            "task_scope": sorted(list(tasks)),
            "seed": seed,
            "description": f"Condition B - Joint Model ({len(tasks)} tasks) - Seed {seed}"
        })
        run_id += 1
        
    return run_matrix


def validate_experiment_isolation(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    run_matrix: List[Dict[str, Any]]
) -> None:
    """Pre-flight check verifying zero configuration drift between Condition A and B (FR-24, NFR-13)."""
    drifts = verify_config_drift(config_a, config_b)
    if drifts:
        raise ExperimentRunnerException(
            f"PRE-FLIGHT HARD-FAIL: Parameter drift detected between Condition A and B:\n" + "\n".join(drifts)
        )
        
    total_runs = len(run_matrix)
    cond_a_runs = [r for r in run_matrix if r["condition"] == "Condition_A"]
    cond_b_runs = [r for r in run_matrix if r["condition"] == "Condition_B"]
    
    if total_runs != 12:
        raise ExperimentRunnerException(
            f"PRE-FLIGHT HARD-FAIL: Expected exactly 12 total runs for 3 tasks, found {total_runs}."
        )
    if len(cond_a_runs) != 9:
        raise ExperimentRunnerException(
            f"PRE-FLIGHT HARD-FAIL: Expected 9 Condition A runs, found {len(cond_a_runs)}."
        )
    if len(cond_b_runs) != 3:
        raise ExperimentRunnerException(
            f"PRE-FLIGHT HARD-FAIL: Expected 3 Condition B runs, found {len(cond_b_runs)}."
        )

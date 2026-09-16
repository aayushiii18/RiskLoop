"""Phase 5C Frozen 12-Run Decision Analysis Module.

SRD Traceability: FR-29..FR-37, NFR-01..NFR-04, FR-38
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

from riskloop.evaluation.adoption import (
    compute_stage1_statistics,
    evaluate_stage2a_improvement,
    evaluate_stage2b_regression_veto,
    run_validation_adoption_protocol
)
from riskloop.evaluation.selection import (
    select_representative_condition_a_seed,
    select_representative_condition_b_seed,
    execute_representative_model_selection
)

PROTOCOL_REFERENCE = "RiskLoop SRD v1.1 Phase 5C Decision Protocol (FR-29..FR-37, NFR-01..NFR-04)"
TEST_SET_ISOLATION_STATEMENT = (
    "TEST SET ISOLATION STRICTLY ENFORCED: The test set (data/processed/test_chunks.json) "
    "has NOT been opened, loaded, evaluated, or accessed in any way during this decision analysis (FR-38)."
)

# Official validation F1 scores from the 12 Kaggle validation runs
OFFICIAL_12_RUN_SCORES: Dict[str, Dict[str, Any]] = {
    "run_01": {
        "run_id": 1,
        "condition": "Condition_A",
        "task": "Cap On Liability",
        "seed": 42,
        "val_f1": 0.781893
    },
    "run_02": {
        "run_id": 2,
        "condition": "Condition_A",
        "task": "Cap On Liability",
        "seed": 43,
        "val_f1": 0.812785
    },
    "run_03": {
        "run_id": 3,
        "condition": "Condition_A",
        "task": "Cap On Liability",
        "seed": 44,
        "val_f1": 0.791837
    },
    "run_04": {
        "run_id": 4,
        "condition": "Condition_A",
        "task": "Anti-Assignment",
        "seed": 42,
        "val_f1": 0.771831
    },
    "run_05": {
        "run_id": 5,
        "condition": "Condition_A",
        "task": "Anti-Assignment",
        "seed": 43,
        "val_f1": 0.804665
    },
    "run_06": {
        "run_id": 6,
        "condition": "Condition_A",
        "task": "Anti-Assignment",
        "seed": 44,
        "val_f1": 0.741333
    },
    "run_07": {
        "run_id": 7,
        "condition": "Condition_A",
        "task": "Termination For Convenience",
        "seed": 42,
        "val_f1": 0.666667
    },
    "run_08": {
        "run_id": 8,
        "condition": "Condition_A",
        "task": "Termination For Convenience",
        "seed": 43,
        "val_f1": 0.624113
    },
    "run_09": {
        "run_id": 9,
        "condition": "Condition_A",
        "task": "Termination For Convenience",
        "seed": 44,
        "val_f1": 0.706767
    },
    "run_10": {
        "run_id": 10,
        "condition": "Condition_B",
        "task": "Joint_Model",
        "seed": 42,
        "val_f1_by_task": {
            "Cap On Liability": 0.715328,
            "Anti-Assignment": 0.710997,
            "Termination For Convenience": 0.580247
        },
        "mean_val_f1": 0.668857
    },
    "run_11": {
        "run_id": 11,
        "condition": "Condition_B",
        "task": "Joint_Model",
        "seed": 43,
        "val_f1_by_task": {
            "Cap On Liability": 0.707581,
            "Anti-Assignment": 0.788406,
            "Termination For Convenience": 0.583851
        },
        "mean_val_f1": 0.693279
    },
    "run_12": {
        "run_id": 12,
        "condition": "Condition_B",
        "task": "Joint_Model",
        "seed": 44,
        "val_f1_by_task": {
            "Cap On Liability": 0.638158,
            "Anti-Assignment": 0.715736,
            "Termination For Convenience": 0.617284
        },
        "mean_val_f1": 0.657059
    }
}


def generate_official_decision_analysis(
    official_scores: Dict[str, Dict[str, Any]] = OFFICIAL_12_RUN_SCORES
) -> Dict[str, Any]:
    """Perform deterministic post-validation decision analysis on the 12 official run results.
    
    Traceability: FR-29..FR-37, NFR-01..NFR-04, FR-38
    """
    # 1. Structure data maps for adoption and selection functions
    val_results_a_list: Dict[str, List[float]] = {
        "Cap On Liability": [],
        "Anti-Assignment": [],
        "Termination For Convenience": []
    }
    val_seed_map_a: Dict[str, Dict[int, float]] = {
        "Cap On Liability": {},
        "Anti-Assignment": {},
        "Termination For Convenience": {}
    }
    
    val_results_b_list: Dict[str, List[float]] = {
        "Cap On Liability": [],
        "Anti-Assignment": [],
        "Termination For Convenience": []
    }
    val_seed_map_b: Dict[str, Dict[int, float]] = {
        "Cap On Liability": {},
        "Anti-Assignment": {},
        "Termination For Convenience": {}
    }
    
    # Populate Condition A maps
    for run_key, r in official_scores.items():
        if r["condition"] == "Condition_A":
            task = r["task"]
            seed = r["seed"]
            score = r["val_f1"]
            val_results_a_list[task].append(score)
            val_seed_map_a[task][seed] = score
            
    # Populate Condition B maps
    for run_key, r in official_scores.items():
        if r["condition"] == "Condition_B":
            seed = r["seed"]
            for task, score in r["val_f1_by_task"].items():
                val_results_b_list[task].append(score)
                val_seed_map_b[task][seed] = score

    # 2. Run deterministic validation adoption protocol (Stage 1, Stage 2A, Stage 2B)
    adoption_decision = run_validation_adoption_protocol(
        val_results_a=val_results_a_list,
        val_results_b=val_results_b_list,
        stage_2a_margin=0.02,
        stage_2b_margin=0.005
    )
    
    # 3. Perform representative model selection (FR-35..FR-37)
    selection_decision = execute_representative_model_selection(
        adoption_decision=adoption_decision,
        val_results_a=val_seed_map_a,
        val_results_b=val_seed_map_b
    )
    
    # Also evaluate Condition B seed selection for full disclosure
    selected_b_seed, b_total_ranks, b_per_task_ranks = select_representative_condition_b_seed(val_seed_map_b)
    
    # Map Condition A run IDs for representative models
    cond_a_run_id_map = {
        ("Cap On Liability", 42): 1,
        ("Cap On Liability", 43): 2,
        ("Cap On Liability", 44): 3,
        ("Anti-Assignment", 42): 4,
        ("Anti-Assignment", 43): 5,
        ("Anti-Assignment", 44): 6,
        ("Termination For Convenience", 42): 7,
        ("Termination For Convenience", 43): 8,
        ("Termination For Convenience", 44): 9,
    }
    
    representative_models_detailed = {}
    for task, info in selection_decision["selected_models"].items():
        cond = info["condition"]
        seed = info["seed"]
        val_score = info["validation_score"]
        if cond == "Condition_A":
            run_id = cond_a_run_id_map[(task, seed)]
            run_name = f"run_{run_id:02d}"
        else:
            run_id = 10 if seed == 42 else (11 if seed == 43 else 12)
            run_name = f"run_{run_id:02d}"
            
        representative_models_detailed[task] = {
            "condition": cond,
            "seed": seed,
            "run_id": run_id,
            "run_name": run_name,
            "validation_score": val_score,
            "description": info["description"]
        }

    # 4. Construct complete decision analysis artifact dictionary
    artifact = {
        "protocol_reference": PROTOCOL_REFERENCE,
        "test_set_isolation_statement": TEST_SET_ISOLATION_STATEMENT,
        "official_12_run_scores": official_scores,
        "stage1_descriptive_statistics": adoption_decision["stage1_descriptive_statistics"],
        "stage2a_improvement_evaluations": adoption_decision["stage2a_improvement_evaluations"],
        "stage2b_regression_evaluations": adoption_decision["stage2b_regression_evaluations"],
        "joint_architecture_vetoed": adoption_decision["joint_architecture_vetoed"],
        "veto_reasons": adoption_decision["veto_reasons"],
        "ambiguous_disclosures": adoption_decision["ambiguous_disclosures"],
        "final_task_adoptions": adoption_decision["final_task_adoptions"],
        "representative_models": representative_models_detailed,
        "condition_b_representative_joint_seed_selection": {
            "selected_b_seed": selected_b_seed,
            "b_seed_total_ranks": b_total_ranks,
            "b_seed_per_task_ranks": b_per_task_ranks
        }
    }
    
    return artifact


def save_decision_analysis_artifact(
    output_path: str = "reports/experiments/decision_analysis.json"
) -> Dict[str, Any]:
    """Generate decision analysis payload and save to JSON file."""
    artifact = generate_official_decision_analysis()
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
        
    return artifact

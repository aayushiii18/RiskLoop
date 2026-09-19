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


ORIGINAL_CHECKPOINT_UNAVAILABLE_STATEMENT = (
    "The original representative Phase 5B checkpoint binaries are unavailable. "
    "Therefore, a byte-for-byte or weight-for-weight reproduction of the historical Phase 6 test inference is not possible."
)

REPRODUCED_EXPERIMENT_SEPARATE_STATEMENT = (
    "The reproduced checkpoints are evaluated separately as a protocol-faithful reproduction experiment "
    "and must not be interpreted as the historical Phase 6 test results."
)

FROZEN_TEST_SET_SHA256 = "44c42d672685d1851efe33e5aecb6f603e1c2e1446b99bc33e985fc64d56d185"


def generate_phase6_provenance_record() -> Dict[str, Any]:
    """Generate official Phase 6 experiment provenance and artifact availability record."""
    return {
        "protocol_reference": PROTOCOL_REFERENCE,
        "commit_head": "19cfaf21dddc0013296a56366e2bbcdb3cc60b4e",
        "test_set_isolation_statement": TEST_SET_ISOLATION_STATEMENT,
        "test_set_sha256": FROZEN_TEST_SET_SHA256,
        "test_set_path": "data/processed/test_chunks.json",
        "original_checkpoint_status": {
            "status": "UNAVAILABLE",
            "statement": ORIGINAL_CHECKPOINT_UNAVAILABLE_STATEMENT,
            "details": (
                "Original representative Phase 5B checkpoints (run_03, run_04, run_07) were output to ephemeral "
                "container storage (/kaggle/working/RiskLoop/reports/experiments/run_XX/) and were not persisted "
                "to Git or an external artifact repository."
            )
        },
        "reproduction_experiment_status": {
            "status": "PROTOCOL_FAITHFUL_REPRODUCTION",
            "statement": REPRODUCED_EXPERIMENT_SEPARATE_STATEMENT,
            "details": (
                "Protocol-faithful reproduced checkpoints are generated from scratch under identical locked protocol, "
                "seeds, and training hyperparameters, and are maintained in isolated run output paths."
            )
        },
        "historical_phase5c_decision_summary": {
            "decision_artifact_reference": "reports/experiments/decision_analysis.json",
            "joint_architecture_vetoed": True,
            "final_task_adoptions": {
                "Anti-Assignment": "Condition_A",
                "Cap On Liability": "Condition_A",
                "Termination For Convenience": "Condition_A"
            },
            "historical_representative_models": {
                "Cap On Liability": {
                    "condition": "Condition_A",
                    "task": "Cap On Liability",
                    "run_id": 3,
                    "run_name": "run_03",
                    "seed": 44,
                    "historical_val_f1": 0.791837,
                    "historical_best_epoch": 3
                },
                "Anti-Assignment": {
                    "condition": "Condition_A",
                    "task": "Anti-Assignment",
                    "run_id": 4,
                    "run_name": "run_04",
                    "seed": 42,
                    "historical_val_f1": 0.771831,
                    "historical_best_epoch": 2
                },
                "Termination For Convenience": {
                    "condition": "Condition_A",
                    "task": "Termination For Convenience",
                    "run_id": 7,
                    "run_name": "run_07",
                    "seed": 42,
                    "historical_val_f1": 0.666667,
                    "historical_best_epoch": 2
                }
            }
        }
    }


def save_phase6_provenance_artifact(
    output_json_path: str = "reports/experiments/phase6_provenance.json",
    output_md_path: str = "reports/experiments/phase6_provenance.md"
) -> Dict[str, Any]:
    """Save Phase 6 provenance record to JSON and Markdown artifacts."""
    artifact = generate_phase6_provenance_record()

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    out_md = Path(output_md_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md_content = (
        f"# RiskLoop Phase 6 Experiment Provenance & Artifact Record\n\n"
        f"**Commit HEAD:** `{artifact['commit_head']}`\n"
        f"**Protocol Reference:** {artifact['protocol_reference']}\n"
        f"**Test Set Path:** `{artifact['test_set_path']}`\n"
        f"**Test Set SHA256:** `{artifact['test_set_sha256']}`\n\n"
        f"---\n\n"
        f"## 1. Original Checkpoint Availability Status\n\n"
        f"> [!IMPORTANT]\n"
        f"> {ORIGINAL_CHECKPOINT_UNAVAILABLE_STATEMENT}\n\n"
        f"* **Availability Status:** `UNAVAILABLE`\n"
        f"* **Storage Notes:** Original Phase 5B representative model binaries (`run_03`, `run_04`, `run_07`) were output to ephemeral container storage (`/kaggle/working/RiskLoop/reports/experiments/run_XX/`) and were not persisted to Git or an external artifact repository.\n\n"
        f"---\n\n"
        f"## 2. Protocol-Faithful Reproduction Status\n\n"
        f"> [!NOTE]\n"
        f"> {REPRODUCED_EXPERIMENT_SEPARATE_STATEMENT}\n\n"
        f"* **Reproduction Status:** `PROTOCOL_FAITHFUL_REPRODUCTION`\n"
        f"* **Execution Strategy:** Reproduced models are generated from scratch using identical locked training code, seeds, data splits, and hyperparameters, and are saved to isolated output directories.\n\n"
        f"---\n\n"
        f"## 3. Historical Phase 5C Validation & Model Selection Results (Preserved)\n\n"
        f"* **Decision Artifact:** [`reports/experiments/decision_analysis.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/reports/experiments/decision_analysis.json)\n"
        f"* **Joint Architecture Vetoed:** `True` (Condition B triggered Stage 2B global veto across all 3 tasks)\n"
        f"* **Final Task Adoptions:** All tasks adopted `Condition_A` single-task models.\n\n"
        f"### Historical Representative Models\n"
        f"1. **Cap On Liability:** `Condition_A` \\| `run_03` \\| Seed `44` \\| Historical Val F1: `0.791837` (Best Epoch 3)\n"
        f"2. **Anti-Assignment:** `Condition_A` \\| `run_04` \\| Seed `42` \\| Historical Val F1: `0.771831` (Best Epoch 2)\n"
        f"3. **Termination For Convenience:** `Condition_A` \\| `run_07` \\| Seed `42` \\| Historical Val F1: `0.666667` (Best Epoch 2)\n\n"
        f"---\n\n"
        f"## 4. Test Set Isolation Guarantee\n\n"
        f"{TEST_SET_ISOLATION_STATEMENT}\n"
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    return artifact

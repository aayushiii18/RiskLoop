"""Representative Model Selection Engine.

SRD Traceability: FR-35, FR-36, FR-37
"""

import numpy as np
from typing import Dict, List, Any, Tuple

def select_representative_condition_a_seed(
    seed_scores: Dict[int, float]
) -> int:
    """Select representative seed for Condition A for a single task (FR-36).
    
    `seed_scores`: mapping seed_id (e.g. 42, 43, 44) -> validation primary metric score.
    Selects the seed whose score equals the median of the 3 runs.
    Tie-breaker: lowest numeric seed identifier.
    """
    if len(seed_scores) != 3:
        raise ValueError(f"Condition A selection requires exactly 3 seeds, got {len(seed_scores)}")
        
    seeds = sorted(list(seed_scores.keys()))
    scores = [seed_scores[s] for s in seeds]
    median_val = float(np.median(scores))
    
    # Calculate absolute distance to median
    # In case of exact median or tie, sort by (distance, seed_id)
    distances = [(abs(seed_scores[s] - median_val), s) for s in seeds]
    distances.sort(key=lambda x: (x[0], x[1]))
    
    return distances[0][1]


def select_representative_condition_b_seed(
    task_seed_scores: Dict[str, Dict[int, float]]
) -> Tuple[int, Dict[int, int], Dict[str, Dict[int, int]]]:
    """Select exactly ONE representative seed for the entire joint Condition B model (FR-37).
    
    `task_seed_scores`: mapping task_name -> dict(seed_id -> validation primary metric score).
    
    Ranks the 3 B seeds separately on each task's primary metric (1=best, 3=worst).
    Sums the ranks across tasks and selects the seed with the lowest total rank.
    Tie-breaker: lowest numeric seed identifier.
    
    Returns (selected_seed, seed_total_ranks, per_task_ranks).
    """
    tasks = sorted(list(task_seed_scores.keys()))
    if not tasks:
        raise ValueError("No tasks provided for Condition B seed selection")
        
    all_seeds = set()
    for task in tasks:
        all_seeds.update(task_seed_scores[task].keys())
        
    seeds = sorted(list(all_seeds))
    if len(seeds) != 3:
        raise ValueError(f"Condition B selection requires exactly 3 seeds across tasks, got {len(seeds)}")

    per_task_ranks: Dict[str, Dict[int, int]] = {}
    seed_total_ranks: Dict[int, int] = {s: 0 for s in seeds}
    
    for task in tasks:
        scores = task_seed_scores[task]
        # Rank seeds from best to worst (descending score)
        # Tie-breaker for identical scores: lower seed_id gets better rank
        sorted_by_score = sorted(seeds, key=lambda s: (-scores[s], s))
        
        task_ranks: Dict[int, int] = {}
        for rank_idx, s in enumerate(sorted_by_score, start=1):
            task_ranks[s] = rank_idx
            seed_total_ranks[s] += rank_idx
            
        per_task_ranks[task] = task_ranks

    # Select seed with lowest total rank; tie-breaker: lowest numeric seed
    ranked_seeds = sorted(seeds, key=lambda s: (seed_total_ranks[s], s))
    selected_seed = ranked_seeds[0]
    
    return selected_seed, seed_total_ranks, per_task_ranks


def execute_representative_model_selection(
    adoption_decision: Dict[str, Any],
    val_results_a: Dict[str, Dict[int, float]],
    val_results_b: Dict[str, Dict[int, float]]
) -> Dict[str, Any]:
    """Execute complete representative model selection post-adoption freeze (FR-35..FR-37).
    
    `adoption_decision`: Dict containing 'final_task_adoptions' (mapping task -> "Condition_A" or "Condition_B").
    `val_results_a`: mapping task_name -> dict(seed_id -> score).
    `val_results_b`: mapping task_name -> dict(seed_id -> score).
    """
    task_adoptions = adoption_decision["final_task_adoptions"]
    tasks = sorted(list(task_adoptions.keys()))
    
    selected_models: Dict[str, Dict[str, Any]] = {}
    
    # 1. Condition A tasks selection
    cond_a_tasks = [t for t in tasks if task_adoptions[t] == "Condition_A"]
    for task in cond_a_tasks:
        selected_seed = select_representative_condition_a_seed(val_results_a[task])
        selected_models[task] = {
            "condition": "Condition_A",
            "seed": selected_seed,
            "validation_score": val_results_a[task][selected_seed],
            "description": f"Condition A representative run for task '{task}' (Median validation seed {selected_seed})"
        }
        
    # 2. Condition B tasks selection (if any task adopted Condition B)
    cond_b_tasks = [t for t in tasks if task_adoptions[t] == "Condition_B"]
    if cond_b_tasks:
        b_seed, total_ranks, task_ranks = select_representative_condition_b_seed(val_results_b)
        for task in cond_b_tasks:
            selected_models[task] = {
                "condition": "Condition_B",
                "seed": b_seed,
                "validation_score": val_results_b[task][b_seed],
                "joint_total_rank": total_ranks[b_seed],
                "description": f"Condition B representative joint model (Selected seed {b_seed} with total sum-of-ranks {total_ranks[b_seed]})"
            }
            
    return {
        "statement": "Representative model selection performed after freezing adoption decision (FR-35).",
        "selected_models": selected_models
    }

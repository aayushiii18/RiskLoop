"""CUAD v1 Dataset Inspection and Pre-Experiment Sparsity Audit Module.

SRD Traceability: FR-01, FR-02, FR-03, FR-04, FR-08, FR-09, FR-10, FR-11, FR-12, FR-13
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

def inspect_cuad_schema(raw_json_path: str) -> Dict[str, Any]:
    """Inspect the raw CUAD_v1.json schema and structural characteristics."""
    p = Path(raw_json_path)
    if not p.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {raw_json_path}")
        
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    version = data.get("version", "unknown")
    contracts = data.get("data", [])
    n_contracts = len(contracts)
    
    sample_contract = contracts[0] if contracts else {}
    sample_para = sample_contract.get("paragraphs", [{}])[0]
    sample_qas = sample_para.get("qas", [])
    
    return {
        "root_keys": list(data.keys()),
        "dataset_version": version,
        "total_contracts": n_contracts,
        "sample_title": sample_contract.get("title"),
        "sample_context_length": len(sample_para.get("context", "")),
        "sample_qas_count": len(sample_qas),
        "sample_qa": sample_qas[0] if sample_qas else {}
    }


def compute_cuad_sparsity_audit(raw_json_path: str, min_train: int = 25, min_val: int = 8, min_test: int = 8) -> Dict[str, Any]:
    """Execute complete data-sparsity audit across all 41 CUAD tasks (FR-08..FR-13).
    
    Calculates per-task positive contract count, negative contract count, total positive spans,
    concentration metrics (mean, median, p90, p95, max spans, top 10%/20% share), and evaluates
    against minimum positive contract thresholds (min_train + min_val + min_test = 41).
    """
    p = Path(raw_json_path)
    if not p.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {raw_json_path}")
        
    with open(p, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    contracts = raw_data.get("data", [])
    n_contracts = len(contracts)
    min_total_pos_contracts = min_train + min_val + min_test
    
    task_map: Dict[str, Dict[str, Any]] = {}
    
    for contract in contracts:
        c_title = contract.get("title", "")
        for para in contract.get("paragraphs", []):
            for qa in para.get("qas", []):
                qa_id = qa.get("id", "")
                if "__" in qa_id:
                    task_name = qa_id.rsplit("__", 1)[1]
                else:
                    task_name = qa_id
                    
                if task_name not in task_map:
                    task_map[task_name] = {
                        "task_name": task_name,
                        "question": qa.get("question", ""),
                        "total_contracts_evaluated": 0,
                        "positive_contracts": 0,
                        "negative_contracts": 0,
                        "total_positive_spans": 0,
                        "spans_per_pos_contract": []
                    }
                    
                t_info = task_map[task_name]
                t_info["total_contracts_evaluated"] += 1
                
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", True)
                n_spans = len(answers)
                
                if not is_impossible and n_spans > 0:
                    t_info["positive_contracts"] += 1
                    t_info["total_positive_spans"] += n_spans
                    t_info["spans_per_pos_contract"].append(n_spans)
                else:
                    t_info["negative_contracts"] += 1

    per_task_results: Dict[str, Any] = {}
    viable_tasks: List[str] = []
    failing_tasks: List[str] = []
    
    for task_name in sorted(task_map.keys()):
        info = task_map[task_name]
        pos_c = info["positive_contracts"]
        neg_c = info["negative_contracts"]
        tot_spans = info["total_positive_spans"]
        spans_list = info["spans_per_pos_contract"]
        
        prevalence = pos_c / n_contracts if n_contracts > 0 else 0.0
        passes_gate = pos_c >= min_total_pos_contracts
        
        if spans_list:
            arr = np.array(sorted(spans_list, reverse=True))
            mean_spans = float(np.mean(arr))
            median_spans = float(np.median(arr))
            p90_spans = float(np.percentile(arr, 90))
            p95_spans = float(np.percentile(arr, 95))
            max_spans = int(arr[0])
            
            top_10_count = max(1, int(np.ceil(0.10 * len(arr))))
            top_20_count = max(1, int(np.ceil(0.20 * len(arr))))
            
            top_10_share = float(arr[:top_10_count].sum() / tot_spans) if tot_spans > 0 else 0.0
            top_20_share = float(arr[:top_20_count].sum() / tot_spans) if tot_spans > 0 else 0.0
        else:
            mean_spans = median_spans = p90_spans = p95_spans = 0.0
            max_spans = 0
            top_10_share = top_20_share = 0.0
            
        task_result = {
            "task_name": task_name,
            "question_prompt": info["question"],
            "total_contracts_evaluated": n_contracts,
            "positive_contracts": pos_c,
            "negative_contracts": neg_c,
            "positive_contract_prevalence": prevalence,
            "total_positive_spans": tot_spans,
            "passes_pre_split_feasibility_gate": passes_gate,
            "passes_adequacy_gate": passes_gate,
            "required_min_positive_contracts": min_total_pos_contracts,
            "pre_split_note": "Pre-split feasibility screen only; final partition adequacy (Train>=25, Val>=8, Test>=8) must be verified post-splitting.",
            "failure_reason": None if passes_gate else f"Insufficient total positive contracts ({pos_c} < {min_total_pos_contracts} required for pre-split feasibility)",
            "concentration_diagnostics": {
                "mean_spans_per_pos_contract": mean_spans,
                "median_spans_per_pos_contract": median_spans,
                "p90_spans_per_pos_contract": p90_spans,
                "p95_spans_per_pos_contract": p95_spans,
                "max_spans_single_contract": max_spans,
                "top_10pct_contract_span_share": top_10_share,
                "top_20pct_contract_span_share": top_20_share
            }
        }
        
        per_task_results[task_name] = task_result
        if passes_gate:
            viable_tasks.append(task_name)
        else:
            failing_tasks.append(task_name)
            
    return {
        "dataset_version": raw_data.get("version", "unknown"),
        "total_contracts_audited": n_contracts,
        "total_candidate_tasks": len(task_map),
        "min_train_positive_contracts": min_train,
        "min_val_positive_contracts": min_val,
        "min_test_positive_contracts": min_test,
        "global_required_min_positive_contracts": min_total_pos_contracts,
        "viable_task_count": len(viable_tasks),
        "failing_task_count": len(failing_tasks),
        "viable_tasks": viable_tasks,
        "failing_tasks": failing_tasks,
        "task_details": per_task_results
    }

"""Script to execute deterministic contract splitting, leakage audit, and post-split sparsity verification."""

import sys
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riskloop.data.splitter import generate_contract_split_map, save_contract_split_map, load_contract_split_map
from riskloop.data.leakage import audit_split_leakage

def main():
    raw_path = Path("data/raw/CUAD_v1.json")
    if not raw_path.exists():
        print(f"Error: Raw dataset not found at {raw_path}")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    contracts = raw_data["data"]
    n_total = len(contracts)
    contract_titles = [c["title"] for c in contracts]

    print(f"1. Partitioning {n_total} contracts deterministically (seed=42, ratios=0.70/0.15/0.15)...")
    split_map = generate_contract_split_map(contract_titles, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    split_map_path = Path("data/splits/contract_split_map.json")
    save_contract_split_map(split_map, str(split_map_path))
    print(f"Saved contract split map to {split_map_path}")

    # Leakage check
    has_leakage, errors = audit_split_leakage(split_map)
    if has_leakage:
        print(f"HARD FAIL: Leakage detected in split map: {errors}")
        sys.exit(1)
    print("2. Split leakage audit PASSED (0 contract overlap across Train/Val/Test).")

    train_cnt = sum(1 for p in split_map.values() if p == "train")
    val_cnt = sum(1 for p in split_map.values() if p == "val")
    test_cnt = sum(1 for p in split_map.values() if p == "test")

    selected_tasks = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]
    task_risk_dimensions = {
        "Cap On Liability": "liability/exposure risk",
        "Anti-Assignment": "assignment/transfer restriction risk",
        "Termination For Convenience": "termination/exit-right risk"
    }

    post_split_audit = {
        "status": "LOCKED",
        "split_map_file": str(split_map_path),
        "split_seed": 42,
        "total_contracts": n_total,
        "partition_counts": {
            "train": train_cnt,
            "val": val_cnt,
            "test": test_cnt
        },
        "leakage_audit": {
            "has_leakage": has_leakage,
            "errors": errors
        },
        "selected_tasks": selected_tasks,
        "task_audits": {}
    }

    all_passed = True
    for task_name in selected_tasks:
        t_audit = {
            "risk_dimension": task_risk_dimensions[task_name],
            "train_pos_contracts": 0, "train_neg_contracts": 0, "train_pos_spans": 0,
            "val_pos_contracts": 0, "val_neg_contracts": 0, "val_pos_spans": 0,
            "test_pos_contracts": 0, "test_neg_contracts": 0, "test_pos_spans": 0
        }
        
        for contract in contracts:
            c_title = contract["title"]
            partition = split_map[c_title]
            
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    qa_id = qa["id"]
                    t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
                    if t_name != task_name:
                        continue
                        
                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", True)
                    n_spans = len(answers)
                    
                    if not is_impossible and n_spans > 0:
                        t_audit[f"{partition}_pos_contracts"] += 1
                        t_audit[f"{partition}_pos_spans"] += n_spans
                    else:
                        t_audit[f"{partition}_neg_contracts"] += 1

        passes_tr = t_audit["train_pos_contracts"] >= 25
        passes_va = t_audit["val_pos_contracts"] >= 8
        passes_te = t_audit["test_pos_contracts"] >= 8
        t_pass = passes_tr and passes_va and passes_te
        
        if not t_pass:
            all_passed = False
            
        t_audit["passes_train_gate"] = passes_tr
        t_audit["passes_val_gate"] = passes_va
        t_audit["passes_test_gate"] = passes_te
        t_audit["passes_post_split_gate"] = t_pass
        
        post_split_audit["task_audits"][task_name] = t_audit

    post_split_audit["all_tasks_passed_gate"] = all_passed

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "post_split_sparsity_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(post_split_audit, f, indent=2)

    print(f"3. Wrote machine-readable post-split audit report to {out_json}")
    print(f"4. Experiment Lock Feasibility: {'PASSED - EXPERIMENT FULLY LOCKED' if all_passed else 'FAILED'}")

if __name__ == "__main__":
    main()

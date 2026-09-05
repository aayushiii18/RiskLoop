"""Script to analyze CUAD audit results and generate candidate task-selection comparison reports."""

import sys
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    json_path = Path("reports/cuad_sparsity_audit.json")
    if not json_path.exists():
        print(f"Error: Audit JSON not found at {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    tasks = audit_data["task_details"]

    # Candidate Set A: Broad High-Prevalence Risk-Balanced Set (RECOMMENDED)
    set_a = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]
    
    # Candidate Set B: Intellectual Property & Commercial License Focus
    set_b = ["License Grant", "Ip Ownership Assignment", "Exclusivity"]

    # Candidate Set C: Operational Compliance & Revenue Risk Focus
    set_c = ["Audit Rights", "Cap On Liability", "Revenue/Profit Sharing"]

    candidate_sets = {
        "Candidate Set A (Broad Risk-Balanced - RECOMMENDED)": set_a,
        "Candidate Set B (IP & Licensing Focus)": set_b,
        "Candidate Set C (Operational & Compliance Focus)": set_c
    }

    evidence_output = {}

    for set_name, task_list in candidate_sets.items():
        set_info = []
        total_spans = 0
        min_pos_contracts = 999
        for t in task_list:
            info = tasks[t]
            pos_c = info["positive_contracts"]
            spans = info["total_positive_spans"]
            prev = info["positive_contract_prevalence"] * 100
            conc = info["concentration_diagnostics"]
            mean_s = conc["mean_spans_per_pos_contract"]
            
            total_spans += spans
            min_pos_contracts = min(min_pos_contracts, pos_c)
            
            set_info.append({
                "task_name": t,
                "positive_contracts": pos_c,
                "prevalence_pct": prev,
                "total_spans": spans,
                "mean_spans_per_pos_contract": mean_s
            })
            
        evidence_output[set_name] = {
            "tasks": set_info,
            "bottleneck_positive_contracts": min_pos_contracts,
            "total_spans_across_tasks": total_spans,
            "pre_split_feasibility_pass": min_pos_contracts >= 41
        }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_json = reports_dir / "phase3_task_selection_evidence.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evidence_output, f, indent=2)

    print(f"Generated task selection evidence JSON at {out_json}")

if __name__ == "__main__":
    main()

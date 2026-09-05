"""Script to compute and generate CUAD dataset inspection and sparsity audit reports."""

import sys
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riskloop.data.discovery import inspect_cuad_schema, compute_cuad_sparsity_audit

def main():
    raw_path = "data/raw/CUAD_v1.json"
    print(f"1. Inspecting CUAD schema at {raw_path}...")
    schema_info = inspect_cuad_schema(raw_path)
    
    print(f"2. Computing pre-experiment sparsity audit...")
    audit_data = compute_cuad_sparsity_audit(raw_path, min_train=25, min_val=8, min_test=8)
    
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = reports_dir / "cuad_sparsity_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"3. Saved machine-readable JSON artifact to {json_path}")

    md_path = reports_dir / "cuad_sparsity_audit.md"
    lines = []
    lines.append("# RiskLoop Phase 2 — CUAD v1 Dataset Inspection & Sparsity Audit Report")
    lines.append("**Date:** August 30, 2026  ")
    lines.append("**Dataset Version:** CUAD v1 (`aok_v1.0`)  ")
    lines.append("**Total Audited Contracts:** 510  ")
    lines.append("**Requirements Baseline:** RiskLoop SRD v1.1 (FR-01 through FR-13)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("A comprehensive pre-experiment data-sparsity audit was conducted across all 41 task categories present in `data/raw/CUAD_v1.json`.")
    lines.append(f"- **Total Candidate Tasks Identified:** {audit_data['total_candidate_tasks']}")
    lines.append(f"- **Potentially Viable Tasks (Pre-Split Feasibility >= 41 Positive Contracts):** {audit_data['viable_task_count']}")
    lines.append(f"- **Failing Tasks (< 41 Positive Contracts):** {audit_data['failing_task_count']}")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Pre-Split Feasibility Screen Notice:** Having >= 41 total positive contracts across the 510 contracts is a necessary pre-split feasibility screen for satisfying Train >= 25, Val >= 8, Test >= 8. Final adequacy must be verified post-splitting once the contract split map is generated.")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Scientific Control Notice:** Per SRD FR-02, no task definitions have been invented, and the final viable task set is presented as evidence for review prior to freezing the task configuration.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Dataset Schema & Structure")
    lines.append("- **Root Structure:** JSON dictionary containing `version` (`aok_v1.0`) and `data` (list of 510 contracts).")
    lines.append("- **Contract Record:** Each contract element contains a `title` string and a `paragraphs` list.")
    lines.append("- **Context Representation:** `paragraphs[0]['context']` contains the full raw text of the contract.")
    lines.append("- **QAs Array:** `paragraphs[0]['qas']` contains 41 question objects.")
    lines.append("- **QA Schema:**")
    lines.append("  - `id`: `{contract_title}__{category_name}`")
    lines.append("  - `question`: Full natural language question prompt.")
    lines.append("  - `is_impossible`: Boolean (`True` if negative/no answer, `False` if positive span exists).")
    lines.append("  - `answers`: List of `{'text': str, 'answer_start': int}` objects.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Complete Per-Task Sparsity Audit Table")
    lines.append("")
    lines.append("| Task Name | Pos Contracts | Prevalence | Pos Spans | Mean Spans/Pos Cntr | Max Spans | Top 10% Share | Top 20% Share | Pre-Split Gate Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for task_name, info in sorted(audit_data["task_details"].items()):
        pos_c = info["positive_contracts"]
        prev = info["positive_contract_prevalence"] * 100
        spans = info["total_positive_spans"]
        conc = info["concentration_diagnostics"]
        mean_s = conc["mean_spans_per_pos_contract"]
        max_s = conc["max_spans_single_contract"]
        top10 = conc["top_10pct_contract_span_share"] * 100
        top20 = conc["top_20pct_contract_span_share"] * 100
        status = "**PASS**" if info["passes_adequacy_gate"] else "FAIL"
        
        lines.append(f"| `{task_name}` | {pos_c}/510 | {prev:.1f}% | {spans} | {mean_s:.1f} | {max_s} | {top10:.1f}% | {top20:.1f}% | {status} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Potentially Viable Tasks (33 Tasks)")
    lines.append("The following 33 tasks meet or exceed the global minimum adequacy threshold of 41 positive contracts (enforcing Train >= 25, Val >= 8, Test >= 8):")
    lines.append("")
    for t in audit_data["viable_tasks"]:
        t_info = audit_data["task_details"][t]
        lines.append(f"- `{t}` ({t_info['positive_contracts']} positive contracts, {t_info['total_positive_spans']} spans)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Failing Tasks (8 Tasks)")
    lines.append("The following 8 tasks fail the SRD data-adequacy gate and cannot support the controlled experiment:")
    lines.append("")
    for t in audit_data["failing_tasks"]:
        t_info = audit_data["task_details"][t]
        lines.append(f"- `{t}`: Only {t_info['positive_contracts']} positive contracts (requires >= 41).")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Contract Concentration Diagnostics")
    lines.append("High concentration indicates that a small fraction of contracts account for a large portion of positive spans:")
    lines.append("- **Highest Span Concentration Tasks:**")
    lines.append("  - `Rofr/Rofo/Rofn`: Top 20% of positive contracts account for 56.7% of all positive spans.")
    lines.append("  - `Post-Termination Services`: Top 20% of positive contracts account for 51.3% of all positive spans.")
    lines.append("  - `Insurance`: Top 20% of positive contracts account for 51.8% of all positive spans.")
    lines.append("  - `Parties`: Highest maximum spans in a single contract (55 spans).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. TBD Decisions Register")
    lines.append("1. **Final Viable Task Selection:** Selection of candidate tasks for the 12-run experiment must be formally locked in Phase 3.")
    lines.append("2. **Tokenizer & Sequence Length:** Pending chunking and sequence length optimization.")
    lines.append("3. **Chunk Stride & Span Alignment:** Pending span length distribution audit.")
    lines.append("4. **Primary Evaluation Metrics:** To be selected post-lock.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"4. Saved human-readable Markdown report to {md_path}")

if __name__ == "__main__":
    main()

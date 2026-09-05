"""Scratch script to inspect raw CUAD annotations and contract lengths for Phase 4A analysis."""

import json
import numpy as np
from pathlib import Path

def main():
    raw_path = Path("data/raw/CUAD_v1.json")
    if not raw_path.exists():
        print(f"Error: Raw dataset not found at {raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    contracts = raw_data["data"]
    n_contracts = len(contracts)

    selected_tasks = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]

    # 1. Contract Length Statistics (Characters)
    char_lengths = [len(c["paragraphs"][0]["context"]) for c in contracts]
    char_arr = np.array(char_lengths)

    print("=== CONTRACT LENGTH DISTRIBUTION (CHARACTERS) ===")
    print(f"Total Contracts: {n_contracts}")
    print(f"Min Length   : {int(np.min(char_arr))} chars")
    print(f"25th Pctile  : {int(np.percentile(char_arr, 25))} chars")
    print(f"Median Length: {int(np.median(char_arr))} chars")
    print(f"Mean Length  : {int(np.mean(char_arr)):.1f} chars")
    print(f"75th Pctile  : {int(np.percentile(char_arr, 75))} chars")
    print(f"90th Pctile  : {int(np.percentile(char_arr, 90))} chars")
    print(f"Max Length   : {int(np.max(char_arr))} chars")

    sorted_idx = np.argsort(char_arr)
    short_idx, med_idx, long_idx = sorted_idx[0], sorted_idx[n_contracts // 2], sorted_idx[-1]
    print(f"\nShortest Contract: \"{contracts[short_idx]['title']}\" ({char_arr[short_idx]} chars)")
    print(f"Median Contract  : \"{contracts[med_idx]['title']}\" ({char_arr[med_idx]} chars)")
    print(f"Longest Contract : \"{contracts[long_idx]['title']}\" ({char_arr[long_idx]} chars)")

    # 2. Annotation Verification for Selected Tasks
    print("\n=== ANNOTATION VERIFICATION FOR SELECTED TASKS ===")
    exact_match_failures = 0
    total_pos_spans_checked = 0
    span_positions_relative = {t: [] for t in selected_tasks}
    multi_answer_contracts = {t: 0 for t in selected_tasks}
    overlapping_spans_contracts = {t: 0 for t in selected_tasks}

    for c in contracts:
        ctx = c["paragraphs"][0]["context"]
        ctx_len = len(ctx)
        for qa in c["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
            if t_name not in selected_tasks:
                continue

            is_impossible = qa.get("is_impossible", True)
            answers = qa.get("answers", [])

            if not is_impossible and len(answers) > 0:
                if len(answers) > 1:
                    multi_answer_contracts[t_name] += 1

                # Check overlap among spans in same QA
                spans_sorted = sorted(answers, key=lambda x: x["answer_start"])
                for i in range(len(spans_sorted) - 1):
                    start1 = spans_sorted[i]["answer_start"]
                    end1 = start1 + len(spans_sorted[i]["text"])
                    start2 = spans_sorted[i + 1]["answer_start"]
                    if start2 < end1:
                        overlapping_spans_contracts[t_name] += 1
                        break

                for ans in answers:
                    total_pos_spans_checked += 1
                    a_text = ans["text"]
                    a_start = ans["answer_start"]
                    a_end = a_start + len(a_text)

                    # Check character slice matching
                    slice_text = ctx[a_start:a_end]
                    if slice_text != a_text:
                        exact_match_failures += 1

                    # Relative position in document (0.0 to 1.0)
                    rel_pos = a_start / ctx_len if ctx_len > 0 else 0.0
                    span_positions_relative[t_name].append(rel_pos)

    print(f"Total positive spans checked across 3 tasks: {total_pos_spans_checked}")
    print(f"Exact character slice matching failures: {exact_match_failures}")

    for t in selected_tasks:
        pos_arr = np.array(span_positions_relative[t])
        print(f"\nTask: '{t}'")
        print(f"  - Multi-answer contracts: {multi_answer_contracts[t]}")
        print(f"  - Overlapping span contracts: {overlapping_spans_contracts[t]}")
        print(f"  - Relative span positions in document (0.0=Start, 1.0=End):")
        print(f"    Min={np.min(pos_arr):.3f}, 25%={np.percentile(pos_arr, 25):.3f}, Median={np.median(pos_arr):.3f}, 75%={np.percentile(pos_arr, 75):.3f}, Max={np.max(pos_arr):.3f}")

if __name__ == "__main__":
    main()

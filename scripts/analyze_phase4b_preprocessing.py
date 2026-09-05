"""Script for Phase 4B Preprocessing Analysis.

Calculates:
1. Answer-span length statistics (character and estimated token lengths)
2. Chunk-level distribution under 512/256 sliding windowing
3. Multiple-answer distribution per chunk
4. Chunk boundary span crossing analysis
"""

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

    # 1. Answer-Span Token & Character Length Analysis
    all_spans = []
    task_spans = {t: [] for t in selected_tasks}

    for c in contracts:
        for qa in c["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
            if t_name not in selected_tasks:
                continue

            is_impossible = qa.get("is_impossible", True)
            answers = qa.get("answers", [])

            if not is_impossible and len(answers) > 0:
                for ans in answers:
                    text = ans["text"]
                    char_len = len(text)
                    words = text.split()
                    word_len = len(words)
                    # Subword estimate for WordPiece/BPE: ~1.35x words or max(words, char_len // 4)
                    est_tokens = max(word_len, int(np.ceil(char_len / 4.0)))

                    span_info = {
                        "task": t_name,
                        "text": text,
                        "char_len": char_len,
                        "word_len": word_len,
                        "est_tokens": est_tokens
                    }
                    all_spans.append(span_info)
                    task_spans[t_name].append(span_info)

    print("=== 1. ANSWER-SPAN LENGTH ANALYSIS ===")
    print(f"Total Positive Spans across 3 tasks: {len(all_spans)}")

    for t_name in selected_tasks:
        spans = task_spans[t_name]
        char_lens = [s["char_len"] for s in spans]
        word_lens = [s["word_len"] for s in spans]
        est_tokens = [s["est_tokens"] for s in spans]

        c_arr = np.array(char_lens)
        w_arr = np.array(word_lens)
        t_arr = np.array(est_tokens)

        print(f"\nTask: '{t_name}' ({len(spans)} spans)")
        print(f"  - Char Lengths : Min={np.min(c_arr)}, Med={int(np.median(c_arr))}, Mean={np.mean(c_arr):.1f}, 95%={int(np.percentile(c_arr, 95))}, 99%={int(np.percentile(c_arr, 99))}, Max={np.max(c_arr)}")
        print(f"  - Word Lengths : Min={np.min(w_arr)}, Med={int(np.median(w_arr))}, Mean={np.mean(w_arr):.1f}, 95%={int(np.percentile(w_arr, 95))}, 99%={int(np.percentile(w_arr, 99))}, Max={np.max(w_arr)}")
        print(f"  - Est Tokens   : Min={np.min(t_arr)}, Med={int(np.median(t_arr))}, Mean={np.mean(t_arr):.1f}, 95%={int(np.percentile(t_arr, 95))}, 99%={int(np.percentile(t_arr, 99))}, Max={np.max(t_arr)}")

    # Unusually long spans (> 200 words / > 1000 chars)
    long_spans = [s for s in all_spans if s["word_len"] > 200 or s["char_len"] > 1000]
    print(f"\nUnusually Long Spans (>200 words or >1000 chars): {len(long_spans)} out of {len(all_spans)}")
    for ls in long_spans[:5]:
        print(f"  - Task: {ls['task']} | CharLen: {ls['char_len']} | WordLen: {ls['word_len']} | Snippet: \"{ls['text'][:80]}...\"")

    # 2. Chunking Simulation (Character Windowing Approximation for 512 tokens / 256 stride)
    # 512 tokens ~ 2048 chars; 256 stride ~ 1024 chars
    WIN_CHARS = 2048
    STRIDE_CHARS = 1024

    print("\n=== 2. CHUNKING & CLASS IMBALANCE SIMULATION (Window=2048 chars ~ 512 tokens, Stride=1024 chars ~ 256 tokens) ===")

    total_chunks = 0
    task_chunk_stats = {t: {"pos_chunks": 0, "neg_chunks": 0} for t in selected_tasks}
    multi_span_chunk_counts = {t: [] for t in selected_tasks}
    cross_boundary_spans = {t: 0 for t in selected_tasks}
    captured_in_overlap_spans = {t: 0 for t in selected_tasks}
    uncaptured_spans = {t: 0 for t in selected_tasks}

    for c in contracts:
        ctx = c["paragraphs"][0]["context"]
        ctx_len = len(ctx)

        # Generate window offsets for contract
        chunk_windows = []
        start = 0
        while start < ctx_len:
            end = min(start + WIN_CHARS, ctx_len)
            chunk_windows.append((start, end))
            if end >= ctx_len:
                break
            start += STRIDE_CHARS

        total_chunks += len(chunk_windows)

        # For each task, check chunk positivity & boundary coverage
        for qa in c["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
            if t_name not in selected_tasks:
                continue

            is_impossible = qa.get("is_impossible", True)
            answers = qa.get("answers", [])

            pos_spans_in_contract = []
            if not is_impossible and len(answers) > 0:
                pos_spans_in_contract = answers

            # Check boundary crossings for contract spans
            for ans in pos_spans_in_contract:
                a_start = ans["answer_start"]
                a_end = a_start + len(ans["text"])
                
                # Check if fully contained in AT LEAST ONE chunk
                fully_contained_in_any = False
                contained_count = 0
                for (w_start, w_end) in chunk_windows:
                    if w_start <= a_start and a_end <= w_end:
                        fully_contained_in_any = True
                        contained_count += 1

                if fully_contained_in_any:
                    if contained_count > 1:
                        captured_in_overlap_spans[t_name] += 1
                else:
                    uncaptured_spans[t_name] += 1

            # Check per-chunk positive span counts
            for (w_start, w_end) in chunk_windows:
                spans_in_chunk = 0
                for ans in pos_spans_in_contract:
                    a_start = ans["answer_start"]
                    a_end = a_start + len(ans["text"])
                    if w_start <= a_start and a_end <= w_end:
                        spans_in_chunk += 1

                if spans_in_chunk > 0:
                    task_chunk_stats[t_name]["pos_chunks"] += 1
                    multi_span_chunk_counts[t_name].append(spans_in_chunk)
                else:
                    task_chunk_stats[t_name]["neg_chunks"] += 1

    print(f"Total Chunks Generated across 510 contracts: {total_chunks} (Avg {total_chunks / n_contracts:.1f} chunks/contract)")

    for t_name in selected_tasks:
        pos_c = task_chunk_stats[t_name]["pos_chunks"]
        neg_c = task_chunk_stats[t_name]["neg_chunks"]
        tot_c = pos_c + neg_c
        pos_pct = (pos_c / tot_c) * 100
        neg_pct = (neg_c / tot_c) * 100

        m_arr = np.array(multi_span_chunk_counts[t_name])
        max_multi = int(np.max(m_arr)) if len(m_arr) > 0 else 0
        multi_greater_1 = sum(1 for x in m_arr if x > 1)

        print(f"\nTask: '{t_name}'")
        print(f"  - Total Chunks: {tot_c} | Pos Chunks: {pos_c} ({pos_pct:.1f}%) | Neg Chunks: {neg_c} ({neg_pct:.1f}%)")
        print(f"  - Multi-span chunks (>=2 spans in same chunk): {multi_greater_1} chunks (Max spans in 1 chunk: {max_multi})")
        print(f"  - Span boundary analysis: Uncaptured spans={uncaptured_spans[t_name]}, Captured in multiple overlapping chunks={captured_in_overlap_spans[t_name]}")

if __name__ == "__main__":
    main()

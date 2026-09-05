"""Phase 4B.1 Targeted Verification Script using actual Legal-BERT FastTokenizer.

Calculates:
1. Exact Legal-BERT token counts for all 1,572 positive spans (Min, Med, Mean, 95th, 99th, Max)
2. Exact window containment under 512 max length / 256 stride using actual token offsets
3. Multi-Task Chunk Label Distribution (Per-task and Union)
4. Negative Sampling Sensitivity Analysis for K = 2, 3, 4, 5
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

def main():
    raw_path = Path("data/raw/CUAD_v1.json")
    if not raw_path.exists():
        print(f"Error: Raw dataset not found at {raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    contracts = raw_data["data"]
    n_contracts = len(contracts)

    split_path = Path("data/splits/contract_split_map.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split_map = json.load(f)

    selected_tasks = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]

    print("Loading actual FastTokenizer: nlpaueb/legal-bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", use_fast=True)

    # 1. ACTUAL LEGAL-BERT TOKEN COUNTS FOR POSITIVE SPANS
    print("\n=== 1. ACTUAL LEGAL-BERT TOKEN COUNTS FOR POSITIVE SPANS ===")
    task_token_counts = {t: [] for t in selected_tasks}
    all_spans = []

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
                    # Tokenize answer text with Legal-BERT tokenizer (without special tokens CLS/SEP)
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    n_toks = len(toks)
                    task_token_counts[t_name].append(n_toks)
                    all_spans.append({"task": t_name, "text": text, "tokens": n_toks, "answer_start": ans["answer_start"], "contract_title": c["title"]})

    for t_name in selected_tasks:
        arr = np.array(task_token_counts[t_name])
        print(f"\nTask: '{t_name}' ({len(arr)} spans)")
        print(f"  - Min Token Length   : {int(np.min(arr))}")
        print(f"  - Median Token Length: {int(np.median(arr))}")
        print(f"  - Mean Token Length  : {np.mean(arr):.1f}")
        print(f"  - 95th Percentile    : {int(np.percentile(arr, 95))}")
        print(f"  - 99th Percentile    : {int(np.percentile(arr, 99))}")
        print(f"  - Maximum Token Length: {int(np.max(arr))}")

    # 2. EXACT SLIDING WINDOW TOKENIZATION (512 max_length, 256 stride) ACROSS ALL CONTRACTS
    print("\n=== 2. SLIDING WINDOW TOKENIZATION & CONTAINMENT ANALYSIS ===")
    MAX_TOKENS = 512
    STRIDE_TOKENS = 256

    total_chunks_510 = 0
    contract_chunks = {}  # c_title -> list of chunk dicts
    span_containment = []

    for c in contracts:
        c_title = c["title"]
        ctx = c["paragraphs"][0]["context"]

        # Fast Tokenizer encode full contract context with return_offsets_mapping=True
        # Avoid truncation parameter so we get full contract token sequence and offsets
        encoding = tokenizer(ctx, return_offsets_mapping=True, add_special_tokens=False)
        all_offsets = encoding["offset_mapping"]
        total_tokens = len(all_offsets)

        # Sliding window over token sequence: window size 512, stride 256
        c_chunk_list = []
        token_start = 0
        chunk_idx = 0

        while token_start < total_tokens:
            token_end = min(token_start + MAX_TOKENS, total_tokens)
            chunk_offsets = all_offsets[token_start:token_end]
            
            char_start = chunk_offsets[0][0] if chunk_offsets else 0
            char_end = chunk_offsets[-1][1] if chunk_offsets else 0

            c_chunk_list.append({
                "contract_title": c_title,
                "partition": split_map.get(c_title),
                "chunk_idx": chunk_idx,
                "token_start": token_start,
                "token_end": token_end,
                "char_start": char_start,
                "char_end": char_end,
                "pos_tasks": set()
            })
            chunk_idx += 1
            if token_end >= total_tokens:
                break
            token_start += STRIDE_TOKENS

        contract_chunks[c_title] = c_chunk_list
        total_chunks_510 += len(c_chunk_list)

        # Check positive span containment for this contract
        for qa in c["paragraphs"][0]["qas"]:
            qa_id = qa["id"]
            t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
            if t_name not in selected_tasks:
                continue

            is_impossible = qa.get("is_impossible", True)
            answers = qa.get("answers", [])

            if not is_impossible and len(answers) > 0:
                for ans in answers:
                    a_start = ans["answer_start"]
                    a_end = a_start + len(ans["text"])

                    # Check if span is fully contained in at least one chunk window
                    contained_chunks = []
                    for ch in c_chunk_list:
                        if ch["char_start"] <= a_start and a_end <= ch["char_end"]:
                            contained_chunks.append(ch)
                            ch["pos_tasks"].add(t_name)

                    is_contained = len(contained_chunks) > 0
                    span_containment.append({
                        "task": t_name,
                        "text": ans["text"],
                        "tokens": len(tokenizer.encode(ans["text"], add_special_tokens=False)),
                        "contained": is_contained,
                        "num_contained_chunks": len(contained_chunks)
                    })

    n_total_spans = len(span_containment)
    n_contained = sum(1 for s in span_containment if s["contained"])
    n_not_contained = n_total_spans - n_contained
    pct_contained = (n_contained / n_total_spans) * 100

    print(f"Total Positive Spans Evaluated : {n_total_spans}")
    print(f"Fully Contained in >=1 Chunk    : {n_contained} ({pct_contained:.2f}%)")
    print(f"Not Fully Contained (Crosses)   : {n_not_contained} ({100 - pct_contained:.2f}%)")

    # 3. MULTI-TASK CHUNK LABEL DISTRIBUTION (ALL 510 CONTRACTS)
    print("\n=== 3. MULTI-TASK CHUNK LABEL DISTRIBUTION (510 CONTRACTS) ===")
    all_chunks_flat = [ch for c_list in contract_chunks.values() for ch in c_list]

    for t_name in selected_tasks:
        pos_cnt = sum(1 for ch in all_chunks_flat if t_name in ch["pos_tasks"])
        neg_cnt = total_chunks_510 - pos_cnt
        print(f"Task: '{t_name:<28}' | Total Chunks: {total_chunks_510:5d} | Pos: {pos_cnt:4d} ({pos_cnt/total_chunks_510*100:4.1f}%) | Neg: {neg_cnt:5d} ({neg_cnt/total_chunks_510*100:4.1f}%)")

    union_pos_chunks = [ch for ch in all_chunks_flat if len(ch["pos_tasks"]) > 0]
    union_pos_cnt = len(union_pos_chunks)
    union_neg_cnt = total_chunks_510 - union_pos_cnt
    print(f"\nUNION DISTRIBUTION across 3 Tasks (All 510 Contracts):")
    print(f"  - Total Chunks                         : {total_chunks_510}")
    print(f"  - Chunks Positive for >=1 Task (Union Pos): {union_pos_cnt} ({union_pos_cnt/total_chunks_510*100:.1f}%)")
    print(f"  - Chunks Negative for ALL 3 Tasks      : {union_neg_cnt} ({union_neg_cnt/total_chunks_510*100:.1f}%)")

    # 4. NEGATIVE SAMPLING SENSITIVITY ANALYSIS (TRAIN SPLIT ONLY: 357 CONTRACTS)
    print("\n=== 4. NEGATIVE SAMPLING SENSITIVITY ANALYSIS (TRAIN SPLIT: 357 CONTRACTS) ===")
    train_contract_titles = sorted([ct for ct, p in split_map.items() if p == "train"])
    train_chunks_flat = [ch for ch in all_chunks_flat if ch["partition"] == "train"]
    total_train_chunks_opt_a = len(train_chunks_flat)
    
    train_union_pos_flat = [ch for ch in train_chunks_flat if len(ch["pos_tasks"]) > 0]
    n_train_union_pos = len(train_union_pos_flat)

    print(f"Option A (Retain All Chunks in Train Split):")
    print(f"  - Total Training Chunks: {total_train_chunks_opt_a}")
    print(f"  - Union-Positive Chunks: {n_train_union_pos} ({n_train_union_pos/total_train_chunks_opt_a*100:.1f}%)")
    print(f"  - Union-Negative Chunks: {total_train_chunks_opt_a - n_train_union_pos} ({(total_train_chunks_opt_a - n_train_union_pos)/total_train_chunks_opt_a*100:.1f}%)")
    for t_name in selected_tasks:
        t_pos = sum(1 for ch in train_chunks_flat if t_name in ch["pos_tasks"])
        t_neg = total_train_chunks_opt_a - t_pos
        print(f"    * {t_name:<28}: Pos={t_pos:4d} ({t_pos/total_train_chunks_opt_a*100:4.1f}%) | Neg={t_neg:5d} ({t_neg/total_train_chunks_opt_a*100:4.1f}%)")

    # Option B evaluation for K = 2, 3, 4, 5
    import random
    rng = random.Random(42)

    for K in [2, 3, 4, 5]:
        retained_chunks = []
        unused_neg_chunks = 0

        for ct in train_contract_titles:
            c_chunks = contract_chunks[ct]
            c_pos = [ch for ch in c_chunks if len(ch["pos_tasks"]) > 0]
            c_neg = [ch for ch in c_chunks if len(ch["pos_tasks"]) == 0]

            # Retain ALL positive chunks for contract
            retained_chunks.extend(c_pos)

            # Deterministically sample K negative chunks
            if len(c_neg) <= K:
                retained_chunks.extend(c_neg)
            else:
                sampled = rng.sample(c_neg, K)
                retained_chunks.extend(sampled)
                unused_neg_chunks += (len(c_neg) - K)

        n_retained = len(retained_chunks)
        n_retained_pos = sum(1 for ch in retained_chunks if len(ch["pos_tasks"]) > 0)
        n_retained_neg = n_retained - n_retained_pos
        reduction_pct = (1.0 - n_retained / total_train_chunks_opt_a) * 100

        print(f"\n  [Option B, K={K} Negatives/Contract]")
        print(f"    - Total Retained Training Chunks: {n_retained} (Reduction vs Option A: {reduction_pct:.1f}%)")
        print(f"    - Union-Positive Chunks         : {n_retained_pos} ({n_retained_pos/n_retained*100:.1f}%)")
        print(f"    - Union-Negative Chunks         : {n_retained_neg} ({n_retained_neg/n_retained*100:.1f}%)")
        print(f"    - Unused Candidate Neg Chunks   : {unused_neg_chunks}")
        print(f"    - Task Label Distributions in Retained Training Set:")
        for t_name in selected_tasks:
            t_pos = sum(1 for ch in retained_chunks if t_name in ch["pos_tasks"])
            t_neg = n_retained - t_pos
            print(f"      * {t_name:<28}: Pos={t_pos:4d} ({t_pos/n_retained*100:4.1f}%) | Neg={t_neg:5d} ({t_neg/n_retained*100:4.1f}%)")

if __name__ == "__main__":
    main()

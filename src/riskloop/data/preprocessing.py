"""RiskLoop Preprocessing Pipeline Module.

SRD Traceability: FR-05..FR-07, FR-14..FR-16, NFR-06..NFR-08, NFR-12
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer

SELECTED_TASKS = [
    "Cap On Liability",
    "Anti-Assignment",
    "Termination For Convenience"
]

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 256
MAX_WINDOW_TOKENS = 510  # 512 minus CLS and SEP tokens
NEGATIVE_K_TRAIN = 4
SAMPLING_SEED = 42

def sanitize_contract_id(title: str) -> str:
    """Sanitize title for clean chunk identifiers."""
    return title.strip().replace(" ", "_").replace("/", "_")


def align_char_span_to_tokens(
    gold_start_char: int,
    gold_end_char: int,
    chunk_offsets: List[List[int]]
) -> Optional[Tuple[int, int]]:
    """Align a character span to token indices inside a chunk window.
    
    chunk_offsets is a list of [start_char, end_char] for token indices 0..511.
    Index 0 is [CLS] (offset [0, 0]), index 511 (or last text token + 1) is [SEP].
    
    Returns (start_token_idx, end_token_idx) if gold span is fully contained in window text tokens,
    else returns None.
    """
    if len(chunk_offsets) < 3:
        return None
        
    # Text tokens are at indices 1 .. len(chunk_offsets) - 2 (or padding)
    window_char_start = chunk_offsets[1][0]
    
    # Find last non-special, non-padding text token index
    last_text_idx = 1
    for idx in range(1, len(chunk_offsets)):
        if chunk_offsets[idx] == [0, 0]:
            break
        last_text_idx = idx
        
    window_char_end = chunk_offsets[last_text_idx][1]
    
    # Complete containment check
    if gold_start_char < window_char_start or gold_end_char > window_char_end:
        return None
        
    start_token = None
    end_token = None
    
    for idx in range(1, last_text_idx + 1):
        t_start, t_end = chunk_offsets[idx]
        if t_start <= gold_start_char < t_end or (start_token is None and t_start >= gold_start_char):
            if start_token is None:
                start_token = idx
                
        if t_start < gold_end_char <= t_end or (start_token is not None and t_end >= gold_end_char):
            end_token = idx
            break
            
    if start_token is not None and end_token is not None and start_token <= end_token:
        return (start_token, end_token)
        
    return None


def preprocess_contract(
    contract: Dict[str, Any],
    partition: str,
    tokenizer: Any,
    selected_tasks: List[str] = SELECTED_TASKS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    doc_stride: int = DOC_STRIDE
) -> List[Dict[str, Any]]:
    """Process a single contract into 512-token sliding-window chunk records."""
    title = contract.get("title", "")
    contract_id = sanitize_contract_id(title)
    paragraphs = contract.get("paragraphs", [])
    if not paragraphs:
        return []
        
    context = paragraphs[0].get("context", "")
    qas = paragraphs[0].get("qas", [])
    
    # Extract gold spans per task for this contract
    task_gold_spans: Dict[str, List[Tuple[int, int]]] = {t: [] for t in selected_tasks}
    for qa in qas:
        qa_id = qa.get("id", "")
        t_name = qa_id.rsplit("__", 1)[1] if "__" in qa_id else qa_id
        if t_name not in selected_tasks:
            continue
            
        is_impossible = qa.get("is_impossible", True)
        answers = qa.get("answers", [])
        if not is_impossible and len(answers) > 0:
            for ans in answers:
                s_char = ans["answer_start"]
                e_char = s_char + len(ans["text"])
                task_gold_spans[t_name].append((s_char, e_char))
                
    # Tokenize context text without special tokens to get clean offset mapping
    encoding = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
    all_input_ids = encoding["input_ids"]
    all_offsets = encoding["offset_mapping"]
    total_text_tokens = len(all_input_ids)
    
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    max_window_text = max_seq_length - 2  # 510 text tokens
    
    chunks: List[Dict[str, Any]] = []
    token_start = 0
    chunk_idx = 0
    
    while token_start < total_text_tokens or (total_text_tokens == 0 and chunk_idx == 0):
        token_end = min(token_start + max_window_text, total_text_tokens)
        
        text_ids = all_input_ids[token_start:token_end]
        text_offsets = all_offsets[token_start:token_end]
        
        chunk_input_ids = [cls_token_id] + text_ids + [sep_token_id]
        chunk_offsets = [[0, 0]] + [list(o) for o in text_offsets] + [[0, 0]]
        
        # Pad to max_seq_length if shorter
        padding_len = max_seq_length - len(chunk_input_ids)
        if padding_len > 0:
            chunk_input_ids.extend([pad_token_id] * padding_len)
            chunk_offsets.extend([[0, 0]] * padding_len)
            
        char_start = text_offsets[0][0] if text_offsets else 0
        char_end = text_offsets[-1][1] if text_offsets else 0
        
        chunk_id = f"{contract_id}__chunk_{chunk_idx:04d}"
        
        task_targets: Dict[str, Dict[str, Any]] = {}
        is_union_positive = False
        
        for t_name in selected_tasks:
            gold_aligned_spans: List[Tuple[int, int]] = []
            for (g_start, g_end) in task_gold_spans[t_name]:
                span_token_pair = align_char_span_to_tokens(g_start, g_end, chunk_offsets)
                if span_token_pair is not None:
                    gold_aligned_spans.append(span_token_pair)
                    
            if len(gold_aligned_spans) > 0:
                is_union_positive = True
                task_targets[t_name] = {
                    "has_positive": True,
                    "spans": gold_aligned_spans
                }
            else:
                task_targets[t_name] = {
                    "has_positive": False,
                    "spans": [(0, 0)]  # Token 0 is CLS / no-answer target
                }
                
        chunk_record = {
            "chunk_id": chunk_id,
            "contract_id": contract_id,
            "contract_title": title,
            "partition": partition,
            "chunk_idx": chunk_idx,
            "char_start": char_start,
            "char_end": char_end,
            "token_start": token_start,
            "token_end": token_end,
            "input_ids": chunk_input_ids,
            "offset_mapping": chunk_offsets,
            "is_union_positive": is_union_positive,
            "task_targets": task_targets
        }
        
        chunks.append(chunk_record)
        chunk_idx += 1
        
        if token_end >= total_text_tokens:
            break
        token_start += doc_stride
        
    return chunks


def build_preprocessing_pipeline(
    raw_json_path: str = "data/raw/CUAD_v1.json",
    split_map_path: str = "data/splits/contract_split_map.json",
    output_dir: str = "data/processed",
    selected_tasks: List[str] = SELECTED_TASKS,
    model_name: str = MODEL_NAME,
    negative_k_train: int = NEGATIVE_K_TRAIN,
    sampling_seed: int = SAMPLING_SEED
) -> Dict[str, Any]:
    """Execute complete deterministic dataset preprocessing pipeline."""
    p_raw = Path(raw_json_path)
    p_split = Path(split_map_path)
    
    if not p_raw.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_json_path}")
    if not p_split.exists():
        raise FileNotFoundError(f"Contract split map not found at {split_map_path}")
        
    with open(p_raw, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(p_split, "r", encoding="utf-8") as f:
        split_map = json.load(f)
        
    contracts = raw_data.get("data", [])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    partition_chunks: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Process all contracts by partition
    for contract in contracts:
        title = contract.get("title", "")
        partition = split_map.get(title)
        if not partition or partition not in partition_chunks:
            raise ValueError(f"Contract '{title}' missing valid partition in split map")
            
        c_chunks = preprocess_contract(
            contract=contract,
            partition=partition,
            tokenizer=tokenizer,
            selected_tasks=selected_tasks
        )
        partition_chunks[partition].extend(c_chunks)
        
    # Apply Option B deterministic negative sampling ONLY to Training partition
    train_all_chunks = partition_chunks["train"]
    train_contract_ids = sorted(list(set(ch["contract_id"] for ch in train_all_chunks)))
    
    retained_train_chunks: List[Dict[str, Any]] = []
    rng = random.Random(sampling_seed)
    
    for c_id in train_contract_ids:
        c_chunks = [ch for ch in train_all_chunks if ch["contract_id"] == c_id]
        c_pos = [ch for ch in c_chunks if ch["is_union_positive"]]
        c_neg = [ch for ch in c_chunks if not ch["is_union_positive"]]
        
        # Sort candidate negatives deterministically by chunk_idx
        c_neg = sorted(c_neg, key=lambda x: x["chunk_idx"])
        
        # Retain ALL union-positive chunks for contract
        retained_train_chunks.extend(c_pos)
        
        # Deterministically sample K negative chunks
        if len(c_neg) <= negative_k_train:
            retained_train_chunks.extend(c_neg)
        else:
            sampled_neg = rng.sample(c_neg, negative_k_train)
            # Re-sort sampled chunks by chunk_idx to maintain order
            sampled_neg = sorted(sampled_neg, key=lambda x: x["chunk_idx"])
            retained_train_chunks.extend(sampled_neg)
            
    # Re-sort retained train chunks deterministically by chunk_id
    retained_train_chunks = sorted(retained_train_chunks, key=lambda x: x["chunk_id"])
    partition_chunks["train"] = retained_train_chunks
    
    # Sort val and test chunks deterministically
    partition_chunks["val"] = sorted(partition_chunks["val"], key=lambda x: x["chunk_id"])
    partition_chunks["test"] = sorted(partition_chunks["test"], key=lambda x: x["chunk_id"])
    
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for part in ["train", "val", "test"]:
        out_file = p_out / f"{part}_chunks.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(partition_chunks[part], f, indent=2)
            
        p_chunks = partition_chunks[part]
        tot = len(p_chunks)
        union_pos = sum(1 for ch in p_chunks if ch["is_union_positive"])
        
        t_stats = {}
        for t_name in selected_tasks:
            t_pos = sum(1 for ch in p_chunks if ch["task_targets"][t_name]["has_positive"])
            t_stats[t_name] = {
                "positive_chunks": t_pos,
                "negative_chunks": tot - t_pos,
                "positive_pct": float(t_pos / tot * 100) if tot > 0 else 0.0
            }
            
        stats[part] = {
            "output_file": str(out_file),
            "total_chunks": tot,
            "union_positive_chunks": union_pos,
            "union_negative_chunks": tot - union_pos,
            "union_positive_pct": float(union_pos / tot * 100) if tot > 0 else 0.0,
            "task_stats": t_stats
        }
        
    return {
        "status": "SUCCESS",
        "selected_tasks": selected_tasks,
        "tokenizer": model_name,
        "sampling_seed": sampling_seed,
        "negative_k_train": negative_k_train,
        "partition_statistics": stats
    }

if __name__ == "__main__":
    res = build_preprocessing_pipeline()
    print(json.dumps(res, indent=2))

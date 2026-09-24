"""RiskLoop Unified Inference Engine & Span Extractor.

SRD Traceability: FR-14..FR-16, FR-21, FR-22, NFR-06..NFR-08
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from transformers import AutoTokenizer

from riskloop.inference.loader import ModelLoader, REPRESENTATIVE_RUN_MAPPING

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 256
MAX_WINDOW_TEXT = 510  # 512 minus CLS and SEP tokens


def decode_inference_chunk_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_offsets: List[List[List[int]]],
    max_span_length: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode raw model_scores and best text span predictions for inference.

    Restricts candidate span tokens to valid document text tokens (non-zero character offsets).
    `start_logits`: (B, 512)
    `end_logits`: (B, 512)
    `attention_mask`: (B, 512)
    `chunk_offsets`: List of B offset lists containing [start_char, end_char] per token.

    Returns:
    - model_scores: (B,) float tensor = max_valid_text_span_score - no_answer_score
    - predicted_spans: (B, 2) long tensor = (best_start_token, best_end_token)
    """
    batch_size, seq_len = start_logits.shape
    device = start_logits.device

    masked_start = start_logits.clone()
    masked_end = end_logits.clone()

    padding_mask = (attention_mask == 0)
    masked_start[padding_mask] = -10000.0
    masked_end[padding_mask] = -10000.0

    no_answer_score = masked_start[:, 0] + masked_end[:, 0]  # (B,)

    span_matrix = masked_start.unsqueeze(2) + masked_end.unsqueeze(1)  # (B, 512, 512)

    valid_text_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for b in range(batch_size):
        offsets_b = chunk_offsets[b]
        for tok_idx in range(1, seq_len - 1):
            if attention_mask[b, tok_idx] == 1 and tok_idx < len(offsets_b):
                if offsets_b[tok_idx] != [0, 0]:
                    valid_text_mask[b, tok_idx] = True

    i_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    j_indices = torch.arange(seq_len, device=device).unsqueeze(0)

    span_len_mask = (
        (i_indices >= 1) &
        (i_indices <= seq_len - 2) &
        (j_indices >= i_indices) &
        (j_indices <= seq_len - 2) &
        ((j_indices - i_indices) <= max_span_length)
    )

    valid_start = valid_text_mask.unsqueeze(2)
    valid_end = valid_text_mask.unsqueeze(1)
    full_valid_mask = span_len_mask.unsqueeze(0) & valid_start & valid_end

    span_matrix = span_matrix.masked_fill(~full_valid_mask, -1e9)

    flat_matrix = span_matrix.view(batch_size, -1)
    best_span_scores, flat_indices = flat_matrix.max(dim=1)

    best_start = torch.div(flat_indices, seq_len, rounding_mode="floor")
    best_end = flat_indices % seq_len

    model_scores = torch.where(
        best_span_scores > -9000.0,
        best_span_scores - no_answer_score,
        torch.tensor(-10000.0, device=device)
    )

    best_spans = torch.stack([best_start, best_end], dim=1)
    return model_scores.cpu(), best_spans.cpu()


class RiskLoopPredictor:
    """Unified inference predictor for legal contract risk clause detection and span extraction."""

    def __init__(
        self,
        checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_name: str = MODEL_NAME,
        pretrained: bool = True
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.loader = ModelLoader(
            base_dir=base_dir,
            checkpoint_overrides=checkpoint_paths,
            model_name=model_name,
            device=device
        )
        self.device = self.loader.device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.loaded_models: Dict[str, Any] = {}

    def get_task_model(self, task_name: str):
        """Lazy-load and cache SingleTaskModel for a given task."""
        if task_name not in self.loaded_models:
            model = self.loader.load_model_for_task(task_name, pretrained=self.pretrained)
            self.loaded_models[task_name] = model
        return self.loaded_models[task_name]

    def tokenize_and_chunk(self, contract_text: str) -> List[Dict[str, Any]]:
        """Chunk raw contract text into 512-token / 256-stride sliding windows.
        
        Preserves exact token-to-character offset mapping relative to the raw contract text.
        """
        if not contract_text or not contract_text.strip():
            return []

        # Tokenize contract text without special tokens to get character offsets
        encoding = self.tokenizer(
            contract_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        all_input_ids = encoding["input_ids"]
        all_offsets = encoding["offset_mapping"]
        total_text_tokens = len(all_input_ids)

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        chunks: List[Dict[str, Any]] = []
        token_start = 0
        chunk_idx = 0

        while token_start < total_text_tokens or (total_text_tokens == 0 and chunk_idx == 0):
            token_end = min(token_start + MAX_WINDOW_TEXT, total_text_tokens)

            text_ids = all_input_ids[token_start:token_end]
            text_offsets = all_offsets[token_start:token_end]

            chunk_input_ids = [cls_token_id] + text_ids + [sep_token_id]
            chunk_offsets = [[0, 0]] + [list(o) for o in text_offsets] + [[0, 0]]

            padding_len = MAX_SEQ_LENGTH - len(chunk_input_ids)
            if padding_len > 0:
                chunk_input_ids.extend([pad_token_id] * padding_len)
                chunk_offsets.extend([[0, 0]] * padding_len)

            attention_mask = [1] * (len(text_ids) + 2) + [0] * padding_len
            token_type_ids = [0] * MAX_SEQ_LENGTH

            char_start = text_offsets[0][0] if text_offsets else 0
            char_end = text_offsets[-1][1] if text_offsets else 0

            chunks.append({
                "chunk_idx": chunk_idx,
                "input_ids": chunk_input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "chunk_offsets": chunk_offsets,
                "char_start": char_start,
                "char_end": char_end
            })

            chunk_idx += 1
            token_start += DOC_STRIDE
            if token_end == total_text_tokens:
                break

        return chunks

    def predict_task_for_chunks(
        self,
        task_name: str,
        contract_text: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute model forward pass and span extraction for a single task across chunks."""
        info = REPRESENTATIVE_RUN_MAPPING[task_name]
        
        if not chunks:
            return {
                "task": task_name,
                "detected": False,
                "model_score": 0.0,
                "best_chunk_index": 0,
                "character_start": None,
                "character_end": None,
                "predicted_text": None,
                "model": info["run_name"],
                "seed": info["seed"]
            }

        model = self.get_task_model(task_name)

        input_ids = torch.tensor([c["input_ids"] for c in chunks], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([c["attention_mask"] for c in chunks], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([c["token_type_ids"] for c in chunks], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            chunk_offsets = [c["chunk_offsets"] for c in chunks]
            scores_tensor, spans_tensor = decode_inference_chunk_logits(
                start_logits, end_logits, attention_mask, chunk_offsets
            )
            scores = scores_tensor.cpu().numpy()
            spans = spans_tensor.cpu().numpy()

        best_chunk_idx = int(scores.argmax())
        best_score = float(scores[best_chunk_idx])
        best_start_token, best_end_token = int(spans[best_chunk_idx][0]), int(spans[best_chunk_idx][1])

        detected = (best_score > 0.0)

        char_start: Optional[int] = None
        char_end: Optional[int] = None
        predicted_text: Optional[str] = None

        if detected:
            target_chunk = chunks[best_chunk_idx]
            offsets = target_chunk["chunk_offsets"]

            if best_start_token < len(offsets) and best_end_token < len(offsets):
                s_off = offsets[best_start_token]
                e_off = offsets[best_end_token]

                if s_off != [0, 0] and e_off != [0, 0]:
                    char_start = s_off[0]
                    char_end = e_off[1]

                    if char_start is not None and char_end is not None and 0 <= char_start < char_end <= len(contract_text):
                        predicted_text = contract_text[char_start:char_end]
                    else:
                        detected = False
                else:
                    detected = False
            else:
                detected = False

        if not detected:
            char_start = None
            char_end = None
            predicted_text = None

        return {
            "task": task_name,
            "detected": detected,
            "model_score": round(best_score, 6),
            "best_chunk_index": best_chunk_idx,
            "character_start": char_start,
            "character_end": char_end,
            "predicted_text": predicted_text,
            "model": info["run_name"],
            "seed": info["seed"]
        }

    def predict(self, contract_text: str) -> Dict[str, Any]:
        """Perform full contract-level risk extraction across all target tasks."""
        if not isinstance(contract_text, str):
            raise ValueError(f"Input contract_text must be a string, got {type(contract_text)}")

        chunks = self.tokenize_and_chunk(contract_text)
        
        task_results = {}
        for task_name in REPRESENTATIVE_RUN_MAPPING.keys():
            task_results[task_name] = self.predict_task_for_chunks(task_name, contract_text, chunks)

        return {
            "contract_length_chars": len(contract_text),
            "num_chunks": len(chunks),
            "model_version": "RiskLoop Phase 5B Representative Protocol-Faithful Models",
            "tasks": task_results
        }

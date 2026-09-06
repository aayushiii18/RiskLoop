"""PyTorch Dataset wrapper for preprocessed CUAD chunk records.

SRD Traceability: FR-15..FR-16, NFR-06..NFR-08
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset

class CUADChunkDataset(Dataset):
    """Dataset for CUAD chunks supporting single-task and multi-task tensor generation."""
    
    def __init__(
        self,
        data: Union[str, Path, List[Dict[str, Any]]],
        task_name: Optional[str] = None,
        selected_tasks: Optional[List[str]] = None,
        multi_span_option: str = "first_span"
    ):
        if isinstance(data, (str, Path)):
            with open(data, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            self.chunks = data
            
        self.task_name = task_name
        self.selected_tasks = selected_tasks or [
            "Cap On Liability",
            "Anti-Assignment",
            "Termination For Convenience"
        ]
        self.multi_span_option = multi_span_option

    def __len__(self) -> int:
        return len(self.chunks)

    def _extract_target_span(self, task_target: Dict[str, Any]) -> Tuple[int, int]:
        """Extract (start_token, end_token) target index pair for a single task."""
        if not task_target.get("has_positive", False):
            return (0, 0)
            
        spans = task_target.get("spans", [])
        if not spans:
            return (0, 0)
            
        if self.multi_span_option == "first_span":
            return tuple(spans[0])
        elif self.multi_span_option == "longest_span":
            max_span = max(spans, key=lambda s: s[1] - s[0])
            return tuple(max_span)
        else:
            return tuple(spans[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        
        input_ids = torch.tensor(chunk["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(
            [1 if token_id != 0 else 0 for token_id in chunk["input_ids"]],
            dtype=torch.long
        )
        token_type_ids = torch.zeros(len(chunk["input_ids"]), dtype=torch.long)
        
        item = {
            "chunk_id": chunk["chunk_id"],
            "contract_id": chunk["contract_id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        task_targets = chunk.get("task_targets", {})
        
        if self.task_name:
            # Single-task mode
            t_target = task_targets.get(self.task_name, {"has_positive": False, "spans": [(0, 0)]})
            start_pos, end_pos = self._extract_target_span(t_target)
            item["start_positions"] = torch.tensor(start_pos, dtype=torch.long)
            item["end_positions"] = torch.tensor(end_pos, dtype=torch.long)
            item["has_positive"] = torch.tensor(t_target.get("has_positive", False), dtype=torch.bool)
        else:
            # Multi-task mode
            targets_dict = {}
            for t in self.selected_tasks:
                t_target = task_targets.get(t, {"has_positive": False, "spans": [(0, 0)]})
                start_pos, end_pos = self._extract_target_span(t_target)
                targets_dict[t] = {
                    "start_positions": torch.tensor(start_pos, dtype=torch.long),
                    "end_positions": torch.tensor(end_pos, dtype=torch.long),
                    "has_positive": torch.tensor(t_target.get("has_positive", False), dtype=torch.bool)
                }
            item["targets"] = targets_dict
            
        return item

"""Phase 5B Training & Validation Infrastructure Module.

SRD Traceability: FR-15..FR-16, FR-21..FR-28, NFR-06..NFR-08, NFR-11, NFR-13
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import get_linear_schedule_with_warmup

from riskloop.data.dataset import CUADChunkDataset
from riskloop.models.single_task import SingleTaskModel
from riskloop.models.multi_task import MultiTaskModel
from riskloop.evaluation.metrics import compute_evaluation_metrics
from riskloop.utils.reproducibility import set_seed
from riskloop.experiment.runner import ExperimentRunnerException


from types import MappingProxyType


PRIMARY_METRIC_DISCLOSURE = (
    "Primary metric (class_1_f1) measures binary clause-presence detection at chunk level; "
    "it does not by itself measure correctness of extracted span boundaries."
)

def _freeze_config(d: Dict[str, Any]) -> MappingProxyType:
    """Recursively convert nested dicts to MappingProxyType and lists/sets to tuples."""
    frozen = {}
    for k, v in d.items():
        if isinstance(v, dict):
            frozen[k] = _freeze_config(v)
        elif isinstance(v, (list, tuple)):
            frozen[k] = tuple(_freeze_config(item) if isinstance(item, dict) else item for item in v)
        elif isinstance(v, set):
            frozen[k] = tuple(v)
        else:
            frozen[k] = v
    return MappingProxyType(frozen)


LOCKED_TRAINING_CONFIG = _freeze_config({
    "model_backbone": "nlpaueb/legal-bert-base-uncased",
    "max_seq_length": 512,
    "doc_stride": 256,
    "selected_tasks": [
        "Cap On Liability",
        "Anti-Assignment",
        "Termination For Convenience"
    ],
    "seeds": [42, 43, 44],
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "num_epochs": 4,
    "effective_batch_size": 16,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "head_init_mean": 0.0,
    "head_init_std": 0.02,
    "multi_span_training_policy": "first_span",
    "primary_validation_metric": "class_1_f1",
    "best_epoch_tie_breaker": "earlier_epoch",
    "train_data_path": "data/processed/train_chunks.json",
    "val_data_path": "data/processed/val_chunks.json"
})


def seed_worker(worker_id: int) -> None:
    """Worker init function for DataLoader determinism."""
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def evaluate_chunk_span_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    max_span_length: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode raw model_scores and best span predictions from start and end logits.
    
    `start_logits`: (B, 512)
    `end_logits`: (B, 512)
    `attention_mask`: (B, 512)
    
    Returns (model_scores, predicted_spans):
    - model_scores: (B,) float tensor = max_span_score - no_answer_score
    - predicted_spans: (B, 2) long tensor = (best_start_token, best_end_token)
    """
    batch_size, seq_len = start_logits.shape
    
    # Logit masking for padding positions (attention_mask == 0)
    masked_start_logits = start_logits.clone()
    masked_end_logits = end_logits.clone()
    
    padding_mask = (attention_mask == 0)
    masked_start_logits[padding_mask] = -10000.0
    masked_end_logits[padding_mask] = -10000.0
    
    # No-answer score at token 0 ([CLS])
    no_answer_score = masked_start_logits[:, 0] + masked_end_logits[:, 0]
    
    model_scores = []
    best_spans = []
    
    for b in range(batch_size):
        s_logits = masked_start_logits[b]
        e_logits = masked_end_logits[b]
        
        best_score = -1e9
        best_start = 0
        best_end = 0
        
        # Valid text positions start at index 1 up to last text token
        for i in range(1, seq_len - 1):
            if s_logits[i] <= -9000.0:
                continue
            for j in range(i, min(i + max_span_length + 1, seq_len - 1)):
                if e_logits[j] <= -9000.0:
                    continue
                score = (s_logits[i] + e_logits[j]).item()
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j
                    
        m_score = (best_score - no_answer_score[b].item()) if best_score > -1e8 else -10000.0
        model_scores.append(m_score)
        best_spans.append((best_start, best_end))
        
    return torch.tensor(model_scores, dtype=torch.float32), torch.tensor(best_spans, dtype=torch.long)


class SingleRunTrainer:
    """Orchestrates training and validation for a single experimental run."""
    
    def __init__(
        self,
        run_info: Dict[str, Any],
        train_data_path: Union[str, Path] = "data/processed/train_chunks.json",
        val_data_path: Union[str, Path] = "data/processed/val_chunks.json",
        output_dir: Union[str, Path] = "reports/experiments",
        config_override: Optional[Dict[str, Any]] = None,
        use_pretrained: bool = True
    ):
        # Strict Test Set Access Check
        str_train = str(train_data_path).lower()
        str_val = str(val_data_path).lower()
        p_train = Path(train_data_path).name.lower()
        p_val = Path(val_data_path).name.lower()
        if p_train == "test_chunks.json" or p_val == "test_chunks.json" or "test_chunks.json" in str_train or "test_chunks.json" in str_val:
            raise ExperimentRunnerException(
                "TEST SET ACCESS PROHIBITED: Training runner is strictly forbidden from loading test set data (FR-38)."
            )
            
        self.run_info = run_info
        self.run_id = run_info["run_id"]
        self.condition = run_info["condition"]
        self.seed = run_info["seed"]
        self.task_scope = run_info["task_scope"]
        self.use_pretrained = use_pretrained
        
        self.config = dict(LOCKED_TRAINING_CONFIG)
        if config_override:
            self.config.update(config_override)
            
        self.output_dir = Path(output_dir) / f"run_{self.run_id:02d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)
        
        # Enforce Seed Lock
        set_seed(self.seed)

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Construct deterministic training and validation DataLoaders."""
        set_seed(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        if self.condition == "Condition_A":
            task_name = self.task_scope[0]
            train_ds = CUADChunkDataset(
                data=self.train_data_path,
                task_name=task_name,
                multi_span_option=self.config["multi_span_training_policy"]
            )
            val_ds = CUADChunkDataset(
                data=self.val_data_path,
                task_name=task_name,
                multi_span_option=self.config["multi_span_training_policy"]
            )
        else:
            train_ds = CUADChunkDataset(
                data=self.train_data_path,
                selected_tasks=self.config["selected_tasks"],
                multi_span_option=self.config["multi_span_training_policy"]
            )
            val_ds = CUADChunkDataset(
                data=self.val_data_path,
                selected_tasks=self.config["selected_tasks"],
                multi_span_option=self.config["multi_span_training_policy"]
            )
            
        batch_size = self.config["per_device_train_batch_size"]
        
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker
        )
        
        return train_loader, val_loader

    def _build_model(self) -> nn.Module:
        """Instantiate single-task or multi-task model under seed lock."""
        set_seed(self.seed)
        
        if self.condition == "Condition_A":
            model = SingleTaskModel(
                model_name=self.config["model_backbone"],
                dropout_prob=0.1,
                pretrained=self.use_pretrained
            )
        else:
            model = MultiTaskModel(
                model_name=self.config["model_backbone"],
                tasks=self.config["selected_tasks"],
                dropout_prob=0.1,
                pretrained=self.use_pretrained
            )
        return model

    def evaluate_epoch(self, model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
        """Evaluate validation set at epoch boundary and compute primary metric class_1_f1."""
        model.eval()
        
        task_data: Dict[str, Dict[str, List]] = {}
        target_tasks = self.task_scope if self.condition == "Condition_A" else self.config["selected_tasks"]
        
        for t in target_tasks:
            task_data[t] = {"y_true": [], "y_score": []}
            
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                
                if self.condition == "Condition_A":
                    task = self.task_scope[0]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    s_logits = outputs["start_logits"]
                    e_logits = outputs["end_logits"]
                    
                    scores, _ = evaluate_chunk_span_logits(s_logits, e_logits, attention_mask)
                    has_pos = batch["has_positive"].cpu().numpy().astype(int)
                    
                    task_data[task]["y_true"].extend(has_pos)
                    task_data[task]["y_score"].extend(scores.numpy())
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    logits_by_task = outputs["logits_by_task"]
                    
                    for task in target_tasks:
                        s_logits, e_logits = logits_by_task[task]
                        scores, _ = evaluate_chunk_span_logits(s_logits, e_logits, attention_mask)
                        has_pos = batch["targets"][task]["has_positive"].cpu().numpy().astype(int)
                        
                        task_data[task]["y_true"].extend(has_pos)
                        task_data[task]["y_score"].extend(scores.numpy())

        metrics_by_task = {}
        primary_scores = []
        
        for task in target_tasks:
            m = compute_evaluation_metrics(
                y_true=task_data[task]["y_true"],
                y_score=task_data[task]["y_score"],
                threshold=0.0
            )
            metrics_by_task[task] = m
            primary_scores.append(m["class_1_f1"])
            
        mean_primary_score = float(sum(primary_scores) / len(primary_scores))
        
        return {
            "mean_primary_score": mean_primary_score,
            "metrics_by_task": metrics_by_task
        }

    def train(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute training and validation loop for this run (or dry_run)."""
        train_loader, val_loader = self._build_dataloaders()
        model = self._build_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if dry_run:
            # Execute dry-run validation check without training steps
            val_eval = self.evaluate_epoch(model, val_loader, device)
            return {
                "run_id": self.run_id,
                "condition": self.condition,
                "seed": self.seed,
                "task_scope": self.task_scope,
                "dry_run": True,
                "status": "SUCCESS",
                "val_eval": val_eval
            }
            
        # Full training loop
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        epochs = self.config["num_epochs"]
        grad_accum = self.config["gradient_accumulation_steps"]
        total_steps = (len(train_loader) // grad_accum) * epochs
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        best_val_score = -1.0
        best_epoch = -1
        best_metrics = None
        best_state_dict = None
        epoch_history = []
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if self.condition == "Condition_A":
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            start_positions=batch["start_positions"].to(device),
                            end_positions=batch["end_positions"].to(device)
                        )
                        loss = outputs["loss"]
                    else:
                        targets = {
                            t: {
                                "start_positions": batch["targets"][t]["start_positions"].to(device),
                                "end_positions": batch["targets"][t]["end_positions"].to(device)
                            }
                            for t in self.config["selected_tasks"]
                        }
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            targets=targets
                        )
                        loss = outputs["loss"]
                        
                    loss = loss / grad_accum
                    
                scaler.scale(loss).backward()
                
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
            val_eval = self.evaluate_epoch(model, val_loader, device)
            score = val_eval["mean_primary_score"]
            
            epoch_history.append({
                "epoch": epoch,
                "val_score": score,
                "metrics": val_eval["metrics_by_task"]
            })
            
            # Best epoch selection with strict earlier epoch tie-breaking
            if round(score, 6) > round(best_val_score, 6):
                best_val_score = score
                best_epoch = epoch
                best_metrics = val_eval
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        # Save best model checkpoint & result JSON
        checkpoint_path = self.output_dir / "best_model.pt"
        result_path = self.output_dir / "run_result.json"
        
        torch.save(best_state_dict, checkpoint_path)
        
        run_artifact = {
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "task_scope": self.task_scope,
            "config": self.config,
            "primary_metric_disclosure": PRIMARY_METRIC_DISCLOSURE,
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "best_val_metrics": best_metrics,
            "epoch_history": epoch_history,
            "checkpoint_path": str(checkpoint_path)
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(run_artifact, f, indent=2)
            
        return run_artifact

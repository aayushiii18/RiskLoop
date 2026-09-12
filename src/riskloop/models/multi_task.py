"""Joint Multi-Task Span Extraction Model Architecture (Condition B).

SRD Traceability: FR-15, FR-21, FR-23, FR-24
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertConfig

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEFAULT_TASKS = [
    "Cap On Liability",
    "Anti-Assignment",
    "Termination For Convenience"
]

class MultiTaskModel(nn.Module):
    """Condition B Shared Multi-Task Span Extraction Model."""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        tasks: List[str] = DEFAULT_TASKS,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        pretrained: bool = True,
        config: Optional[AutoConfig] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.tasks = tasks
        self.hidden_size = hidden_size
        
        if config is not None:
            self.config = config
            self.encoder = AutoModel.from_config(config)
            self.hidden_size = getattr(config, "hidden_size", hidden_size)
        elif not pretrained:
            self.config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=256,
                max_position_embeddings=512
            )
            self.encoder = AutoModel.from_config(self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            self.hidden_size = getattr(self.config, "hidden_size", hidden_size)
            
        self.dropout = nn.Dropout(dropout_prob)
        self.heads = nn.ModuleDict({
            task: nn.Linear(self.hidden_size, 2) for task in tasks
        })
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize task linear QA heads with weight ~ Normal(mean=0.0, std=0.02) and bias = 0."""
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """Forward pass for multi-task span extraction across all active tasks.
        
        input_ids: (B, 512)
        attention_mask: (B, 512)
        token_type_ids: (B, 512)
        targets: Optional dict mapping task_name -> {"start_positions": (B,), "end_positions": (B,)}
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = encoder_outputs.last_hidden_state  # (B, 512, H)
        sequence_output = self.dropout(sequence_output)
        
        logits_by_task: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        task_losses: Dict[str, torch.Tensor] = {}
        total_loss: Optional[torch.Tensor] = None
        
        loss_fct = nn.CrossEntropyLoss()
        
        for task in self.tasks:
            head = self.heads[task]
            logits = head(sequence_output)  # (B, 512, 2)
            
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)  # (B, 512)
            end_logits = end_logits.squeeze(-1)      # (B, 512)
            
            logits_by_task[task] = (start_logits, end_logits)
            
            if targets and task in targets:
                t_targets = targets[task]
                start_positions = t_targets.get("start_positions")
                end_positions = t_targets.get("end_positions")
                
                if start_positions is not None and end_positions is not None:
                    s_loss = loss_fct(start_logits, start_positions)
                    e_loss = loss_fct(end_logits, end_positions)
                    t_loss = (s_loss + e_loss) / 2.0
                    task_losses[task] = t_loss
                    
                    if total_loss is None:
                        total_loss = t_loss
                    else:
                        total_loss = total_loss + t_loss
                        
        return {
            "logits_by_task": logits_by_task,
            "loss": total_loss,
            "task_losses": task_losses if targets else None
        }

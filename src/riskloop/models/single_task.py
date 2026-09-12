"""Single-Task Span Extraction Model Architecture (Condition A).

SRD Traceability: FR-15, FR-21, FR-22
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertConfig

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

class SingleTaskModel(nn.Module):
    """Condition A Independent Single-Task Span Extraction Model."""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        pretrained: bool = True,
        config: Optional[AutoConfig] = None
    ):
        super().__init__()
        self.model_name = model_name
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
        self.qa_outputs = nn.Linear(self.hidden_size, 2)
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize linear QA head with weight ~ Normal(mean=0.0, std=0.02) and bias = 0."""
        nn.init.normal_(self.qa_outputs.weight, mean=0.0, std=0.02)
        if self.qa_outputs.bias is not None:
            nn.init.zeros_(self.qa_outputs.bias)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Forward pass for single-task span extraction.
        
        input_ids: (B, 512)
        attention_mask: (B, 512)
        token_type_ids: (B, 512)
        
        Returns dict containing:
        - start_logits: (B, 512)
        - end_logits: (B, 512)
        - loss: scalar tensor if start_positions and end_positions provided, else None
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = encoder_outputs.last_hidden_state  # (B, 512, H)
        sequence_output = self.dropout(sequence_output)
        
        logits = self.qa_outputs(sequence_output)  # (B, 512, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)  # (B, 512)
        end_logits = end_logits.squeeze(-1)      # (B, 512)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0
            
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "loss": total_loss
        }

"""Model Loader Module for RiskLoop Inference.

SRD Traceability: FR-21, FR-22, FR-26, FR-37
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from riskloop.models.single_task import SingleTaskModel

REPRESENTATIVE_RUN_MAPPING: Dict[str, Dict[str, Any]] = {
    "Cap On Liability": {
        "run_id": 3,
        "run_name": "run_03",
        "condition": "Condition_A",
        "seed": 44,
        "default_rel_path": "reports/experiments/run_03/best_model.pt",
        "alt_rel_path": "reports/reproductions/phase5b_reproduction_v2/run_03/best_model.pt"
    },
    "Anti-Assignment": {
        "run_id": 4,
        "run_name": "run_04",
        "condition": "Condition_A",
        "seed": 42,
        "default_rel_path": "reports/experiments/run_04/best_model.pt",
        "alt_rel_path": "reports/reproductions/phase5b_reproduction_v2/run_04/best_model.pt"
    },
    "Termination For Convenience": {
        "run_id": 7,
        "run_name": "run_07",
        "condition": "Condition_A",
        "seed": 42,
        "default_rel_path": "reports/experiments/run_07/best_model.pt",
        "alt_rel_path": "reports/reproductions/phase5b_reproduction_v2/run_07/best_model.pt"
    }
}


class ModelLoader:
    """Loader abstraction for resolving and loading RiskLoop model checkpoints."""

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        checkpoint_overrides: Optional[Dict[str, Union[str, Path]]] = None,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        device: Optional[Union[str, torch.device]] = None
    ):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()
        self.checkpoint_overrides = checkpoint_overrides or {}
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def resolve_checkpoint_path(self, task_name: str) -> Optional[Path]:
        """Resolve physical checkpoint path on disk for a given task."""
        if task_name in self.checkpoint_overrides:
            override_p = Path(self.checkpoint_overrides[task_name])
            if override_p.exists():
                return override_p.resolve()

        if task_name not in REPRESENTATIVE_RUN_MAPPING:
            return None

        info = REPRESENTATIVE_RUN_MAPPING[task_name]
        
        # Check primary expected directory
        primary_path = self.base_dir / info["default_rel_path"]
        if primary_path.exists():
            return primary_path.resolve()

        # Check secondary reproduction directory
        alt_path = self.base_dir / info["alt_rel_path"]
        if alt_path.exists():
            return alt_path.resolve()

        return None

    def load_model_for_task(
        self,
        task_name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        pretrained: bool = True
    ) -> SingleTaskModel:
        """Instantiate SingleTaskModel architecture and load trained checkpoint weights."""
        if task_name not in REPRESENTATIVE_RUN_MAPPING:
            raise ValueError(f"Unsupported task '{task_name}'. Supported tasks: {list(REPRESENTATIVE_RUN_MAPPING.keys())}")

        ckpt_p = Path(checkpoint_path).resolve() if checkpoint_path else self.resolve_checkpoint_path(task_name)

        model = SingleTaskModel(
            model_name=self.model_name,
            dropout_prob=0.1,
            pretrained=pretrained
        )

        if ckpt_p and ckpt_p.exists():
            state_dict = torch.load(ckpt_p, map_location=self.device)
            if isinstance(state_dict, dict) and "model_state" in state_dict:
                state_dict = state_dict["model_state"]
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

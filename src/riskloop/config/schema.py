"""Configuration management, hashing, immutability, and drift verification.

SRD Traceability: FR-17, FR-18, FR-19, FR-20, NFR-08, NFR-13
"""

import json
import hashlib
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of configuration dictionary.
    
    Excludes existing hash or metadata fields before hashing.
    """
    clean_dict = {k: v for k, v in config_dict.items() if k not in ("config_hash", "locked_status")}
    serialized = json.dumps(clean_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class LockedConfig(BaseModel):
    """Pydantic model representing locked pre-experiment configuration."""
    version: str = "1.0.0"
    description: str = "Locked Pre-Experiment Configuration"
    locked_status: bool = False
    config_hash: Optional[str] = None
    
    tasks: Dict[str, Any] = Field(default_factory=dict)
    splitting: Dict[str, Any] = Field(default_factory=dict)
    data_audit: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)

    def lock_config(self) -> None:
        """Freeze configuration and assign deterministic SHA-256 hash."""
        d = self.model_dump() if hasattr(self, 'model_dump') else self.dict()
        self.config_hash = compute_config_hash(d)
        self.locked_status = True

    def verify_hash(self) -> bool:
        """Verify current configuration against stored hash."""
        if not self.locked_status or not self.config_hash:
            return False
        d = self.model_dump() if hasattr(self, 'model_dump') else self.dict()
        current_hash = compute_config_hash(d)
        return current_hash == self.config_hash


def verify_config_drift(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> List[str]:
    """Detect unauthorized configuration drift between Condition A and Condition B.
    
    Returns a list of error messages describing any parameter drift.
    SRD Traceability: NFR-13
    """
    drifts: List[str] = []
    
    # Shared keys that MUST be identical across Condition A and Condition B
    shared_sections = ["splitting", "data_audit", "preprocessing", "training", "evaluation"]
    
    for section in shared_sections:
        val_a = config_a.get(section, {})
        val_b = config_b.get(section, {})
        
        # Compare all parameters except condition-specific model structure
        if val_a != val_b:
            drifts.append(f"Section '{section}' differs between Condition A and Condition B:\n  A: {val_a}\n  B: {val_b}")

    # Check model backbone
    backbone_a = config_a.get("model", {}).get("backbone_name")
    backbone_b = config_b.get("model", {}).get("backbone_name")
    if backbone_a != backbone_b:
        drifts.append(f"Model backbone mismatch: Condition A uses '{backbone_a}' while Condition B uses '{backbone_b}'")

    return drifts

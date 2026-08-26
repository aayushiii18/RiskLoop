"""Contract-level data splitter module.

SRD Traceability: FR-05, FR-06, FR-07
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path

def generate_contract_split_map(
    contract_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """Deterministically partition contracts into Train, Val, and Test sets.
    
    Ensures every contract is assigned to exactly ONE partition.
    Returns a dictionary mapping contract_id -> partition_name ("train", "val", "test").
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    sorted_contracts = sorted(list(set(contract_ids)))
    rng = random.Random(seed)
    shuffled_contracts = sorted_contracts.copy()
    rng.shuffle(shuffled_contracts)
    
    n_total = len(shuffled_contracts)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_set = set(shuffled_contracts[:n_train])
    val_set = set(shuffled_contracts[n_train:n_train + n_val])
    test_set = set(shuffled_contracts[n_train + n_val:])
    
    split_map: Dict[str, str] = {}
    for c in sorted_contracts:
        if c in train_set:
            split_map[c] = "train"
        elif c in val_set:
            split_map[c] = "val"
        elif c in test_set:
            split_map[c] = "test"
        else:
            raise ValueError(f"Contract '{c}' was not assigned to any split")
            
    return split_map


def save_contract_split_map(split_map: Dict[str, str], filepath: str) -> None:
    """Save split map to JSON file for reuse across all runs (FR-07)."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, sort_keys=True)


def load_contract_split_map(filepath: str) -> Dict[str, str]:
    """Load contract split map from JSON file (FR-07)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

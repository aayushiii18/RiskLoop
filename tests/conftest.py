"""Pytest fixtures for RiskLoop protocol verification."""

import pytest
from typing import List, Dict, Any

@pytest.fixture
def mock_contracts() -> List[str]:
    """Generates 100 mock contract IDs."""
    return [f"Contract_{i:03d}.pdf" for i in range(1, 101)]

@pytest.fixture
def mock_task_data() -> Dict[str, Dict[str, Any]]:
    """Generates mock task data for data-sparsity audit testing."""
    return {
        "Task_A": {
            "train_pos_contracts": 30,
            "val_pos_contracts": 10,
            "test_pos_contracts": 10,
            "usable_positive_spans": 50,
            "positive_examples": 60,
            "negative_examples": 500,
            "prevalence": 0.107,
            "spans_per_contract": [3, 2, 2, 1, 1] + [1] * 25,
            "data_quality_issue": False
        },
        "Task_B": {
            "train_pos_contracts": 28,
            "val_pos_contracts": 9,
            "test_pos_contracts": 9,
            "usable_positive_spans": 40,
            "positive_examples": 45,
            "negative_examples": 450,
            "prevalence": 0.091,
            "spans_per_contract": [2] * 20 + [1] * 8,
            "data_quality_issue": False
        },
        "Task_C": {
            "train_pos_contracts": 26,
            "val_pos_contracts": 8,
            "test_pos_contracts": 8,
            "usable_positive_spans": 35,
            "positive_examples": 35,
            "negative_examples": 350,
            "prevalence": 0.091,
            "spans_per_contract": [1] * 26,
            "data_quality_issue": False
        }
    }

@pytest.fixture
def mock_config_dicts() -> Dict[str, Dict[str, Any]]:
    """Generates matching Condition A and Condition B configuration dictionaries."""
    base_config = {
        "version": "1.0.0",
        "description": "Mock Config",
        "locked_status": True,
        "config_hash": None,
        "tasks": {"list": ["Task_A", "Task_B", "Task_C"]},
        "splitting": {"split_map_path": "data/splits/map.json", "seed": 42},
        "data_audit": {"min_train_positive_contracts": 25, "min_val_positive_contracts": 8, "min_test_positive_contracts": 8},
        "preprocessing": {"max_sequence_length": 512, "chunk_stride": 128},
        "model": {"backbone_name": "roberta-base"},
        "training": {"seeds": [42, 43, 44], "learning_rate": 2e-5, "batch_size": 16},
        "evaluation": {"stage_2a_margin": 0.02, "stage_2b_veto_margin": 0.005}
    }
    return {
        "config_a": base_config,
        "config_b": base_config.copy()
    }

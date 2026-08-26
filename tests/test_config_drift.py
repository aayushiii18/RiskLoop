"""Tests for Condition A vs Condition B configuration drift detection.

SRD Traceability: FR-24, NFR-13
"""

import pytest
from riskloop.config.schema import verify_config_drift

def test_identical_configs_no_drift(mock_config_dicts):
    """Verify that identical configs between Condition A and B produce zero drift errors."""
    config_a = mock_config_dicts["config_a"]
    config_b = mock_config_dicts["config_b"]
    
    drifts = verify_config_drift(config_a, config_b)
    assert len(drifts) == 0


def test_parameter_drift_detection(mock_config_dicts):
    """Verify that modifying hyperparameters in Condition B triggers drift detection."""
    config_a = mock_config_dicts["config_a"]
    config_b = mock_config_dicts["config_b"].copy()
    
    # Introduce unauthorized parameter drift in Condition B
    config_b["preprocessing"] = {"max_sequence_length": 512, "chunk_stride": 64} # Drift!
    
    drifts = verify_config_drift(config_a, config_b)
    assert len(drifts) > 0
    assert "preprocessing" in drifts[0]

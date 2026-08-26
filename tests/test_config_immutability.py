"""Tests for configuration immutability and SHA-256 hash locking.

SRD Traceability: FR-17, FR-18, FR-19, FR-20, NFR-08
"""

import pytest
from riskloop.config.schema import LockedConfig, compute_config_hash

def test_config_hash_locking(mock_config_dicts):
    """Verify that lock_config generates SHA-256 hash and verify_hash succeeds."""
    cfg = LockedConfig(**mock_config_dicts["config_a"])
    cfg.lock_config()
    
    assert cfg.locked_status is True
    assert cfg.config_hash is not None
    assert len(cfg.config_hash) == 64 # SHA-256 hex length
    assert cfg.verify_hash() is True


def test_config_mutation_invalidates_hash(mock_config_dicts):
    """Verify that mutating any field in a locked config invalidates the hash check."""
    cfg = LockedConfig(**mock_config_dicts["config_a"])
    cfg.lock_config()
    
    # Mutate a parameter mid-experiment (simulating unauthorized modification)
    cfg.training["learning_rate"] = 5e-5
    
    assert cfg.verify_hash() is False, "Mutating a parameter must invalidate the configuration hash check"

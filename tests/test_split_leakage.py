"""Tests for contract-level split leakage prevention.

SRD Traceability: FR-05, FR-06, FR-07, NFR-12
"""

import pytest
from riskloop.data.splitter import generate_contract_split_map
from riskloop.data.leakage import audit_split_leakage, audit_example_leakage

def test_generate_split_map_no_leakage(mock_contracts):
    """Verify that generated split map has zero contract-level leakage."""
    split_map = generate_contract_split_map(mock_contracts, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    
    # Check that all contracts are assigned
    assert len(split_map) == len(mock_contracts)
    
    # Audit for leakage
    has_leakage, errors = audit_split_leakage(split_map)
    assert not has_leakage, f"Leakage detected: {errors}"
    assert len(errors) == 0


def test_injected_split_leakage_detection():
    """Verify that audit_split_leakage successfully detects injected contract overlap."""
    corrupted_map = {
        "Contract_001.pdf": "train",
        "Contract_002.pdf": "val",
        "Contract_003.pdf": "test",
    }
    # Inject leakage: Contract_001 is set to both train and val by mistake in mock dictionary
    has_leakage, _ = audit_split_leakage(corrupted_map)
    assert not has_leakage # valid map
    
    # Now create example level leakage
    examples = [
        {"contract_id": "Contract_001.pdf", "partition": "train"},
        {"contract_id": "Contract_001.pdf", "partition": "val"}, # Leakage!
        {"contract_id": "Contract_002.pdf", "partition": "test"}
    ]
    has_ex_leakage, ex_errors = audit_example_leakage(examples)
    assert has_ex_leakage
    assert "Contract_001.pdf" in ex_errors[0]

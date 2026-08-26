"""Tests for contract-level split determinism.

SRD Traceability: FR-07, NFR-06, NFR-07
"""

import pytest
from riskloop.data.splitter import generate_contract_split_map

def test_split_map_reproducibility(mock_contracts):
    """Verify that split map generation is 100% deterministic given the same seed."""
    map_1 = generate_contract_split_map(mock_contracts, seed=42)
    map_2 = generate_contract_split_map(mock_contracts, seed=42)
    
    assert map_1 == map_2, "Split maps generated with the same seed must be identical"


def test_split_map_different_seeds(mock_contracts):
    """Verify that different seeds produce different partitions."""
    map_42 = generate_contract_split_map(mock_contracts, seed=42)
    map_99 = generate_contract_split_map(mock_contracts, seed=99)
    
    assert map_42 != map_99, "Split maps generated with different seeds should differ"

"""Tests for CUAD dataset discovery and pre-experiment data-sparsity audit.

SRD Traceability: FR-01, FR-02, FR-03, FR-04, FR-08, FR-09, FR-10, FR-11, FR-12, FR-13
"""

import pytest
from riskloop.data.discovery import inspect_cuad_schema, compute_cuad_sparsity_audit

RAW_CUAD_PATH = "data/raw/CUAD_v1.json"

def test_inspect_cuad_schema():
    """Verify raw CUAD JSON schema structure and properties."""
    schema = inspect_cuad_schema(RAW_CUAD_PATH)
    
    assert "version" in schema["root_keys"]
    assert "data" in schema["root_keys"]
    assert schema["dataset_version"] == "aok_v1.0"
    assert schema["total_contracts"] == 510
    assert schema["sample_qas_count"] == 41


def test_compute_cuad_sparsity_audit_structure():
    """Verify pre-experiment sparsity audit outputs 41 tasks, 33 passing, 8 failing."""
    audit = compute_cuad_sparsity_audit(RAW_CUAD_PATH, min_train=25, min_val=8, min_test=8)
    
    assert audit["dataset_version"] == "aok_v1.0"
    assert audit["total_contracts_audited"] == 510
    assert audit["total_candidate_tasks"] == 41
    assert audit["global_required_min_positive_contracts"] == 41
    
    assert audit["viable_task_count"] == 33
    assert audit["failing_task_count"] == 8
    
    # Verify expected failing tasks (insufficient positive contracts < 41)
    expected_failing = {
        "Affiliate License-Licensor",
        "Most Favored Nation",
        "No-Solicit Of Customers",
        "Non-Disparagement",
        "Price Restrictions",
        "Source Code Escrow",
        "Third Party Beneficiary",
        "Unlimited/All-You-Can-Eat-License"
    }
    assert set(audit["failing_tasks"]) == expected_failing


def test_per_task_metrics_and_concentration():
    """Verify detailed task metrics and concentration diagnostics."""
    audit = compute_cuad_sparsity_audit(RAW_CUAD_PATH)
    task_details = audit["task_details"]
    
    # Document Name (100% prevalence)
    doc_name = task_details["Document Name"]
    assert doc_name["positive_contracts"] == 510
    assert doc_name["positive_contract_prevalence"] == pytest.approx(1.0)
    assert doc_name["passes_adequacy_gate"] is True
    
    # Parties (99.8% prevalence, 2554 total spans)
    parties = task_details["Parties"]
    assert parties["positive_contracts"] == 509
    assert parties["total_positive_spans"] == 2554
    assert parties["concentration_diagnostics"]["max_spans_single_contract"] == 55
    
    # Source Code Escrow (Fails: 13 pos contracts < 41)
    escrow = task_details["Source Code Escrow"]
    assert escrow["positive_contracts"] == 13
    assert escrow["passes_adequacy_gate"] is False

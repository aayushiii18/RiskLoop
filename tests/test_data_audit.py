"""Tests for data-sparsity audit gates and adequacy thresholds.

SRD Traceability: FR-08, FR-09, FR-10, FR-11, FR-12, FR-13, NFR-10
"""

import pytest
from riskloop.data.audit import run_sparsity_audit, DataAuditFailureException

def test_sparsity_audit_all_pass(mock_task_data):
    """Verify that audit passes when all tasks meet minimum positive contract thresholds (25/8/8)."""
    report = run_sparsity_audit(mock_task_data, min_train=25, min_val=8, min_test=8)
    
    assert report["global_status"] == "PASS"
    assert len(report["viable_tasks"]) == 3
    assert len(report["failed_tasks"]) == 0


def test_sparsity_audit_partial_pass(mock_task_data):
    """Verify that audit filters out failing tasks and proceeds with passing tasks (FR-12)."""
    mock_task_data["Task_C"]["train_pos_contracts"] = 15 # Fails train threshold 25
    
    report = run_sparsity_audit(mock_task_data, min_train=25, min_val=8, min_test=8)
    
    assert report["global_status"] == "PARTIAL_PASS"
    assert report["viable_tasks"] == ["Task_A", "Task_B"]
    assert "Task_C" in report["failed_tasks"]


def test_sparsity_audit_global_hard_fail(mock_task_data):
    """Verify that audit raises DataAuditFailureException when all tasks fail thresholds (NFR-10)."""
    for t in mock_task_data:
        mock_task_data[t]["train_pos_contracts"] = 5 # All fail!
        
    with pytest.raises(DataAuditFailureException) as exc_info:
        run_sparsity_audit(mock_task_data, min_train=25, min_val=8, min_test=8)
        
    assert "GLOBAL AUDIT HARD-FAIL" in str(exc_info.value)

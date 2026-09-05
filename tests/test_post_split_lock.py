"""Test suite verifying locked contract split map and post-split task adequacy."""

import json
from pathlib import Path
import pytest
from riskloop.data.splitter import load_contract_split_map
from riskloop.data.leakage import audit_split_leakage

def test_locked_contract_split_map_exists_and_valid():
    split_path = Path("data/splits/contract_split_map.json")
    assert split_path.exists(), "contract_split_map.json must exist in data/splits/"
    
    split_map = load_contract_split_map(str(split_path))
    assert len(split_map) == 510, f"Expected 510 contracts in split map, found {len(split_map)}"
    
    has_leakage, errors = audit_split_leakage(split_map)
    assert not has_leakage, f"Split leakage detected: {errors}"
    assert len(errors) == 0

    train_cnt = sum(1 for p in split_map.values() if p == "train")
    val_cnt = sum(1 for p in split_map.values() if p == "val")
    test_cnt = sum(1 for p in split_map.values() if p == "test")

    assert train_cnt == 357
    assert val_cnt == 76
    assert test_cnt == 77

def test_post_split_task_sparsity_gates():
    audit_path = Path("reports/post_split_sparsity_audit.json")
    assert audit_path.exists(), "post_split_sparsity_audit.json must exist in reports/"

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "LOCKED"
    assert audit["all_tasks_passed_gate"] is True

    tasks = audit["task_audits"]
    expected_tasks = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]

    for task_name in expected_tasks:
        assert task_name in tasks, f"Task {task_name} missing from post-split audit"
        t_info = tasks[task_name]
        assert t_info["train_pos_contracts"] >= 25, f"{task_name} train pos contracts < 25"
        assert t_info["val_pos_contracts"] >= 8, f"{task_name} val pos contracts < 8"
        assert t_info["test_pos_contracts"] >= 8, f"{task_name} test pos contracts < 8"
        assert t_info["passes_post_split_gate"] is True

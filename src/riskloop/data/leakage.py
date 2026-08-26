"""Contract-level split leakage auditor.

SRD Traceability: FR-06, NFR-12
"""

from typing import Dict, List, Set, Tuple

def audit_split_leakage(split_map: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Audit a contract split mapping for contract-level partition overlap.
    
    Returns (has_leakage, list_of_error_messages).
    """
    train_contracts: Set[str] = set()
    val_contracts: Set[str] = set()
    test_contracts: Set[str] = set()
    
    for contract_id, partition in split_map.items():
        if partition == "train":
            train_contracts.add(contract_id)
        elif partition == "val":
            val_contracts.add(contract_id)
        elif partition == "test":
            test_contracts.add(contract_id)
        else:
            return True, [f"Invalid partition name '{partition}' for contract '{contract_id}'"]

    errors: List[str] = []
    
    train_val_overlap = train_contracts.intersection(val_contracts)
    if train_val_overlap:
        errors.append(f"Leakage detected between Train and Val: {sorted(list(train_val_overlap))}")
        
    train_test_overlap = train_contracts.intersection(test_contracts)
    if train_test_overlap:
        errors.append(f"Leakage detected between Train and Test: {sorted(list(train_test_overlap))}")
        
    val_test_overlap = val_contracts.intersection(test_contracts)
    if val_test_overlap:
        errors.append(f"Leakage detected between Val and Test: {sorted(list(val_test_overlap))}")
        
    has_leakage = len(errors) > 0
    return has_leakage, errors


def audit_example_leakage(examples: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """Audit derived examples/chunks/spans to ensure no contract appears in multiple splits.
    
    Each item in `examples` must have keys "contract_id" and "partition".
    """
    contract_to_partitions: Dict[str, Set[str]] = {}
    for ex in examples:
        c_id = ex["contract_id"]
        part = ex["partition"]
        if c_id not in contract_to_partitions:
            contract_to_partitions[c_id] = set()
        contract_to_partitions[c_id].add(part)
        
    errors: List[str] = []
    for c_id, partitions in contract_to_partitions.items():
        if len(partitions) > 1:
            errors.append(f"Contract '{c_id}' appears in multiple partitions: {sorted(list(partitions))}")
            
    return len(errors) > 0, errors

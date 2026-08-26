"""Script to execute data-sparsity audit against task metadata."""

import sys
import json
import argparse
from pathlib import Path
from riskloop.data.audit import run_sparsity_audit, DataAuditFailureException

def main():
    parser = argparse.ArgumentParser(description="Run data-sparsity audit against task metadata.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to task metadata JSON file")
    args = parser.parse_args()

    filepath = Path(args.metadata)
    if not filepath.exists():
        print(f"Error: Task metadata file not found at {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    try:
        report = run_sparsity_audit(task_data)
        print("Data Sparsity Audit Results:")
        print(json.dumps(report, indent=2))
        sys.exit(0)
    except DataAuditFailureException as e:
        print(f"HARD FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

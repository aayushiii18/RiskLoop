"""Script to check contract split map for partition leakage."""

import sys
import argparse
from pathlib import Path
from riskloop.data.splitter import load_contract_split_map
from riskloop.data.leakage import audit_split_leakage

def main():
    parser = argparse.ArgumentParser(description="Audit contract split map for partition leakage.")
    parser.add_argument("--split-map", type=str, default="data/splits/contract_split_map.json", help="Path to split map JSON")
    args = parser.parse_args()

    filepath = Path(args.split_map)
    if not filepath.exists():
        print(f"Error: Split map file not found at {filepath}")
        sys.exit(1)

    split_map = load_contract_split_map(str(filepath))
    has_leakage, errors = audit_split_leakage(split_map)

    if has_leakage:
        print("FAIL: Contract split leakage detected!")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"PASS: Zero contract-level leakage detected across {len(split_map)} contracts.")
        sys.exit(0)

if __name__ == "__main__":
    main()

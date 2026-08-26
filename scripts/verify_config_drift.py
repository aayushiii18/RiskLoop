"""Script to verify parameter drift between Condition A and Condition B configurations."""

import sys
import json
import argparse
from pathlib import Path
from riskloop.config.schema import verify_config_drift

def main():
    parser = argparse.ArgumentParser(description="Verify config drift between Condition A and B.")
    parser.add_argument("--config-a", type=str, required=True, help="Path to Condition A config JSON")
    parser.add_argument("--config-b", type=str, required=True, help="Path to Condition B config JSON")
    args = parser.parse_args()

    with open(args.config_a, "r", encoding="utf-8") as f:
        cfg_a = json.load(f)
    with open(args.config_b, "r", encoding="utf-8") as f:
        cfg_b = json.load(f)

    drifts = verify_config_drift(cfg_a, cfg_b)
    if drifts:
        print("FAIL: Configuration drift detected!")
        for d in drifts:
            print(f"  - {d}")
        sys.exit(1)
    else:
        print("PASS: Zero configuration drift between Condition A and Condition B.")
        sys.exit(0)

if __name__ == "__main__":
    main()

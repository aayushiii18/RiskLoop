"""CLI Contract Risk Extraction Tool for RiskLoop.

Usage:
  python scripts/predict_contract.py --file examples/sample_contract_nda.txt
  python scripts/predict_contract.py --text "Neither party may assign this agreement." --json
"""

import sys
import json
import argparse
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riskloop.inference.predictor import RiskLoopPredictor


def main():
    parser = argparse.ArgumentParser(description="RiskLoop Contract Risk Analysis CLI")
    parser.add_argument("--file", type=str, help="Path to input contract text file (.txt)")
    parser.add_argument("--text", type=str, help="Raw contract text string")
    parser.add_argument("--json", action="store_true", help="Output raw machine-readable JSON")
    parser.add_argument("--output", type=str, help="Save prediction JSON payload to file")
    parser.add_argument("--checkpoint_dir", type=str, help="Base directory containing run_03, run_04, run_07 checkpoints")

    args = parser.parse_args()

    if not args.file and not args.text:
        parser.error("Must specify either --file <path> or --text <string>")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: Contract file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
    else:
        contract_text = args.text

    predictor = RiskLoopPredictor(base_dir=args.checkpoint_dir)
    result = predictor.predict(contract_text)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n==================================================")
        print("RISKLOOP CONTRACT RISK EXTRACTION REPORT")
        print("==================================================")
        print(f"Contract Length: {result['contract_length_chars']} chars | Chunks Processed: {result['num_chunks']}")
        print("--------------------------------------------------")

        for task, data in result["tasks"].items():
            status = "DETECTED" if data["detected"] else "NOT DETECTED"
            print(f"\nTask: {task}")
            print(f"  Status:         [{status}]")
            print(f"  Model Score:    {data['model_score']:.6f}")
            print(f"  Representative: {data['model']} (Seed {data['seed']})")

            if data["detected"] and data["predicted_text"]:
                c_start = data["character_start"]
                c_end = data["character_end"]
                print(f"  Offsets:        Chars [{c_start} .. {c_end}]")
                print(f"  Clause Text:    \"{data['predicted_text']}\"")
            else:
                print("  Clause Text:    N/A")

        print("\n==================================================\n")


if __name__ == "__main__":
    main()

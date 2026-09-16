"""Script to execute deterministic 12-run decision analysis and save decision_analysis.json."""

import sys
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riskloop.evaluation.decision import save_decision_analysis_artifact

def main():
    print("Executing Phase 5C Frozen 12-Run Decision Analysis...")
    artifact = save_decision_analysis_artifact("reports/experiments/decision_analysis.json")
    
    print("\n==================================================")
    print("PHASE 5C DECISION ANALYSIS COMPLETE")
    print("==================================================")
    print(f"Protocol Reference: {artifact['protocol_reference']}")
    print(f"Test Set Isolation: {artifact['test_set_isolation_statement']}")
    print(f"Joint Architecture Vetoed: {artifact['joint_architecture_vetoed']}")
    print("Final Task Adoptions:")
    for task, cond in artifact['final_task_adoptions'].items():
        rep = artifact['representative_models'][task]
        print(f"  - {task}: {cond} (Representative: {rep['run_name']}, Seed {rep['seed']}, Val F1: {rep['validation_score']:.6f})")
    print("\nSaved decision artifact to: reports/experiments/decision_analysis.json")
    print("==================================================\n")

if __name__ == "__main__":
    main()

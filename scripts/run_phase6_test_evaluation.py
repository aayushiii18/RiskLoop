"""Official Phase 6 Final Test Evaluation Runner Script.

SRD Traceability: FR-38, FR-39, FR-40
"""

import sys
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from riskloop.evaluation.test_evaluator import (
    TestSetEvaluator,
    TestSetIsolationException,
    validate_phase6_preflight
)
from riskloop.experiment.trainer import (
    evaluate_chunk_span_logits,
    get_device,
    seed_worker
)
from riskloop.data.dataset import CUADChunkDataset
from riskloop.models.single_task import SingleTaskModel
from riskloop.utils.reproducibility import set_seed


def main():
    base_dir = Path(".")
    decision_artifact_path = base_dir / "reports" / "experiments" / "decision_analysis.json"
    test_chunks_path = base_dir / "data" / "processed" / "test_chunks.json"

    if not decision_artifact_path.exists():
        raise TestSetIsolationException(
            f"PRE-FLIGHT HARD-FAIL: Decision artifact missing at '{decision_artifact_path}' (FR-38)."
        )

    with open(decision_artifact_path, "r", encoding="utf-8") as f:
        decision_artifact = json.load(f)

    # 1. Pre-flight checks MUST pass BEFORE touching test_chunks.json
    print("Executing Phase 6 Pre-flight Validation (6 Gates)...", flush=True)
    preflight_res = validate_phase6_preflight(
        decision_artifact=decision_artifact,
        base_dir=base_dir,
        threshold=0.0
    )
    print("Pre-flight Validation Passed successfully!", flush=True)

    # 2. Open test_chunks.json strictly ONCE from disk ONLY after pre-flight passes
    if not test_chunks_path.exists():
        raise TestSetIsolationException(
            f"TEST ISOLATION VIOLATION: Test dataset missing at '{test_chunks_path}'."
        )

    print("Opening test_chunks.json for single-pass read into memory...", flush=True)
    with open(test_chunks_path, "r", encoding="utf-8") as f:
        raw_test_chunks = json.load(f)

    device = get_device()
    evaluator = TestSetEvaluator(decision_artifact)

    test_data_by_task = {}
    representative_models = decision_artifact.get(
        "representative_models", decision_artifact.get("selected_models", {})
    )

    for task, model_info in representative_models.items():
        seed = model_info["seed"]
        run_name = model_info.get("run_name", f"run_{model_info.get('run_id', 0):02d}")
        ckpt_path = base_dir / "reports" / "experiments" / run_name / "best_model.pt"

        print(f"Evaluating task '{task}' using frozen model '{run_name}' (Seed {seed})...", flush=True)

        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        # Pass in-memory raw_test_chunks list to prevent reopening file from disk per task
        test_ds = CUADChunkDataset(
            data=raw_test_chunks,
            task_name=task,
            multi_span_option="first_span"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            worker_init_fn=seed_worker
        )

        model = SingleTaskModel(
            model_name="nlpaueb/legal-bert-base-uncased",
            pretrained=True
        )
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        task_y_true = []
        task_y_score = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                scores, _ = evaluate_chunk_span_logits(
                    outputs["start_logits"],
                    outputs["end_logits"],
                    attention_mask
                )
                has_pos = batch["has_positive"].cpu().numpy().astype(int)

                task_y_true.extend(has_pos)
                task_y_score.extend(scores.numpy())

        test_data_by_task[task] = {
            "y_true": task_y_true,
            "y_score": task_y_score
        }

    # 3. Pass to official Phase 6 evaluator path (threshold=0.0)
    official_results = evaluator.evaluate_official_phase6_test_set(
        test_data_by_task=test_data_by_task,
        threshold=0.0
    )

    # 4. Generate JSON and Markdown artifacts
    json_path = base_dir / "reports" / "experiments" / "phase6_test_evaluation.json"
    md_path = base_dir / "reports" / "experiments" / "phase6_test_evaluation.md"

    payload = {
        "protocol_reference": "RiskLoop SRD v1.1 Phase 6 Final Test Evaluation (FR-38..FR-40)",
        "test_set_isolation_statement": (
            "TEST SET EVALUATION EXECUTED EXACTLY ONCE POST-FREEZE: Test set (data/processed/test_chunks.json) "
            "evaluated for frozen representative models (FR-38, FR-39). Test set metrics cannot alter adoption decisions (FR-40)."
        ),
        "frozen_representative_models": representative_models,
        "test_results": official_results["test_results"]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Build Markdown report
    lines = [
        "# RiskLoop Phase 6 — Final Test Evaluation Report",
        "**Protocol Reference:** RiskLoop SRD v1.1 Phase 6 (FR-38..FR-40)  ",
        "**Test Set Isolation Statement:** TEST SET EVALUATION EXECUTED EXACTLY ONCE POST-FREEZE. "
        "Test set results strictly record performance of frozen representative models and cannot alter Phase 5C architecture adoption decisions (FR-40).",
        "",
        "---",
        "",
        "## 1. Frozen Representative Model Performance",
        "",
        "| Task | Condition | Run | Seed | Validation F1 (Phase 5C) | Final Test F1 (Phase 6) | Test Precision | Test Recall | Test Macro F1 | ROC-AUC | PR-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for task, res in official_results["test_results"].items():
        val_f1 = representative_models[task]["validation_score"]
        m = res["test_metrics"]
        roc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        lines.append(
            f"| **{task}** | {res['selected_condition']} | `{res['run_name']}` | {res['selected_seed']} | "
            f"{val_f1:.6f} | **{m['class_1_f1']:.6f}** | {m['class_1_precision']:.4f} | {m['class_1_recall']:.4f} | "
            f"{m['macro_f1']:.4f} | {roc_str} | {pr_str} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Phase 6 Test Evaluation completed successfully. Artifacts saved to '{json_path}' and '{md_path}'.", flush=True)


if __name__ == "__main__":
    main()

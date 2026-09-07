"""Unit and integration tests for Phase 5B Training Infrastructure.

SRD Traceability: FR-15..FR-16, FR-21..FR-28, NFR-06..NFR-08, NFR-11, NFR-13
"""

import json
import pytest
import torch
from pathlib import Path

from riskloop.experiment.trainer import (
    SingleRunTrainer,
    evaluate_chunk_span_logits,
    LOCKED_TRAINING_CONFIG,
    PRIMARY_METRIC_DISCLOSURE
)
from riskloop.experiment.runner import (
    generate_12_run_matrix,
    validate_experiment_isolation,
    execute_experiment_plan,
    ExperimentRunnerException
)


@pytest.fixture(scope="module")
def mock_dataset_files(tmp_path_factory):
    """Create lightweight mock train and val chunk JSON files for testing."""
    tmp_dir = tmp_path_factory.mktemp("mock_data")
    
    train_file = tmp_dir / "mock_train_chunks.json"
    val_file = tmp_dir / "mock_val_chunks.json"
    
    # Create 4 synthetic chunks
    mock_chunks = []
    for i in range(4):
        chunk = {
            "chunk_id": f"contract_{i:02d}__chunk_0000",
            "contract_id": f"contract_{i:02d}",
            "contract_title": f"Contract {i}",
            "partition": "train" if "train" in str(train_file) else "val",
            "chunk_idx": 0,
            "char_start": 0,
            "char_end": 100,
            "token_start": 0,
            "token_end": 20,
            "input_ids": [101] + [1000 + j for j in range(18)] + [102] + [0] * (512 - 20),
            "offset_mapping": [[0, 0]] + [[j * 5, (j + 1) * 5] for j in range(18)] + [[0, 0]] + [[0, 0]] * (512 - 20),
            "is_union_positive": (i % 2 == 0),
            "task_targets": {
                "Cap On Liability": {
                    "has_positive": (i == 0),
                    "spans": [(2, 5)] if (i == 0) else [(0, 0)]
                },
                "Anti-Assignment": {
                    "has_positive": (i == 2),
                    "spans": [(3, 6)] if (i == 2) else [(0, 0)]
                },
                "Termination For Convenience": {
                    "has_positive": False,
                    "spans": [(0, 0)]
                }
            }
        }
        mock_chunks.append(chunk)
        
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f)
        
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f)
        
    return train_file, val_file, tmp_dir


def test_12_run_manifest_expansion():
    tasks = ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]
    seeds = [42, 43, 44]
    
    matrix = generate_12_run_matrix(tasks, seeds)
    assert len(matrix) == 12
    
    cond_a = [r for r in matrix if r["condition"] == "Condition_A"]
    cond_b = [r for r in matrix if r["condition"] == "Condition_B"]
    
    assert len(cond_a) == 9
    assert len(cond_b) == 3
    
    # Check that Condition A maps each task to each seed
    for t in tasks:
        t_runs = [r for r in cond_a if r["task_scope"] == [t]]
        assert len(t_runs) == 3
        assert set(r["seed"] for r in t_runs) == set(seeds)
        
    # Check Condition B covers all 3 tasks across all 3 seeds
    for r in cond_b:
        assert sorted(r["task_scope"]) == sorted(tasks)
    assert set(r["seed"] for r in cond_b) == set(seeds)


def test_test_set_access_prohibition(mock_dataset_files):
    train_file, val_file, _ = mock_dataset_files
    
    run_info = {
        "run_id": 1,
        "condition": "Condition_A",
        "task_scope": ["Cap On Liability"],
        "seed": 42
    }
    
    # Attempt to pass a path containing "test"
    with pytest.raises(ExperimentRunnerException) as exc_info:
        SingleRunTrainer(
            run_info=run_info,
            train_data_path="data/processed/test_chunks.json",
            val_data_path=val_file,
            use_pretrained=False
        )
    assert "TEST SET ACCESS PROHIBITED" in str(exc_info.value)
    
    with pytest.raises(ExperimentRunnerException) as exc_info:
        execute_experiment_plan(
            train_data_path=str(train_file),
            val_data_path="data/processed/test_chunks.json"
        )
    assert "TEST SET ACCESS PROHIBITED" in str(exc_info.value)


def test_evaluate_chunk_span_logits():
    batch_size = 2
    seq_len = 512
    
    start_logits = torch.zeros(batch_size, seq_len)
    end_logits = torch.zeros(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Chunk 0: positive span at token (5, 8)
    start_logits[0, 5] = 4.0
    end_logits[0, 8] = 3.0
    start_logits[0, 0] = 1.0  # CLS
    end_logits[0, 0] = 1.0  # CLS
    
    # Chunk 1: negative / no-answer at CLS (0, 0)
    start_logits[1, 0] = 5.0
    end_logits[1, 0] = 5.0
    
    scores, spans = evaluate_chunk_span_logits(start_logits, end_logits, attention_mask)
    
    assert scores.shape == (2,)
    assert spans.shape == (2, 2)
    
    # Chunk 0 model score: (4.0 + 3.0) - (1.0 + 1.0) = 5.0
    assert pytest.approx(scores[0].item()) == 5.0
    assert tuple(spans[0].tolist()) == (5, 8)
    
    # Chunk 1 model score: max span score - no answer score
    assert tuple(spans[1].tolist()) == (0, 0) or scores[1].item() <= 0.0


def test_trainer_forward_loss_and_metrics_condition_a(mock_dataset_files):
    train_file, val_file, tmp_dir = mock_dataset_files
    
    run_info = {
        "run_id": 1,
        "condition": "Condition_A",
        "task_scope": ["Cap On Liability"],
        "seed": 42
    }
    
    trainer = SingleRunTrainer(
        run_info=run_info,
        train_data_path=train_file,
        val_data_path=val_file,
        output_dir=tmp_dir / "exp_a",
        use_pretrained=False
    )
    
    res = trainer.train(dry_run=True)
    assert res["status"] == "SUCCESS"
    assert res["dry_run"] is True
    assert "val_eval" in res
    assert "Cap On Liability" in res["val_eval"]["metrics_by_task"]
    assert "class_1_f1" in res["val_eval"]["metrics_by_task"]["Cap On Liability"]


def test_trainer_forward_loss_and_metrics_condition_b(mock_dataset_files):
    train_file, val_file, tmp_dir = mock_dataset_files
    
    run_info = {
        "run_id": 10,
        "condition": "Condition_B",
        "task_scope": ["Cap On Liability", "Anti-Assignment", "Termination For Convenience"],
        "seed": 42
    }
    
    trainer = SingleRunTrainer(
        run_info=run_info,
        train_data_path=train_file,
        val_data_path=val_file,
        output_dir=tmp_dir / "exp_b",
        use_pretrained=False
    )
    
    res = trainer.train(dry_run=True)
    assert res["status"] == "SUCCESS"
    assert res["dry_run"] is True
    assert "val_eval" in res
    metrics = res["val_eval"]["metrics_by_task"]
    assert "Cap On Liability" in metrics
    assert "Anti-Assignment" in metrics
    assert "Termination For Convenience" in metrics


def test_execute_experiment_plan_dry_run(mock_dataset_files):
    train_file, val_file, tmp_dir = mock_dataset_files
    
    plan_res = execute_experiment_plan(
        train_data_path=str(train_file),
        val_data_path=str(val_file),
        output_dir=str(tmp_dir / "full_plan"),
        dry_run=True,
        use_pretrained=False
    )
    
    assert plan_res["status"] == "SUCCESS"
    assert plan_res["total_runs"] == 12
    assert plan_res["dry_run"] is True
    assert len(plan_res["results"]) == 12


def test_locked_training_config_immutability():
    """Verify runtime immutability of top-level and nested parameters in LOCKED_TRAINING_CONFIG."""
    # Top-level immutability tests
    with pytest.raises(TypeError):
        LOCKED_TRAINING_CONFIG["learning_rate"] = 0.1

    with pytest.raises(TypeError):
        LOCKED_TRAINING_CONFIG["new_key"] = "unauthorized"

    with pytest.raises(TypeError):
        del LOCKED_TRAINING_CONFIG["learning_rate"]

    with pytest.raises(AttributeError):
        LOCKED_TRAINING_CONFIG.pop("learning_rate")

    # Nested list/value immutability tests
    with pytest.raises(AttributeError):
        LOCKED_TRAINING_CONFIG["selected_tasks"].append("Unauthorized Task")

    with pytest.raises(TypeError):
        LOCKED_TRAINING_CONFIG["selected_tasks"][0] = "Unauthorized Task"

    with pytest.raises(AttributeError):
        LOCKED_TRAINING_CONFIG["seeds"].append(999)

    with pytest.raises(TypeError):
        LOCKED_TRAINING_CONFIG["seeds"][0] = 999

    # Exact value preservation verification
    assert LOCKED_TRAINING_CONFIG["model_backbone"] == "nlpaueb/legal-bert-base-uncased"
    assert LOCKED_TRAINING_CONFIG["max_seq_length"] == 512
    assert LOCKED_TRAINING_CONFIG["doc_stride"] == 256
    assert list(LOCKED_TRAINING_CONFIG["selected_tasks"]) == [
        "Cap On Liability",
        "Anti-Assignment",
        "Termination For Convenience"
    ]
    assert list(LOCKED_TRAINING_CONFIG["seeds"]) == [42, 43, 44]
    assert LOCKED_TRAINING_CONFIG["learning_rate"] == 3e-5
    assert LOCKED_TRAINING_CONFIG["weight_decay"] == 0.01
    assert LOCKED_TRAINING_CONFIG["warmup_ratio"] == 0.10
    assert LOCKED_TRAINING_CONFIG["num_epochs"] == 4
    assert LOCKED_TRAINING_CONFIG["effective_batch_size"] == 16
    assert LOCKED_TRAINING_CONFIG["per_device_train_batch_size"] == 8
    assert LOCKED_TRAINING_CONFIG["gradient_accumulation_steps"] == 2
    assert LOCKED_TRAINING_CONFIG["head_init_mean"] == 0.0
    assert LOCKED_TRAINING_CONFIG["head_init_std"] == 0.02
    assert LOCKED_TRAINING_CONFIG["multi_span_training_policy"] == "first_span"
    assert LOCKED_TRAINING_CONFIG["primary_validation_metric"] == "class_1_f1"
    assert LOCKED_TRAINING_CONFIG["best_epoch_tie_breaker"] == "earlier_epoch"
    assert LOCKED_TRAINING_CONFIG["train_data_path"] == "data/processed/train_chunks.json"
    assert LOCKED_TRAINING_CONFIG["val_data_path"] == "data/processed/val_chunks.json"


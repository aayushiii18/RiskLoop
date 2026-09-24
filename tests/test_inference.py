"""Unit tests for RiskLoop Inference Engine and Model Loader.

SRD Traceability: FR-14..FR-16, FR-21, FR-22, FR-26, FR-37, NFR-06..NFR-08
"""

import os
import json
import pytest
import torch
import subprocess
import sys
from pathlib import Path

from riskloop.inference.loader import ModelLoader, REPRESENTATIVE_RUN_MAPPING
from riskloop.inference.predictor import RiskLoopPredictor, decode_inference_chunk_logits


def test_model_loader_task_validation():
    """Verify ModelLoader handles invalid tasks with descriptive ValueError."""
    loader = ModelLoader()
    with pytest.raises(ValueError, match="Unsupported task"):
        loader.load_model_for_task("Invalid Task Name")


def test_model_loader_checkpoint_overrides(tmp_path):
    """Verify ModelLoader respects custom checkpoint_overrides."""
    fake_ckpt = tmp_path / "fake_model.pt"
    fake_ckpt.write_text("dummy")

    loader = ModelLoader(
        base_dir=tmp_path,
        checkpoint_overrides={"Cap On Liability": fake_ckpt}
    )
    resolved = loader.resolve_checkpoint_path("Cap On Liability")
    assert resolved == fake_ckpt.resolve()


def test_model_loader_missing_checkpoint(tmp_path):
    """Verify ModelLoader returns None when checkpoint files do not exist."""
    loader = ModelLoader(base_dir=tmp_path)
    resolved = loader.resolve_checkpoint_path("Cap On Liability")
    assert resolved is None


def test_decode_inference_chunk_logits_special_token_masking():
    """Deterministic test verifying special tokens ([CLS], [SEP], [PAD]) are masked out from evidence spans."""
    batch_size = 1
    seq_len = 512

    start_logits = torch.zeros((batch_size, seq_len))
    end_logits = torch.zeros((batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask[:, 100:] = 0  # Tokens >= 100 are padded

    # Mock offsets: [0, 0] for CLS (0), text for 1..10, [0, 0] for SEP (11), [0, 0] for PAD (12..511)
    chunk_offsets = [[ [0, 0] if i in (0, 11) or i >= 12 else [i * 5, (i + 1) * 5] for i in range(seq_len) ]]

    # Set highest logits on [SEP] token (index 11)
    start_logits[0, 11] = 100.0
    end_logits[0, 11] = 100.0

    # Set moderate logits on valid text tokens (start 2, end 5)
    start_logits[0, 2] = 5.0
    end_logits[0, 5] = 4.0

    # No-answer score at token 0 ([CLS]) = 0 + 0 = 0.0
    start_logits[0, 0] = 1.0
    end_logits[0, 0] = 1.0

    scores, spans = decode_inference_chunk_logits(start_logits, end_logits, attention_mask, chunk_offsets)

    # Must select valid text tokens [2, 5] instead of special token 11 ([SEP])
    assert spans[0, 0].item() == 2
    assert spans[0, 1].item() == 5
    # Model score must equal max_valid_span (5.0 + 4.0) - no_answer (1.0 + 1.0) = 7.0
    assert abs(scores[0].item() - 7.0) < 1e-4


def test_predictor_chunking_short_text():
    """Verify tokenization and chunking for short contract text (fits in 1 chunk)."""
    predictor = RiskLoopPredictor(pretrained=False)
    text = "Neither Party shall assign this Agreement without prior written consent."
    chunks = predictor.tokenize_and_chunk(text)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_idx"] == 0
    assert len(chunk["input_ids"]) == 512
    assert len(chunk["attention_mask"]) == 512
    assert chunk["chunk_offsets"][0] == [0, 0]  # CLS
    assert chunk["char_start"] == 0
    assert chunk["char_end"] == len(text)


def test_predictor_chunking_empty_text():
    """Verify tokenization and chunking for empty or whitespace text."""
    predictor = RiskLoopPredictor(pretrained=False)
    assert predictor.tokenize_and_chunk("") == []
    assert predictor.tokenize_and_chunk("   \n\t  ") == []

    result = predictor.predict("")
    assert result["contract_length_chars"] == 0
    assert result["num_chunks"] == 0
    for task_name, task_res in result["tasks"].items():
        assert task_res["detected"] is False
        assert task_res["predicted_text"] is None


def test_predictor_chunking_long_text_offset_alignment():
    """Verify sliding window chunking with 256-stride preserves precise original character offsets."""
    predictor = RiskLoopPredictor(pretrained=False)
    long_text = ("Section 1. Terms and conditions of the agreement. " * 120)
    chunks = predictor.tokenize_and_chunk(long_text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["input_ids"][0] == predictor.tokenizer.cls_token_id

        # Verify every non-zero offset maps strictly within long_text bounds
        for s_char, e_char in chunk["chunk_offsets"]:
            if [s_char, e_char] != [0, 0] and s_char < e_char:
                assert 0 <= s_char < e_char <= len(long_text)
                # Verify character slice is valid text
                assert len(long_text[s_char:e_char]) > 0


def test_predictor_end_to_end_schema_and_offset_slicing():
    """Verify full prediction pipeline schema, offset validity, and text slice equality."""
    predictor = RiskLoopPredictor(pretrained=False)
    sample_text = (
        "3. ANTI-ASSIGNMENT\n"
        "Neither party may assign, transfer, or delegate any of its rights under this Agreement.\n\n"
        "4. LIMITATION OF LIABILITY\n"
        "In no event shall total liability exceed $50,000 USD."
    )

    result = predictor.predict(sample_text)

    assert "contract_length_chars" in result
    assert result["contract_length_chars"] == len(sample_text)
    assert "num_chunks" in result
    assert result["num_chunks"] >= 1
    assert "tasks" in result

    tasks = result["tasks"]
    assert "Cap On Liability" in tasks
    assert "Anti-Assignment" in tasks
    assert "Termination For Convenience" in tasks

    for task_name, res in tasks.items():
        assert res["task"] == task_name
        assert isinstance(res["detected"], bool)
        assert isinstance(res["model_score"], float)
        assert "model" in res
        assert "seed" in res

        # ENFORCE RULE: Never detected=True with predicted_text=None/N/A
        if res["detected"]:
            assert res["predicted_text"] is not None
            c_start = res["character_start"]
            c_end = res["character_end"]
            assert c_start is not None and c_end is not None
            assert 0 <= c_start < c_end <= len(sample_text)
            assert sample_text[c_start:c_end] == res["predicted_text"]
        else:
            assert res["predicted_text"] is None
            assert res["character_start"] is None
            assert res["character_end"] is None


def test_predict_contract_cli_help():
    """Verify predict_contract.py CLI tool responds to --help."""
    script_path = Path(__file__).parent.parent / "scripts" / "predict_contract.py"
    cmd = [sys.executable, str(script_path), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "RiskLoop Contract Risk Analysis CLI" in res.stdout


def test_predict_contract_cli_execution():
    """Verify predict_contract.py execution with sample file and --json flag."""
    script_path = Path(__file__).parent.parent / "scripts" / "predict_contract.py"
    sample_path = Path(__file__).parent.parent / "examples" / "sample_contract_nda.txt"

    cmd = [sys.executable, str(script_path), "--file", str(sample_path), "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0

    data = json.loads(res.stdout)
    assert "contract_length_chars" in data
    assert "tasks" in data
    assert "Cap On Liability" in data["tasks"]

    # Verify JSON task results follow strict schema rules
    for task_name, task_res in data["tasks"].items():
        if task_res["detected"]:
            assert task_res["predicted_text"] is not None
            assert task_res["character_start"] is not None
            assert task_res["character_end"] is not None
        else:
            assert task_res["predicted_text"] is None

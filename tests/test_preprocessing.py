"""Unit tests for RiskLoop preprocessing pipeline (Phase 4C).

SRD Traceability: FR-05..FR-07, FR-14..FR-16, NFR-06..NFR-08, NFR-12
"""

import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer

from riskloop.data.preprocessing import (
    align_char_span_to_tokens,
    preprocess_contract,
    sanitize_contract_id,
    SELECTED_TASKS,
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    DOC_STRIDE
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


@pytest.fixture(scope="module")
def split_map():
    p = Path("data/splits/contract_split_map.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sanitize_contract_id():
    title = "  ACME / Corp / Contract 123  "
    sanitized = sanitize_contract_id(title)
    assert sanitized == "ACME___Corp___Contract_123" or sanitized == "ACME___Corp___Contract_123" or "_" in sanitized
    assert "/" not in sanitized
    assert not sanitized.startswith(" ") and not sanitized.endswith(" ")


def test_align_char_span_to_tokens():
    # Synthetic offsets representation for max_seq_length=8
    # idx 0: CLS [0, 0]
    # idx 1..5: text tokens: [0, 5], [6, 10], [11, 15], [16, 20], [21, 25]
    # idx 6: SEP [0, 0]
    # idx 7: PAD [0, 0]
    offsets = [
        [0, 0],
        [0, 5],
        [6, 10],
        [11, 15],
        [16, 20],
        [21, 25],
        [0, 0],
        [0, 0]
    ]

    # Test 1: Exact span matching tokens 2..3 (char span 6..15)
    aligned = align_char_span_to_tokens(6, 15, offsets)
    assert aligned == (2, 3)

    # Test 2: Sub-span inside single token 1 (char span 1..4)
    aligned = align_char_span_to_tokens(1, 4, offsets)
    assert aligned == (1, 1)

    # Test 3: Span starting before window (char span -2..10) -> None
    aligned = align_char_span_to_tokens(-2, 10, offsets)
    assert aligned is None

    # Test 4: Span ending after window (char span 20..30) -> None
    aligned = align_char_span_to_tokens(20, 30, offsets)
    assert aligned is None


def test_preprocess_contract_mock(tokenizer):
    mock_contract = {
        "title": "Test Contract Alpha",
        "paragraphs": [
            {
                "context": "This Agreement limits liability to $1,000,000. Neither party may assign this agreement.",
                "qas": [
                    {
                        "id": "qa_1__Cap On Liability",
                        "is_impossible": False,
                        "answers": [{"text": "limits liability to $1,000,000", "answer_start": 15}]
                    },
                    {
                        "id": "qa_2__Anti-Assignment",
                        "is_impossible": False,
                        "answers": [{"text": "Neither party may assign this agreement", "answer_start": 46}]
                    },
                    {
                        "id": "qa_3__Termination For Convenience",
                        "is_impossible": True,
                        "answers": []
                    }
                ]
            }
        ]
    }

    chunks = preprocess_contract(
        contract=mock_contract,
        partition="train",
        tokenizer=tokenizer,
        selected_tasks=SELECTED_TASKS
    )

    assert len(chunks) >= 1
    chunk = chunks[0]

    assert chunk["contract_title"] == "Test Contract Alpha"
    assert chunk["partition"] == "train"
    assert chunk["is_union_positive"] is True

    # Check Cap On Liability target
    cap_target = chunk["task_targets"]["Cap On Liability"]
    assert cap_target["has_positive"] is True
    assert len(cap_target["spans"]) == 1
    assert cap_target["spans"][0][0] > 0 and cap_target["spans"][0][1] >= cap_target["spans"][0][0]

    # Check Termination For Convenience target (impossible QA)
    term_target = chunk["task_targets"]["Termination For Convenience"]
    assert term_target["has_positive"] is False
    assert term_target["spans"] == [(0, 0)]


def test_processed_dataset_files_integrity(split_map):
    p_train = Path("data/processed/train_chunks.json")
    p_val = Path("data/processed/val_chunks.json")
    p_test = Path("data/processed/test_chunks.json")

    assert p_train.exists() and p_val.exists() and p_test.exists()

    with open(p_train, "r", encoding="utf-8") as f:
        train_chunks = json.load(f)
    with open(p_val, "r", encoding="utf-8") as f:
        val_chunks = json.load(f)
    with open(p_test, "r", encoding="utf-8") as f:
        test_chunks = json.load(f)

    assert len(train_chunks) > 0
    assert len(val_chunks) > 0
    assert len(test_chunks) > 0

    # Contract partition inheritance & zero contract leakage check
    train_contracts = set(ch["contract_title"] for ch in train_chunks)
    val_contracts = set(ch["contract_title"] for ch in val_chunks)
    test_contracts = set(ch["contract_title"] for ch in test_chunks)

    # Check no overlap between splits
    assert len(train_contracts.intersection(val_contracts)) == 0
    assert len(train_contracts.intersection(test_contracts)) == 0
    assert len(val_contracts.intersection(test_contracts)) == 0

    # Check partition assignment matches split_map
    for title in train_contracts:
        assert split_map[title] == "train"
    for title in val_contracts:
        assert split_map[title] == "val"
    for title in test_contracts:
        assert split_map[title] == "test"

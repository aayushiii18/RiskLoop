"""Unit tests for RiskLoop model architectures and PyTorch Dataset module.

SRD Traceability: FR-15, FR-21, FR-22, FR-23, FR-24
"""

import pytest
import torch
from riskloop.data.dataset import CUADChunkDataset
from riskloop.models.single_task import SingleTaskModel
from riskloop.models.multi_task import MultiTaskModel

@pytest.fixture
def mock_chunk_records():
    """Generate mock 512-token chunk records matching preprocessed JSON schema."""
    input_ids = [101] + [2000 + i for i in range(500)] + [102] + [0] * 10
    assert len(input_ids) == 512
    
    return [
        {
            "chunk_id": "test_contract_001__chunk_0000",
            "contract_id": "test_contract_001",
            "contract_title": "Test Contract 001",
            "partition": "train",
            "chunk_idx": 0,
            "input_ids": input_ids,
            "is_union_positive": True,
            "task_targets": {
                "Cap On Liability": {
                    "has_positive": True,
                    "spans": [[20, 50], [80, 100]]  # Multi-span chunk
                },
                "Anti-Assignment": {
                    "has_positive": False,
                    "spans": [[0, 0]]
                },
                "Termination For Convenience": {
                    "has_positive": True,
                    "spans": [[120, 140]]
                }
            }
        },
        {
            "chunk_id": "test_contract_001__chunk_0001",
            "contract_id": "test_contract_001",
            "contract_title": "Test Contract 001",
            "partition": "train",
            "chunk_idx": 1,
            "input_ids": input_ids,
            "is_union_positive": False,
            "task_targets": {
                "Cap On Liability": {"has_positive": False, "spans": [[0, 0]]},
                "Anti-Assignment": {"has_positive": False, "spans": [[0, 0]]},
                "Termination For Convenience": {"has_positive": False, "spans": [[0, 0]]}
            }
        }
    ]


def test_cuad_chunk_dataset_single_task(mock_chunk_records):
    """Verify single-task dataset item shapes and First-Span target selection."""
    ds = CUADChunkDataset(
        data=mock_chunk_records,
        task_name="Cap On Liability",
        multi_span_option="first_span"
    )
    assert len(ds) == 2
    
    item0 = ds[0]
    assert item0["input_ids"].shape == (512,)
    assert item0["attention_mask"].shape == (512,)
    assert item0["token_type_ids"].shape == (512,)
    
    # First span option selects [20, 50] for item0
    assert item0["start_positions"].item() == 20
    assert item0["end_positions"].item() == 50
    assert item0["has_positive"].item() is True
    
    item1 = ds[1]
    # Negative chunk selects (0, 0)
    assert item1["start_positions"].item() == 0
    assert item1["end_positions"].item() == 0
    assert item1["has_positive"].item() is False


def test_cuad_chunk_dataset_multi_task(mock_chunk_records):
    """Verify multi-task dataset item shapes across all 3 tasks."""
    ds = CUADChunkDataset(
        data=mock_chunk_records,
        task_name=None  # Multi-task mode
    )
    assert len(ds) == 2
    
    item0 = ds[0]
    assert "targets" in item0
    targets = item0["targets"]
    assert "Cap On Liability" in targets
    assert "Anti-Assignment" in targets
    assert "Termination For Convenience" in targets
    
    assert targets["Cap On Liability"]["start_positions"].item() == 20
    assert targets["Cap On Liability"]["end_positions"].item() == 50
    assert targets["Anti-Assignment"]["start_positions"].item() == 0
    assert targets["Termination For Convenience"]["start_positions"].item() == 120


def test_single_task_model_forward_shapes(mock_chunk_records):
    """Verify SingleTaskModel forward pass tensor shapes (B=2, L=512)."""
    ds = CUADChunkDataset(data=mock_chunk_records, task_name="Cap On Liability")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(dataloader))
    
    model = SingleTaskModel(pretrained=False, hidden_size=256)
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"]
        )
        
    assert outputs["start_logits"].shape == (2, 512)
    assert outputs["end_logits"].shape == (2, 512)
    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0  # Scalar tensor


def test_multi_task_model_forward_shapes(mock_chunk_records):
    """Verify MultiTaskModel forward pass tensor shapes and loss aggregation."""
    ds = CUADChunkDataset(data=mock_chunk_records, task_name=None)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(dataloader))
    
    model = MultiTaskModel(pretrained=False, hidden_size=256)
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            targets=batch["targets"]
        )
        
    logits_by_task = outputs["logits_by_task"]
    assert len(logits_by_task) == 3
    
    for task_name, (start_logits, end_logits) in logits_by_task.items():
        assert start_logits.shape == (2, 512)
        assert end_logits.shape == (2, 512)
        
    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0
    assert "task_losses" in outputs
    assert len(outputs["task_losses"]) == 3


def test_model_parameter_sharing_isolation():
    """Verify parameter sharing in MultiTaskModel vs isolation in SingleTaskModel."""
    model_single = SingleTaskModel(pretrained=False, hidden_size=256)
    model_multi = MultiTaskModel(pretrained=False, hidden_size=256)
    
    params_single = sum(p.numel() for p in model_single.parameters())
    params_multi = sum(p.numel() for p in model_multi.parameters())
    
    # Single task model has 1 head (256*2 + 2 = 514 params), multi-task has 3 heads (1542 params)
    # Difference should be exactly 2 extra heads * 514 = 1028 parameters
    assert params_multi - params_single == 1028

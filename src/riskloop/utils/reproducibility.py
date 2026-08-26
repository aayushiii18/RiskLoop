"""Reproducibility utilities for seed locking."""

import random
import numpy as np

def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (if installed).
    
    SRD Traceability: NFR-06, NFR-07
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

import random 
import numpy as np 
import torch 
import os 

def set_seed(seed:int = 42):
    """
    Set all random seeds for Reproducibility

    Critical for:
    - Debugging
    - Experiment comparision
    - Scientific Rigor
    Note: Even with the seeds, GPU ops may have non-deterministic behaviour
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASSEED'] = str(seed)

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get computation device with fallback
    """
    
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
        print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

    

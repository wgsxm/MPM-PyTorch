from torch import device

from .elasticity import *
from .plasticity import *

import mpm_pytorch.constitutive_models as constitutive_models

def get_constitutive(constitutive_name: str, device: device='cuda'): 
    return getattr(constitutive_models, constitutive_name)().to(device=device)
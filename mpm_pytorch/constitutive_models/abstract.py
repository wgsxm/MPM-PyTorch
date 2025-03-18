from typing import *

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .warp_svd import SVD

class Material(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 3
        self.useless = nn.Parameter(torch.Tensor([1.0]))
        self.svd = SVD()

    def transpose(self, F: Tensor) -> Tensor:
        return F.permute(0, 2, 1)

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError


class Elasticity(Material):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        # F -> P
        raise NotImplementedError


class Plasticity(Material):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        # F -> F
        raise NotImplementedError
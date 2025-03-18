from typing import *

import torch
import torch.nn as nn
from torch import Tensor

from .abstract import Elasticity

class SigmaElasticity(Elasticity):
    def __init__(self, E: float=2e6, nu: float=0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)
            
        # warp svd
        U, sigma, Vh = self.svd(F)
        thredhold = 0.001
        sigma = torch.clamp_min(sigma, thredhold)
        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace
        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))
        return stress

class CorotatedElasticity(Elasticity):
    def __init__(self, E: float=2e6, nu: float=0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)
        
        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        assert torch.all(torch.isfinite(J))
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        assert torch.all(torch.isfinite(stress))
        return stress
    
class FluidElasticity(Elasticity):
    def __init__(self, E: float=2e6, nu: float=0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        mu = 0
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)
        
        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        assert torch.all(torch.isfinite(J))
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        assert torch.all(torch.isfinite(stress))
        return stress
    
class StVKElasticity(Elasticity):
    def __init__(self, E: float=2e6, nu: float=0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
        
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        E = 0.5 * (FtF - I)

        stvk_stress = 2 * mu * torch.matmul(F, E)

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        volume_stress = la * J * (J - 1) * I

        stress = stvk_stress + volume_stress

        return stress
    
    
class VolumeElasticity(Elasticity):
    def __init__(self, E: float=2e6, nu: float=0.4) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))


        self.mode = 'taichi'

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)

        if self.mode.casefold() == 'ziran':

            #  https://en.wikipedia.org/wiki/Bulk_modulus
            kappa = 2 / 3 * mu + la

            # https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/ConstitutiveModel/EquationOfState.h
            # using gamma = 7 would have gradient issue, fix later
            gamma = 2

            stress = kappa * (J - 1 / torch.pow(J, gamma-1)) * I

        elif self.mode.casefold() == 'taichi':

            stress = la * J * (J - 1) * I

        else:
            raise ValueError('invalid mode for volume plasticity: {}'.format(self.mode))

        return stress
import math
from typing import *
import torch
import torch.nn as nn
from torch import Tensor

from .abstract import Plasticity

class DruckerPragerPlasticity(Plasticity):
    def __init__(self, E: float=2e6, nu: float=0.4, friction_angle: float=25.0, cohesion: float=0.0) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))
        self.register_buffer('friction_angle', torch.Tensor([friction_angle]))
        self.register_buffer('cohesion', torch.Tensor([cohesion]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:

        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan
        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F
    
class IdentityPlasticity(Plasticity):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        return F
    
    
class SigmaPlasticity(Plasticity):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        J = torch.clamp(J, min=0.05, max=1.2)

        Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
        F = torch.diag_embed(Je_1_3)
        return F
    
    
class VonMisesPlasticity(Plasticity):
    def __init__(self, E: float=2e6, nu: float=0.4, sigma_y=1e3) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([E]).log())
        self.register_buffer('nu', torch.Tensor([nu]))
        self.register_buffer('sigma_y', torch.Tensor([sigma_y]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:


        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        sigma_y = self.sigma_y

        mu = E / (2 * (1 + nu))
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan
        
        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)

        F = torch.where(cond_yield, yield_F, F)

        return F
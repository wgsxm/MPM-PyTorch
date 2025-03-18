from typing import *

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig

class MPMSolver:
    def __init__(
        self, 
        # material parameters
        init_pos: Tensor, 
        rho: float=1000.0,
        # simulation parameters
        num_grids: int=25,
        dt: float=3e-4,
        gravity: List[float]=[0.0, 0.0, -9.8],
        clip_bound: float=0.5,
        damping: float=1.0,
        center: List[float]=[0.5, 0.5, 0.5],
        size: List[float]=[1.0, 1.0, 1.0],
        # training parameters
        enable_train: bool=False,
        device: torch.device='cuda',
    ):
        # save simulation parameters
        self.num_grids: int = num_grids
        self.dt: float = dt
        self.gravity: Tensor = torch.tensor(gravity, device=device)

        self.dx: float = 1 / self.num_grids
        self.inv_dx: float = float(self.num_grids)
        
        self.clip_bound: float = clip_bound * self.dx
        self.damping = damping
        assert self.clip_bound >= 0.0
        assert self.damping >= 0.0 and self.damping <= 1.0
        
        self.n_particles: int = init_pos.shape[0]
        self.init_pos: Tensor = init_pos.detach()
        
        self.center: np.ndarray = np.array(center)
        self.size: np.ndarray = np.array(size)
        self.vol: float = np.prod(self.size) / self.n_particles
        self.p_mass: float = rho * self.vol  # TODO: the mass can be non-constant.

        self.enable_train: bool = enable_train
        self.device: torch.device = device
        
        # init tensors
        num_grids = self.num_grids
        n_dim = 3 # 3D
        self.grid_mv = torch.empty((num_grids ** n_dim, n_dim), device=device)
        self.grid_m = torch.empty((num_grids ** n_dim,), device=device)
        grid_ranges = torch.arange(num_grids, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_ranges, grid_ranges, grid_ranges, indexing='ij')
        self.grid_x = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3).float() # (n_grid * n_grid * n_grid, 3)
        
        self.offset = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=device).float() # (27, 3)

        # bc
        self.pre_particle_process = []
        self.post_grid_process = []

        self.time = 0.0        
        
    def reset(self) -> None:
        self.time = 0.0
    
    def __call__(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.p2g2p(x, v, C, F, stress)
    
    def p2g2p(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        # prepare constants
        dt = self.dt
        vol = self.vol
        p_mass = self.p_mass 
        dx = self.dx
        inv_dx = self.inv_dx 
        n_grids = self.num_grids
        n_particles = self.n_particles
        clip_bound = self.clip_bound
        
        # calculate temporary variables for both p2g and g2p (weight, dpos, index)
        px = x * inv_dx
        base = (px - 0.5).long() # (n_particles, 3)
        fx = px - base.float() # (n_particles, 3)
        
        w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2
        ]
        w = torch.stack(w, dim=-1) # (n_particles, 3, 3)
        w_e = torch.einsum('bi, bj, bk -> bijk', w[:, 0], w[:, 1], w[:, 2]) # (n_particles, 3, 3, 3)
        weight = w_e.reshape(-1, 27) # (n_particles, 27)
        
        dw = [
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5
        ]
        dw = torch.stack(dw, dim=-1) # (n_particles, 3, 3)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2])
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3) # (n_particles, 3, 3, 3, 3) -> (n_particles, 27, 3)
        
        dpos = (self.offset - fx.unsqueeze(1)) * dx # (n_particles, 27, 3)
        
        index = base.unsqueeze(1) + self.offset.unsqueeze(0).long() # (n_particles, 27, 3)
        index = (index[:, :, 0] * n_grids * n_grids + index[:, :, 1] * n_grids + index[:, :, 2]).reshape(-1) # (n_particles * 27)
        index = index.clamp(0, n_grids ** 3 - 1) # (n_particles * 27) TODO: simple clipping leads to some numerical problems, but it's acceptable for now.
        
        # zero grid
        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m = torch.zeros_like(self.grid_m)
        
        # pre-particle operation
        for operation in self.pre_particle_process:
            operation(self, x, v)
        
        # p2g
        mv = -dt * vol * torch.einsum('bij, bkj -> bki', stress, dweight) +\
            p_mass * weight.unsqueeze(2) * (v.unsqueeze(1) + torch.einsum('bij, bkj -> bki', C, dpos)) # (n_particles, 3, 3), (n_particles, 27, 3) -> (n_particles, 27, 3)
        mv = mv.reshape(-1, 3) # (n_particles * 27, 3)
        
        m = weight * p_mass # (n_particles, 27)
        m = m.reshape(-1) # (n_particles * 27)
        
        self.grid_mv = self.grid_mv.index_add(dim=0, index=index, source=mv) # (n_grid * n_grid * n_grid, 3)
        self.grid_m = self.grid_m.index_add(dim=0, index=index, source=m) # (n_grid * n_grid * n_grid)        
        
        # grid update
        self.grid_update()
        
        # post-grid operation
        for operation in self.post_grid_process:
            operation(self)
        
        # g2p
        v = self.grid_mv.index_select(dim=0, index=index).reshape(-1, 27, 3) # (n_particles, 27, 3)
        C = torch.einsum('bij, bik -> bijk', v, dpos) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        new_F = torch.einsum('bij, bik -> bijk', v, dweight) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        
        v = (weight.unsqueeze(2) * v).sum(dim=1) # (n_particles, 3)
        C = (4.0 * inv_dx * inv_dx * weight.unsqueeze(2).unsqueeze(3) * C).sum(dim=1)# (n_particles, 3, 3)
        new_F = dt * new_F.sum(dim=1) # (n_particles, 3, 3)
        
        x = x + v * dt
        x = x.clamp(clip_bound, 1.0 - clip_bound)
        F = F + torch.bmm(new_F, F)
        F = F.clamp(-2.0, 2.0)
        self.time += dt
        
        return x, v, C, F
    
    def grid_update(self) -> None:
        selected_idx = self.grid_m > 1e-15
        self.grid_mv[selected_idx] = self.grid_mv[selected_idx] / (self.grid_m[selected_idx].unsqueeze(1))
        self.grid_mv = self.damping * (self.grid_mv + self.dt * self.gravity)
        
    def pre_p2g_operation(self) -> None:
        pass
    
    def post_grid_operation(self) -> None:
        pass
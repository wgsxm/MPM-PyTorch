from typing import *

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive

def get_cube(
        center: List[float], 
        size: List[float], 
        num: int, 
        add_noise: bool=False, 
        device: torch.device=torch.device("cuda")
    ) -> Tensor:
    start = torch.tensor(center) - torch.tensor(size) / 2
    end = torch.tensor(center) + torch.tensor(size) / 2
    # Generate a cube
    x = torch.linspace(start[0], end[0], num)
    y = torch.linspace(start[1], end[1], num)
    z = torch.linspace(start[2], end[2], num)
    cube = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).view(-1, 3)
    if add_noise:
        # Add noise to the cube
        noisy_cube = start + torch.rand_like(cube) * (end - start)
        cube = torch.cat([cube, noisy_cube], dim=0)
    return cube.to(device)

def visualize_frames(
    frames: List[np.ndarray], 
    export_path: str, 
    center: List[float] = [0.5, 0.5, 0.5],
    size: List[float] = [2.0, 2.0, 2.0],
    c: str = 'blue',
    s: float = 20,
    fps: int = 30,
): 
    xlim = [center[0] - size[0] / 2, center[0] + size[0] / 2]
    ylim = [center[1] - size[1] / 2, center[1] + size[1] / 2]
    zlim = [center[2] - size[2] / 2, center[2] + size[2] / 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], s=s)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    def update(frame):
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        scat = ax.scatter(frames[frame][:, 0], frames[frame][:, 1], frames[frame][:, 2], s=s, c=c)
        ax.set_title(f'Frame {frame}')
        return scat
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    print(f'Start simulation with config: {args.config}')

    # Load config
    cfg = OmegaConf.load(args.config)
    material_params = cfg.material
    sim_params = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    export_path = os.path.join(cfg.output_dir, cfg.tag + ".gif")

    # Create a cube for simulation
    particles = get_cube(
        center=[0.5, 0.5, 0.5], 
        size=[0.5, 0.5, 0.5], 
        num=10, 
        add_noise=True,
        device=device
    )
    n_particles = particles.shape[0]

    # Initialize MPM solver
    mpm_solver = MPMSolver(
        particles, 
        enable_train=False,
        device=device
    )
    set_boundary_conditions(mpm_solver, sim_params.boundary_conditions)
    # Initialize Constitutive models
    elasticity = get_constitutive(material_params.elasticity, device=device)
    plasticity = get_constitutive(material_params.plasticity, device=device)

    # Initialize particle states
    x = particles
    v = torch.stack([torch.tensor(sim_params.initial_velocity, device=device) for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    # Run simulation
    frames = []
    for frame in tqdm(range(sim_params.num_frames), desc='Simulating', leave=False):
        frames.append(x.cpu().numpy())
        for step in tqdm(range(sim_params.steps_per_frame), desc='Step', leave=False):
            # Update stress
            stress = elasticity(F)
            # Particle to grid, grid update, grid to particle
            x, v, C, F = mpm_solver(x, v, C, F, stress)
            # Plasticity correction
            F = plasticity(F)
    
    # Visualize
    print(f'Rendering to {export_path}...')
    visualize_frames(
        frames, 
        export_path=export_path, 
        size=[1, 1, 1], 
        c=material_params.color
    )
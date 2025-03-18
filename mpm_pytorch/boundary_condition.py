from typing import *

from functools import partial
from omegaconf import DictConfig
import torch
from torch import Tensor

from .mpm_solver import MPMSolver

def set_boundary_conditions(model: MPMSolver, bc_params: DictConfig):
    if bc_params is None:
        return
    for bc in bc_params:
        if bc["type"] == "cuboid":
            assert "point" in bc.keys()
            assert "size" in bc.keys()
            assert "velocity" in bc.keys()
            
            start_time = 0.0
            end_time = 1e3
            reset = False
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            if "end_time" in bc.keys():
                end_time = bc["end_time"]
            if "reset" in bc.keys():
                reset = bc["reset"]
            set_velocity_on_cuboid(
                model,
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=start_time,
                end_time=end_time,
                reset=reset,
            )

        elif bc["type"] == "particle_impulse":
            assert "force" in bc.keys()

            start_time = 0.0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            num_dt = 1
            if "num_dt" in bc.keys():
                num_dt = bc["num_dt"]
            point = [1, 1, 1]
            if "point" in bc.keys():
                point = bc["point"]
            size = [1, 1, 1]
            if "size" in bc.keys():
                size = bc["size"]

            add_impulse_on_particles(
                model,
                force=bc["force"],
                dt=model.dt,
                point=point,
                size=size,
                num_dt=num_dt,
                start_time=start_time,
            )

        elif bc["type"] == "enforce_particle_translation":
            assert "point" in bc.keys()
            assert "size" in bc.keys()
            assert "velocity" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            enforce_particle_velocity_translation(
                model,
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        
        elif bc["type"] == "sdf_collider":
            assert "bound" in bc.keys()
            assert "dim" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            
            add_sdf_collider(
                model,
                bound=bc["bound"],
                dim=bc["dim"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        
        elif bc["type"] == "surface_collider":
            assert "point" in bc.keys()
            assert "normal" in bc.keys()
            assert "surface" in bc.keys()
            assert "friction" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            add_surface_collider(
                model,
                point=bc["point"],
                normal=bc["normal"],
                surface=bc["surface"],
                friction=bc["friction"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
            
        elif bc["type"] == "enforce_particle_velocity_rotation":
            assert "normal" in bc.keys()
            assert "point" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "half_height_and_radius" in bc.keys()
            assert "rotation_scale" in bc.keys()
            assert "translation_scale" in bc.keys()

            enforce_particle_velocity_rotation(
                model,
                point=bc["point"],
                normal=bc["normal"],
                half_height_and_radius=bc["half_height_and_radius"],
                rotation_scale=bc["rotation_scale"],
                translation_scale=bc["translation_scale"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        else:
            raise TypeError("Undefined BC type")

# add a surface collider with a given SDF
# x = x - sdf(x) * sdf_grad(x)
# v = v - sdf(x) * sdf_grad(x) / dt
# pre_particle_process
def add_sdf_collider(
    model: MPMSolver,
    bound: float,
    dim: int,
    start_time: float=0.0,
    end_time: float=999.0,
):
    def sdf_collider(
        model: MPMSolver,
        x: Tensor, v: Tensor,
        bound: float, dim: int, 
        start_time: float, end_time: float
    ):
        time = model.time
        if time >= start_time and time < end_time:
            
            if dim == 0:
                sdf_grad = torch.tensor([1.0, 0.0, 0.0], device=model.device).float()
            elif dim == 1:
                sdf_grad = torch.tensor([0.0, 1.0, 0.0], device=model.device).float()
            elif dim == 2:
                sdf_grad = torch.tensor([0.0, 0.0, 1.0], device=model.device).float()
            
            choice = x[:, dim] < bound
            sdf = x[choice, dim] - bound
            x[choice] = x[choice] - sdf[:, None] * sdf_grad
            v[choice] = v[choice] - sdf[:, None] * sdf_grad / model.dt
            
            up_bound = 1.0 - bound
            choice = x[:, dim] > up_bound
            sdf = x[choice, dim] - up_bound
            x[choice] = x[choice] - sdf[:, None] * sdf_grad
            v[choice] = v[choice] - sdf[:, None] * sdf_grad / model.dt
            
    model.pre_particle_process.append(
        partial(
            sdf_collider,
            bound=bound,
            dim=dim,
            start_time=start_time,
            end_time=end_time,
        )
    )
       
        
# a cubiod is a rectangular cube'
# centered at `point`
# dimension is x: point[0]+-size[0]
#              y: point[1]+-size[1]
#              z: point[2]+-size[2]
# all grid nodes lie within the cubiod will have their speed set to velocity
# the cuboid itself is also moving with const speed = velocity
# set the speed to zero to fix BC
def set_velocity_on_cuboid(
    model: MPMSolver,
    point: list,
    size: list,
    velocity: list,
    start_time: float=0.0,
    end_time: float=999.0,
    reset: bool=False,
):
    point = torch.tensor(point, device=model.device).float()
    size = torch.tensor(size, device=model.device).float()
    velocity = torch.tensor(velocity, device=model.device).float()
    offset = model.grid_x * model.dx - point
    target = torch.all((torch.abs(offset) < size), dim=1)
    
    def collide(
        model: MPMSolver,
        target: Tensor, velocity: Tensor, 
        start_time: float, end_time: float, reset: bool
    ):
        time = model.time
        if time >= start_time and time <= end_time:
            model.grid_mv[target] = velocity
        elif reset:
            if time < end_time + 15.0 * model.dt:
                model.grid_mv[target] = 0.0

    model.post_grid_process.append(
        partial(
            collide,
            target=target,
            velocity=velocity,
            start_time=start_time,
            end_time=end_time,
            reset=reset,
        )
    )
    # TODO: add param update
    
    
# particle_v += force/particle_mass * dt
# this is applied from start_dt, ends after num_dt p2g2p's
# particle velocity is changed before p2g at each timestep
def add_impulse_on_particles(
    model: MPMSolver,
    force: list,
    dt: float,
    point: list=[1, 1, 1],
    size: list=[1, 1, 1],
    num_dt: int=1,
    start_time: float=0.0,
):
    point = torch.tensor(point, device=model.device).float()
    size = torch.tensor(size, device=model.device).float()
    force = torch.tensor(force, device=model.device).float()
    offset = model.init_pos - point
    target = torch.all((torch.abs(offset) < size), dim=1)
    end_time = start_time + num_dt * model.dt
    
    def impulse(
        model: MPMSolver,
        x: Tensor, v: Tensor,
        target: Tensor, force: Tensor, 
        start_time: float, end_time: float
    ):
        time = model.time
        if time >= start_time and time < start_time + num_dt * model.dt:
            v[target] = v[target] + force / model.p_mass * dt

    model.pre_particle_process.append(
        partial(
            impulse,
            target=target,
            force=force,
            start_time=start_time,
            end_time=end_time,
        )
    )
    
# enforce particle velocity translation where the velocity of 
# particles within the cuboid is set to predefined velocity in pre-p2g stage
def enforce_particle_velocity_translation(
    model: MPMSolver,
    point: list,
    size: list,
    velocity: list,
    start_time: float,
    end_time: float
):
    point = torch.tensor(point, device=model.device).float()
    size = torch.tensor(size, device=model.device).float()
    velocity = torch.tensor(velocity, device=model.device).float()
    offset = model.init_pos - point
    target = torch.all((torch.abs(offset) < size), dim=1)


    def enforce_velocity_translation(
        model: MPMSolver,
        x: Tensor, v: Tensor,
        target: Tensor, velocity: Tensor,
        start_time: float, end_time: float
    ):
        time = model.time
        if time >= start_time and time < end_time:
            v[target] = velocity
            
            
    model.pre_particle_process.append(
        partial(
            enforce_velocity_translation,
            target=target,
            velocity=velocity,
            start_time=start_time,
            end_time=end_time,
        )
    )
    

# define a cylinder with center point, half_height, radius, normal
# particles within the cylinder are rotating along the normal direction
# may also have a translational velocity along the normal direction
def enforce_particle_velocity_rotation(
    model: MPMSolver,
    point: list,
    normal: list,
    half_height_and_radius: list,
    rotation_scale: float,
    translation_scale: float,
    start_time: float,
    end_time: float
):
    
    point = torch.tensor(point, device=model.device).float()
    normal = torch.tensor(normal, device=model.device).float()
    normal = normal / torch.norm(normal)
    
    half_height_and_radius = torch.tensor(half_height_and_radius, device=model.device).float()
    
    horizontal_1 = torch.tensor([1.0, 1.0, 1.0], device=model.device).float()
    if torch.abs(torch.dot(normal, horizontal_1)) < 0.01:
        horizontal_1 = torch.tensor([0.72, 0.37, -0.67], device=model.device).float()
    horizontal_1 = horizontal_1 - torch.dot(horizontal_1, normal) * normal
    horizontal_1 = horizontal_1 / torch.norm(horizontal_1)
    horizontal_2 = torch.cross(horizontal_1, normal)
    
    offset = model.init_pos - point
    vertical_distance = torch.abs(torch.dot(offset, normal))
    horizontal_distance = torch.norm(offset - torch.dot(offset, normal) * normal)
    target = torch.all(
        (vertical_distance < half_height_and_radius[0]) &
        (horizontal_distance < half_height_and_radius[1]),
        dim=1
    )
    
    def enforce_velocity_rotation(
        model: MPMSolver,
        x: Tensor, v: Tensor,
        target: Tensor, point: Tensor, normal: Tensor, 
        horizontal_1: Tensor, horizontal_2: Tensor, rotation_scale: float, translation_scale: float,
        start_time: float, end_time: float
    ):
        time = model.time
        if time >= start_time and time < end_time:
            offset = x[target] - point
            horizontal_distance = torch.norm(offset - torch.dot(offset, normal) * normal)
            cosine = torch.dot(offset, horizontal_1) / horizontal_distance
            theta = torch.acos(cosine)
            if torch.dot(offset, horizontal_2) > 0:
                theta = theta
            else:
                theta = -theta
            axis1_scale = -horizontal_distance * torch.sin(theta) * rotation_scale
            axis2_scale = horizontal_distance * torch.cos(theta) * rotation_scale
            axis_vertical_scale = translation_scale
            v[target] = axis1_scale * horizontal_1 + axis2_scale * horizontal_2 + axis_vertical_scale * normal

    model.pre_particle_process.append(
        partial(
            enforce_velocity_rotation,
            target=target,
            point=point,
            normal=normal,
            horizontal_1=horizontal_1,
            horizontal_2=horizontal_2,
            rotation_scale=rotation_scale,
            translation_scale=translation_scale,
            start_time=start_time,
            end_time=end_time,
        )
    )
    
def add_surface_collider(
        model: MPMSolver,
        point: list,
        normal: list,
        surface: str="sticky",
        friction: float=0.0,
        start_time: float=0.0,
        end_time: float=999.0,
    ):
        point = torch.tensor(point, device=model.device).float()
        normal = torch.tensor(normal, device=model.device).float()
        normal = normal / torch.norm(normal)
        offset = model.grid_x * model.dx - point
        dotproduct = torch.sum(offset * normal, dim=1)
        target = dotproduct < 0.0
        
        def collide(
            model: MPMSolver,
            target: Tensor, 
            surface: str, start_time: float, end_time: float
        ):
            time = model.time
            if time >= start_time and time < end_time:
                
                if surface == "sticky":
                    model.grid_mv[target] = 0.0
                # fix it
                elif surface == "slip":
                    # set the velocity to be parallel to the surface
                    model.grid_mv[target] = model.grid_mv[target] - normal * torch.sum(model.grid_mv[target] * normal, dim=1, keepdim=True)
                elif surface == "collide":
                    # set the vertical velocity to be opposite
                    model.grid_mv[target] = model.grid_mv[target] - normal * 2.0 * torch.sum(model.grid_mv[target] * normal, dim=1, keepdim=True)
                else:
                    raise TypeError("Undefined surface type")
        model.post_grid_process.append(
            partial(
                collide,
                target=target,
                surface=surface,
                start_time=start_time,
                end_time=end_time,
            )
        )
                    
            
        
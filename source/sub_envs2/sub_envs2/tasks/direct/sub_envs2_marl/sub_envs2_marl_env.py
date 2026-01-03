# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .sub_envs2_marl_env_cfg import SubEnvs2MarlEnvCfg

from isaaclab.assets import RigidObject


class SubEnvs2MarlEnv(DirectMARLEnv):
    cfg: SubEnvs2MarlEnvCfg

    def __init__(
        self, cfg: SubEnvs2MarlEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):

        self.cube_red = RigidObject(self.cfg.red_cube_cfg)
        self.cube_green = RigidObject(self.cfg.green_cube_cfg)
        self.cube_blue = RigidObject(self.cfg.blue_cube_cfg)

        self.scene.rigid_objects["cube_red"] = self.cube_red
        self.scene.rigid_objects["cube_green"] = self.cube_green
        self.scene.rigid_objects["cube_blue"] = self.cube_blue

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)



    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        action_scale = 1.0

        current_vel_red = self.cube_red.data.root_link_vel_w
        current_vel_red[:, 2] = self.actions["cube_red"][:, 0] * action_scale
        # note: this can be applied to only specific envs if needed
        self.cube_red.write_root_link_velocity_to_sim(current_vel_red)

        current_vel_green = self.cube_green.data.root_link_vel_w
        current_vel_green[:, 2] = self.actions["cube_green"][:, 0] * action_scale
        self.cube_green.write_root_link_velocity_to_sim(current_vel_green)

        current_vel_blue = self.cube_blue.data.root_link_vel_w
        current_vel_blue[:, 2] = self.actions["cube_blue"][:, 0] * action_scale
        self.cube_blue.write_root_link_velocity_to_sim(current_vel_blue)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = {
            "cube_red": self.cube_red.data.root_pos_w,
            "cube_green": self.cube_green.data.root_pos_w,
            "cube_blue": self.cube_blue.data.root_pos_w,
        }

        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        reward = {
            "cube_red": torch.zeros(self.num_envs, device=self.device),
            "cube_green": torch.zeros(self.num_envs, device=self.device),
            "cube_blue": torch.zeros(self.num_envs, device=self.device),
        }

        return reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {
            "cube_red": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "cube_green": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "cube_blue": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
        }

        # TODO this needs to change for sub envs
        truncated = {
            "cube_red": self.episode_length_buf >= self.max_episode_length,
            "cube_green": self.episode_length_buf >= self.max_episode_length,
            "cube_blue": self.episode_length_buf >= self.max_episode_length,
        }

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cube_red._ALL_INDICES
            
        super()._reset_idx(env_ids)

        # TODO this needs to change for sub envs
        default_red_state = self.cube_red.data.default_root_state[env_ids].clone()
        default_red_state[:, :3] += self.scene.env_origins[env_ids]
        
        default_green_state = self.cube_green.data.default_root_state[env_ids].clone()
        default_green_state[:, :3] += self.scene.env_origins[env_ids]
        
        default_blue_state = self.cube_blue.data.default_root_state[env_ids].clone()
        default_blue_state[:, :3] += self.scene.env_origins[env_ids]

        self.cube_red.write_root_pose_to_sim(default_red_state[:, :7], env_ids)
        self.cube_green.write_root_pose_to_sim(default_green_state[:, :7], env_ids)
        self.cube_blue.write_root_pose_to_sim(default_blue_state[:, :7], env_ids)

        self.cube_red.write_root_velocity_to_sim(default_red_state[:, 7:], env_ids)
        self.cube_green.write_root_velocity_to_sim(default_green_state[:, 7:], env_ids)
        self.cube_blue.write_root_velocity_to_sim(default_blue_state[:, 7:], env_ids)

@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.sum(
        torch.square(pole_pos + pendulum_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(
        torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1
    )

    total_reward = {
        "cart": rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_cart_vel
        + rew_pole_vel,
        "pendulum": rew_alive + rew_termination + rew_pendulum_pos + rew_pendulum_vel,
    }
    return total_reward

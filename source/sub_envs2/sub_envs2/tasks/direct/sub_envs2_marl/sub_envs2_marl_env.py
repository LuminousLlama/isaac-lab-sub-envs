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

from sub_envs2 import subscene


class SubEnvs2MarlEnv(DirectMARLEnv):
    cfg: SubEnvs2MarlEnvCfg

    def __init__(
        self, cfg: SubEnvs2MarlEnvCfg, render_mode: str | None = None, **kwargs
    ):

        super().__init__(cfg, render_mode, **kwargs)

        self.num_sub_envs = 1
        self.temp_reset_count = 0
        self.sub_scene_episode_length_buf = torch.zeros(
            self.num_sub_envs, self.num_envs, device=self.device
        )

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

        # setup sub scenes
        self.sub_scene_0 = SubScene0(self, 0)

        self.sub_scene_0._rigid_objects["cube_red"] = self.cube_red

        self.sub_scenes_dict = {"cube_red": self.sub_scene_0}

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
            # "cube_red": self.episode_length_buf >= self.max_episode_length,
            "cube_red": self.sub_scene_episode_length_buf[0] >= self.max_episode_length,
            "cube_green": self.episode_length_buf >= self.max_episode_length,
            "cube_blue": self.episode_length_buf >= self.max_episode_length,
        }

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print("PASSED SEQUENCE", env_ids)

        test_envs = torch.tensor([0, 1], device=self.device)

        print("in _reset_idx")
        self.sub_scene_0._reset(test_envs)

        self.temp_reset_count = 0

        # TODO reset noise models
        # TODO reset events
        # TODO episode length buf

        # super()._reset_idx(env_ids)

        # TODO this needs to change for sub envs
        # default_red_state = self.cube_red.data.default_root_state[env_ids].clone()
        # default_red_state[:, :3] += self.scene.env_origins[env_ids]

        # default_green_state = self.cube_green.data.default_root_state[env_ids].clone()
        # default_green_state[:, :3] += self.scene.env_origins[env_ids]

        # default_blue_state = self.cube_blue.data.default_root_state[env_ids].clone()
        # default_blue_state[:, :3] += self.scene.env_origins[env_ids]

        # self.cube_red.write_root_pose_to_sim(default_red_state[:, :7], env_ids)
        # self.cube_green.write_root_pose_to_sim(default_green_state[:, :7], env_ids)
        # self.cube_blue.write_root_pose_to_sim(default_blue_state[:, :7], env_ids)

        # self.cube_red.write_root_velocity_to_sim(default_red_state[:, 7:], env_ids)
        # self.cube_green.write_root_velocity_to_sim(default_green_state[:, 7:], env_ids)
        # self.cube_blue.write_root_velocity_to_sim(default_blue_state[:, 7:], env_ids)

    def step(self, actions: dict[AgentID, ActionType]) -> EnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectMARLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectMARLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            actions: The actions to apply on the environment (keyed by the agent ID).
                Shape of individual tensors is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras (keyed by the agent ID).
        """

        actions = {agent: action.to(self.device) for agent, action in actions.items()}

        # add action noise
        if self.cfg.action_noise_model:
            for agent, action in actions.items():
                if agent in self._action_noise_model:
                    actions[agent] = self._action_noise_model[agent](action)
        # process actions
        self._pre_physics_step(actions)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.sub_scene_episode_length_buf[:] += 1

        self.terminated_dict, self.time_out_dict = self._get_dones()
        self.reset_buf[:] = math.prod(self.terminated_dict.values()) | math.prod(
            self.time_out_dict.values()
        )
        self.reward_dict = self._get_rewards()

        # TODO compute the reset_buf with subscene awareness
        # TODO probably make it so calling _reset_idx() crashes, resets should be handled properly with subscene reset
        # TODO probably have episode length buf of shape (sub scenes, num_envs)
        # so you can do sub_scene_epi_buf[subscene_idx, env_idx] = 0 when resetting and then you can also handle truncation gracefully
        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self._reset_idx(reset_env_ids)

        self.reset_sub_scenes()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations and the list of current agents (sorted as in possible_agents)
        self.obs_dict = self._get_observations()
        self.agents = [
            agent for agent in self.possible_agents if agent in self.obs_dict
        ]

        # add observation noise
        # note: we apply no noise to the state space (since it is used for centralized training or critic networks)
        if self.cfg.observation_noise_model:
            for agent, obs in self.obs_dict.items():
                if agent in self._observation_noise_model:
                    self.obs_dict[agent] = self._observation_noise_model[agent](obs)

        # return observations, rewards, resets and extras
        return (
            self.obs_dict,
            self.reward_dict,
            self.terminated_dict,
            self.time_out_dict,
            self.extras,
        )

    def reset_sub_scenes(self):

        # loop through all sub scenes
        for key in self.sub_scenes_dict.keys():
            reset_buf = self.terminated_dict[key] | self.time_out_dict[key]
            reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.sub_scenes_dict[key]._reset(reset_env_ids)

        # self.temp_reset_count += 1

        # if self.temp_reset_count > 300:
        #     idxs = torch.tensor([0], device=self.device)
        #     self.temp_reset_count = 0
        #     self.sub_scene_0._reset(idxs)

        # self.terminated_dict, self.time_out_dict = self._get_dones()
        # self.reset_buf[:] = math.prod(self.terminated_dict.values()) | math.prod(self.time_out_dict.values())
        # self.reward_dict = self._get_rewards()

        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self._reset_idx(reset_env_ids)


# ---------
# sub scenes
# ---------


class SubScene0(subscene.SubScene):

    def _reset(self, env_ids):
        super()._reset(env_ids)

        cube_red: RigidObject = self._rigid_objects["cube_red"]
        default_red_state = cube_red.data.default_root_state[env_ids].clone()
        default_red_state[:, :3] += self.env.scene.env_origins[env_ids]

        cube_red.write_root_pose_to_sim(default_red_state[:, :7], env_ids)
        cube_red.write_root_velocity_to_sim(default_red_state[:, 7:], env_ids)

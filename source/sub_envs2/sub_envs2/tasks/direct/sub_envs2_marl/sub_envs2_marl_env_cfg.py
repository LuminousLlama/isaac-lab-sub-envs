# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils


@configclass
class SubEnvs2MarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 2.0

    possible_agents = [
        "cube_red",
        "cube_green",
        "cube_blue",
    ]

    action_spaces = {
        "cube_red": 1,
        "cube_green": 1,
        "cube_blue": 1,
    }

    observation_spaces = {
        "cube_red": 3,
        "cube_green": 3,
        "cube_blue": 3,
    }

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10,
        env_spacing=4.0,
        replicate_physics=True,
    )

    cube_size = 0.2
    cube_mass = 1.0

    color_red = (1.0, 0.0, 0.0)
    color_green = (0.0, 1.0, 0.0)
    color_blue = (0.0, 0.0, 1.0)

    initial_pos_red = (0.0, 0.0, 1.0)
    initial_pos_green = (1.0, 0.0, 1.0)
    initial_pos_blue = (2.0, 0.0, 1.0)

    red_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedCube",
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_red),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=initial_pos_red),
    )

    green_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GreenCube",
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_green),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=initial_pos_green),
    )

    blue_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueCube",
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_blue),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=initial_pos_blue),
    )

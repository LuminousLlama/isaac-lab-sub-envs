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
from isaaclab.actuators import ImplicitActuatorCfg
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

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.5, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.7, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
        ),
    )

    side_panel_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SidePanel",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.06, 1.5, 1.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.53, 0.0, 0.55), 
        ),
    )
    
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/sub_envs2/assets/ability_robot_test.usd",
            activate_contact_sensors=True, # TODO what does this do?
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 2.0],
            joint_pos={".*": 0.0},
        ),
        actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # Applies to all joints
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
    )

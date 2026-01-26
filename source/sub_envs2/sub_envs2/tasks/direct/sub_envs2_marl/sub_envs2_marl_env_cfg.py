# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils


from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class SubEnvs2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for SubEnvs2 with robot and table."""

    # Red cube
    red_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # Green cube
    green_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.0)),
    )

    # Blue cube
    blue_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0)),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RobotTest",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/sub_envs2/assets/ability_robot_test.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-0.5, 0.0, 0.75],
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )

    # Table (static asset)
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=[1.0, 1.5, 1.0],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
        ),
    )


@configclass
class SubEnvs2MarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 2.0

    possible_agents = ["cube_red", "cube_green", "cube_blue"]
    action_spaces = {"cube_red": 1, "cube_green": 1, "cube_blue": 1}
    observation_spaces = {"cube_red": 3, "cube_green": 3, "cube_blue": 3}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene - use the custom scene config
    scene: SubEnvs2SceneCfg = SubEnvs2SceneCfg(
        num_envs=10,
        env_spacing=4.0,
        replicate_physics=True,
    )

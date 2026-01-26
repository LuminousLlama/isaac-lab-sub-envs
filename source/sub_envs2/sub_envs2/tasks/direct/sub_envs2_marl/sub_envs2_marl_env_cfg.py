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

from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg


import numpy as np

SUB_ENV_OFFSET = np.array([0, 2.0, 0.0])

ROBOT_INIT_POS = np.array([-0.5, 0.0, 0.8])

TABLE_SCALE = [1.0, 1.5, 1.0]
TABLE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd"
TABLE_INIT_POS = np.array([0.0, 0.0, 0.0])

@configclass
class SubEnvs2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for SubEnvs2 with robot and table."""

    # Spawn robots
    robot_ability: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RobotAbility",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/sub_envs2/assets/ability_robot_test.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos= (ROBOT_INIT_POS + 0 * SUB_ENV_OFFSET).tolist(),
        ),
        
        # TODO this needs to actually reflect the mimic / under actuation of ability 
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=80.0,
                damping=10.0,
            ),
        },
    )
    
    
    

    robot_shadow: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/RobotShadow",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/sub_envs2/assets/rm65_shadow_right.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ROBOT_INIT_POS + 1 * SUB_ENV_OFFSET).tolist(),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=80.0,
                damping=10.0,
            ),
        },
    )


    # TABLES
    table_ability: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableAbility",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            scale=TABLE_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos= (TABLE_INIT_POS + 0 * SUB_ENV_OFFSET).tolist(),
        ),
    )
    
    
    table_shadow: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableShadow",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            scale=TABLE_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_INIT_POS + 1 * SUB_ENV_OFFSET).tolist(),
        ),
    )


@configclass
class SubEnvs2MarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 2.0

    possible_agents = ["ability", "shadow"]
    action_spaces = {"ability": 16, "shadow": 1} # TODO SHADOW
    observation_spaces = {"ability": 3, "shadow": 3}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene - use the custom scene config
    scene: SubEnvs2SceneCfg = SubEnvs2SceneCfg(
        num_envs=10,
        env_spacing=10.0,
        replicate_physics=True,
    )
    

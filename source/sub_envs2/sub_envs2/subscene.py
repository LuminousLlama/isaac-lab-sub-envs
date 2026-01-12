

from collections.abc import Sequence
from isaaclab.scene.interactive_scene import InteractiveScene


class SubScene:
   
    def __init__(self, env, subscene_idx: int):
        self.env = env
        self.subscene_idx = subscene_idx
        
        self._articulations = dict()
        self._deformable_objects = dict()
        self._rigid_objects = dict()
        self._rigid_object_collections = dict()
        self._sensors = dict()
        self._surface_grippers = dict()
        self._extras = dict()
    
    def _reset(self, env_ids: Sequence[int]):
        # -- assets
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_ids)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.reset(env_ids)
            
        self.env.sub_scene_episode_length_buf[self.subscene_idx][env_ids] = 0


    
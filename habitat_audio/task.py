from typing import Any, Dict, List, Optional, Type, Union, Tuple

import math
import numpy as np
from gym import spaces
import torch

from habitat.config import Config
from habitat.core.dataset import Episode

from habitat.utils.visualizations import maps
from habitat.tasks.nav.nav import NavigationTask, Measure, EmbodiedTask, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)

from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.utils.visualizations import fog_of_war, maps

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_audio.dataset import ExplorationEpisode

cv2 = try_cv2_import()


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.QUERY_POSITION_IDXS = episode.query_locations
        agent_cfg.SUBGRAPH = episode.subgraph
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="Exploration")
class ExplorationTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        #TODO: below line makes agent start at random navigable point in scene. Determine whether this should be used or not.
        #episode.start_position = self._sim.pathfinder.get_random_navigable_point().tolist()
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return self._sim._is_episode_active
    
    def reset(self, episode: Episode):
        observations  = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        self._is_episode_active = True

        return observations


@registry.register_task_action
class PauseAction(SimulatorTaskAction):
    name: str = "PAUSE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.PAUSE)


@registry.register_sensor
class BinSpectMagSensor(Sensor):
    r"""Binaural spectrogram magnitude at the current pose
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "bin_spect_mag"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode: Episode, *args: Any, **kwargs: Any):
        return self._sim.get_current_bin_spec_mag()


@registry.register_sensor(name="ActivePoseSensor")
class ActivePoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. 
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)
    
    def _compute_rotation_from_azimuth(self, azimuth):
        """
        compute rotation angle from azimuth angle
        :param azimuth: azimuth angle
        :return: rotation angle
        """
        # rotation is calculated in the habitat coordinate frame counter-clocwise so -Z is 0 and -X is -90
        return -(azimuth + 0) % 360

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        #TODO: If there is an issue with pose orientation, try uncommenting line 195 and 200 below.
        ref_position_xyz = np.array(episode.start_position, dtype=np.float32)
        rotation_world_ref = quaternion_from_coeff(episode.start_rotation)
        #rotation_world_ref = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(episode.start_rotation)),
        #                                        np.array([0, 1, 0]))

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation
        #rotation_world_agent = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(agent_state.rotation)),
        #                                            np.array([0, 1, 0]))

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_ref.inverse(), agent_position_xyz - ref_position_xyz
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_ref
        )

        return np.array(
            [round(-agent_position_xyz[2]), round(agent_position_xyz[0]), round(-agent_position_xyz[0]), round(-agent_position_xyz[2]), agent_heading[0],],
            dtype=np.float32
        )
        
@registry.register_sensor(name="RelativePoseSensor")
class RelativePoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. 
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._current_episode_id = episode_uniq_id

        reference_pose = self._sim.get_reference_pose()
        current_pose = [self._sim.get_receiver_position_idx(), self._sim.get_receiver_position_idx(), self._sim.azimuth_angle]
        relative_pose = self._sim.compute_relative_pose(current_pose, reference_pose)

        return relative_pose

@registry.register_sensor(name="GlobalMap")
class GlobalMap(Sensor):
    r"""Global occupancy map sensor. Placeholder sensor to be populated in ppo_trainer 
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the global map size
    Attributes:
        _dimensionality: number of dimensions used to specify the global map size
    """
    cls_uuid: str = "occupancy_map"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._current_episode_id = episode_uniq_id
            
        return np.zeros(self.config.FEATURE_SHAPE)


@registry.register_sensor(name="TimestepSensor")
class TimestepSensor(Sensor):
    r"""Sensor for number of maximum number of remaining timesteps in episode.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: 
    Attributes:
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "timestep_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode: Episode, *args: Any, **kwargs: Any
    ):
        return self._sim.get_remaining_timesteps()

@registry.register_sensor(name="ContextLengthSensor")
class ContextLengthSensor(Sensor):
    r"""Sensor for number of remaining observation slots in context buffer.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: 
    Attributes:
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "context_length_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode: Episode, *args: Any, **kwargs: Any
    ):
        return self._sim.get_remaining_observations()

@registry.register_measure
class AcousticMapError(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "acoustic_map_error"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):

        self._sim = sim
        self._config = config
        self._query_positions = None
        self._gt_rirs = None
        self.context_observations = []

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None
        self._query_positions = episode.query_locations
        self._gt_rirs = self._sim.generate_gt_RIRs(self._query_positions)
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(
        self, episode: ExplorationEpisode, *args: Any, **kwargs: Any
    ):
        #properly format observations/context and query_positions
        #pass observations to model, along with query_positions
        #get predicted RIRs at those positions
        #compute RIR error.

        self._metric = 0
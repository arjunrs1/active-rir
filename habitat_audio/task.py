from typing import Any, Dict, List, Optional, Type, Union, Tuple

import math
import numpy as np
from gym import spaces

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
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="Exploration")
class ExplorationTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
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

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        return np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], -agent_position_xyz[2], agent_position_xyz[0], agent_heading[0],],
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

@registry.register_sensor(name="GTEgoMap")
class GTEgoMap(Sensor):
    def __init__(self, config, *args, **kwargs):
        # Map statistics
        self.map_size = MAP_SIZE
        self.map_scale = MAP_SCALE
        if MAX_SENSOR_RANGE > 0:
            self.max_forward_range = MAX_SENSOR_RANGE
        else:
            self.max_forward_range = self.map_size * self.map_scale
        # Agent height for pointcloud tranforms
        self.camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        # Compute intrinsic matrix
        depth_H = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        depth_W = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        hfov = float(
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180
        vfov = 2 * np.arctan((depth_H / depth_W) * np.tan(hfov / 2.0))
        self.intrinsic_matrix = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1, 0],
                [0.0, 0.0, 0, 1],
            ]
        )
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        # Height thresholds for obstacles
        self.height_thresh = HEIGHT_THRESH
        # Depth processing
        self.min_depth = float(
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)
        # Pre-compute a grid of locations for depth projection
        W = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        H = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
            np.linspace(-1, 1, W), np.linspace(1, -1, H)
        )
        self.config = config
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (MAP_SIZE, MAP_SIZE, 2)
        return spaces.Box(low=0, high=1, shape=sensor_shape, dtype=np.uint8,)
    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """
        depth_float = depth.astype(np.float32)[..., 0]
        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        valid_depths = (depth_float != self.min_depth) & (
            depth_float <= self.max_forward_range
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.inverse_intrinsic_matrix
        # XYZ in the camera coordinate system
        xyz_camera = np.matmul(inv_K, xys).T
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]
        return xyz_camera
    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value
    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """
        if self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * \
                (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth
        XYZ_ego = self.convert_to_pointcloud(depth)
        # Adding agent's height to the pointcloud
        XYZ_ego[:, 1] += self.camera_height
        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2
        points = XYZ_ego
        grid_x = (points[:, 0] / self.map_scale) + Vby2
        grid_y = (points[:, 2] / self.map_scale) + V
        # Filter out invalid points
        valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
        )
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)
        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)
        self.safe_assign(
            obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)
        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(
            explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)
        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)
        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)
        return np.stack([obstacle_mat, explore_mat], axis=2)
    
    def get_observation(self, *args: Any, observation, **kwargs: Any):
        # observation["depth"] = observation["depth"].reshape(1, 128, 128)
        sim_depth = asnumpy(observation["depth"])
        ego_map_gt = self._get_depth_projection(sim_depth)
        return ego_map_gt
from typing import Any, List, Optional
from abc import ABC
from collections import defaultdict, namedtuple
import logging
import time
import pickle
import os
import json

import librosa
import psutil
import scipy
import torch
import numpy as np
import networkx as nx
from scipy.io import wavfile
from scipy.signal import fftconvolve
import scipy
from gym import spaces

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim, HabitatSimSensor, overwrite_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (Config, AgentState, Observations, ShortestPathPoint, SensorSuite, Simulator)
from habitat_audio.utils import load_points_data, _to_tensor
from soundspaces.utils import load_metadata
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from soundspaces.mp3d_utils import HouseReader

def calculate_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0 / 1024 / 1024  # in GB


def crossfade(x1, x2, sr):
    crossfade_samples = int(0.05 * sr)  # 30 ms
    x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
    x1_weight = np.flip(x2_weight)
    x3 = [x1[:, :crossfade_samples+1] * x1_weight + x2[:, :crossfade_samples+1] * x2_weight, x2[:, crossfade_samples+1:]]

    return np.concatenate(x3, axis=1)

class DummySimulator:
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class HabitatSimAudioEnabledTrain(HabitatSim):
    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        r"""Changes made to simulator wrapper over habitat-sim

        This simulator allows the agent to be moved to location specified in the
        Args:
            config: configuration for initializing the simulator.
        """
        super().__init__(config)

        self.config_yaml = config
        assert self.config_yaml.SCENE_DATASET in ["mp3d"], "SCENE_DATASET needs to be in ['mp3d']"
        self.temp_scene_dataset = self.config_yaml.SCENE_DATASET
        self._receiver_position_index = None
        self._rotation_angle = None
        self._frame_cache = defaultdict(dict)
        self._is_episode_active = None
        self._previous_step_collided = None
        self._position_to_index_mapping = dict()
        self.points, self.graph = load_points_data(self.meta_dir, self.config_yaml.AUDIO.GRAPH_FILE,
                                                   scene_dataset=self.temp_scene_dataset)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        logging.info('Current scene: {}'.format(self.current_scene_name,))

        if self.config_yaml.USE_RENDERED_OBSERVATIONS:
            if hasattr(self, '_sim'):
                self._sim.close()
                del self._sim
            self._sim = DummySimulator()
            logging.info('Loaded the rendered observations for all scenes')
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
        else:
            self._sim = habitat_sim.Simulator(config=self.sim_config)
            self.add_acoustic_config()
            self.material_configured = False

    def add_acoustic_config(self):
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = False
        audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        audio_sensor_spec.channelLayout.channelCount = 2
        audio_sensor_spec.acousticsConfig.sampleRate = self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        audio_sensor_spec.acousticsConfig.threadCount = 1
        audio_sensor_spec.acousticsConfig.indirectRayCount = 500
        audio_sensor_spec.acousticsConfig.temporalCoherence = True
        audio_sensor_spec.acousticsConfig.transmission = True
        self._sim.add_sensor(audio_sensor_spec)           

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        r"""
        get current agent state
        :param agent_id: agent ID
        :return: agent state
        """
        if not self.config_yaml.USE_RENDERED_OBSERVATIONS:
            agent_state = super().get_agent_state(agent_id)
        else:
            agent_state = self._sim.get_agent_state()

        return agent_state

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""
        set agent's state when not using pre-rendered observations
        :param position: 3D position of the agent
        :param rotation: rotation angle of the agent
        :param agent_id: agent ID
        :param reset_sensors: reset sensors or not
        :return: None
        """
        if not self.config_yaml.USE_RENDERED_OBSERVATIONS:
            super().set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=reset_sensors)
        else:
            self._sim.set_agent_state(position, rotation)
    
    @property
    def current_scene_observation_file(self):
        r"""
        get path to pre-rendered observations for the current scene
        :return: path to pre-rendered observations for the current scene
        """
        return os.path.join(self.config_yaml.RENDERED_OBSERVATIONS, self.temp_scene_dataset,
                            self.current_scene_name + '.pkl')

    @property
    def meta_dir(self):
        r"""
        get path to meta-dir containing data about location of navigation nodes and their connectivity
        :return: path to meta-dir containing data about location of navigation nodes and their connectivity
        """
        return os.path.join(self.config_yaml.AUDIO.META_DIR, self.current_scene_name)

    @property
    def current_scene_name(self):
        r"""
        get current scene name
        :return: current scene name
        """
        if self.temp_scene_dataset == "mp3d":
            return self._current_scene.split('/')[-2]
        elif self.temp_scene_dataset == "replica":
            return self._current_scene.split('/')[-3]
        else:
            raise ValueError

    def reconfigure(self, config: Config) -> None:
        r"""
        reconfigure for new episode
        :param config: config for reconfiguration
        :return: None
        """
        self.config = config
        self.config_yaml = config
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {}'.format(self.current_scene_name))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self.add_acoustic_config()
                self.material_configured = False
                self._update_agents_state()
            else:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            logging.info('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                       scene_dataset=self.temp_scene_dataset)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
        self._episode_step_count = 0

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        r"""
        get current observation from simulator
        :return: current observation
        """
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            assert not self.config.USE_RENDERED_OBSERVATIONS
            sim_obs = self._sim.get_sensor_observations()
            for sensor in sim_obs:
                sim_obs[sensor] = sim_obs[sensor]
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        r"""
        reset simulator for new episode
        :return: None
        """
        logging.debug('Reset simulation')

        if not self.config_yaml.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False

        # PAUSE: 0, FORWARD: 1, LEFT: 2, RIGHT: 3
        if action == HabitatSimActions.MOVE_FORWARD:
            self._previous_step_collided = True
            # the agent initially faces -Z by default
            for neighbor in self.graph[self._receiver_position_index]:
                p1 = self.graph.nodes[self._receiver_position_index]['point']
                p2 = self.graph.nodes[neighbor]['point']
                direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                if direction not in [0, 90, 180, 270]:
                    # diagonal connection
                    if int(abs(direction - self.get_orientation())) == 45:
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
                elif direction == self.get_orientation():
                    self._receiver_position_index = neighbor
                    self._previous_step_collided = False
                    break
        elif action == HabitatSimActions.TURN_LEFT:
            # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
            self._rotation_angle = (self._rotation_angle + 90) % 360
        elif action == HabitatSimActions.TURN_RIGHT:
            self._rotation_angle = (self._rotation_angle - 90) % 360
        elif action == HabitatSimActions.PAUSE:
            raise ValueError
            pass
        else:
            raise NotImplementedError(str(action) + " not in action space -- [PAUSE: 0, MOVE_FORWARD: 1, TURN_LEFT: 2,"
                                                    "TURN_RIGHT: 3]")

        if not self.config_yaml.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))

        # log debugging info
        logging.debug('After taking action {}, r: {}, orientation: {}, location: {}'.format(
            action, self._receiver_position_index, self.get_orientation(),
            self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        if self.config_yaml.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def get_orientation(self):
        r"""
        get current orientation of the agent
        :return: current orientation of the agent
        """
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    def write_info_to_obs(self, observations):
        r"""
        write agent location and orientation info, and scene name to observation dict... probably
        redundant
        :param observations: observation dict containing different info about the current observation
        :return: None
        """
        observations["agent node and location"] = (self._receiver_position_index,
                                                   self.graph.nodes[self._receiver_position_index]["point"])
        observations["scene name"] = self.current_scene_name
        observations["orientation"] = self._rotation_angle

    @property
    def azimuth_angle(self):
        r"""
        get current azimuth of the agent
        :return: current azimuth of the agent
        """
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    def geodesic_distance(self, position_a, position_b):
        r"""
        get geodesic distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: geodesic distance between 2 nodes
        """
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None
        steps = nx.shortest_path_length(self.graph, index_a, index_b) * self.config_yaml.GRID_SIZE
        return steps

    def euclidean_distance(self, position_a, position_b):
        r"""
        get euclidean distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: euclidean distance between 2 nodes
        """
        assert len(position_a) == len(position_b) == 3
        assert position_a[1] == position_b[1], "height should be same for node a and b"
        return np.power(np.power(position_a[0] - position_b[0],  2) + np.power(position_a[2] - position_b[2], 2), 0.5)

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def get_current_bin_spec_mag(self):
        raise NotImplementedError

    def get_current_spatial_mono_spec_mag(self):
        raise NotImplementedError

@registry.register_simulator()
class SoundSpacesTeleportSim(HabitatSimAudioEnabledTrain):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self._get_queries_and_RIRs = True
        self.query_poses = None
        self.gt_rirs_mags = None
        self.gt_rirs_phases = None



    def reset(self):
        r"""
        reset simulator for new episode
        :return: None
        """
        logging.debug('Reset simulation')

        if not self.config_yaml.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._previous_step_collided = False
        self._prev_sim_obs = sim_obs
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)


    
        if self._get_queries_and_RIRs:
            query_pose_idxs = self.config_yaml.AGENT_0.QUERY_POSITION_IDXS
            self.gt_rirs_mags, self.gt_rirs_phases = self.generate_gt_RIRs(query_pose_idxs)
            self.query_poses = []
            for query in query_pose_idxs:
                pose = np.array(self.compute_relative_pose(current_pose=query, ref_pose=self.reference_pose)).astype("float32")
                self.query_poses.append(pose)

        return observations
    
    def get_RIR_reward_queries_RIRS(self):
        assert self.query_poses is not None
        assert self.gt_rirs_mags is not None
        assert self.gt_rirs_phases is not None
        return self.query_poses, np.array(self.gt_rirs_mags), np.array(self.gt_rirs_phases)
    
    def convert_external_pose_to_relative(self, external_pose):
        assert self.reference_pose is not None
        return np.array(self.compute_relative_pose(current_pose=external_pose, ref_pose=self.reference_pose)).astype("float32") 

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._previous_step_collided = False
        if self._episode_step_count == self.config_yaml.MAX_EPISODE_STEPS:
            self._is_episode_active = False
        else:
            prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle

            #perform action in simulator
            if action == HabitatSimActions.MOVE_FORWARD:
                p1 = self.graph.nodes[prev_position_index]['point']
                self._previous_step_collided = True
                for neighbor in self.graph[prev_position_index]:
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._source_position_index = self._receiver_position_index
                        self._previous_step_collided = False
                        break

            if action == HabitatSimActions.TURN_LEFT:
                self._rotation_angle = (prev_rotation_angle + 90) % 360

            if action == HabitatSimActions.TURN_RIGHT:
                self._rotation_angle = (prev_rotation_angle - 90) % 360

            if self.config_yaml.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config_yaml.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)

            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
            
        #TO DO: change to sample budget, where there are 6 actions: FORWARD_SAMPLE, RIGHT_SAMPLE, LEFT_SAMPLE, FORWARD, RIGHT, LEFT
        #Only the actions with SAMPLE will add to the episode step count.
        self._episode_step_count += 1

        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()

        if self.config_yaml.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if self.config_yaml.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations
        
        return observations
    
    def reconfigure(self, config: Config) -> None:
        self.config = config
        self.config_yaml = config
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)
        self._audio_index = 0
        #is_same_sound = config.AGENT_0.SOUND_ID == self._current_sound
        #if not is_same_sound:
        #    self._current_sound = self.config.AGENT_0.SOUND_ID
        #    self._load_single_source_sound()
        #    logging.debug("Switch to sound {} with duration {} seconds".format(self._current_sound, self._duration))
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {}'.format(self.current_scene_name))

            if self.config.USE_RENDERED_OBSERVATIONS:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            else:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                if not self.config.USE_RENDERED_OBSERVATIONS:
                    self.add_acoustic_config()
                    self.material_configured = False
                self._update_agents_state()
                self._frame_cache = dict()
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_points_data(self.meta_dir, self.config_yaml.AUDIO.GRAPH_FILE,
                                                       scene_dataset=self.temp_scene_dataset)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
            self._instance2label_mapping = None

        if not self.config.USE_RENDERED_OBSERVATIONS:
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
            audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.START_POSITION) + np.array([0, 1.5, 0]))
            if not self.material_configured:
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
                self.material_configured = True

        if not is_same_scene:
            self._audiogoal_cache = dict()
            self._spectrogram_cache = dict()

        self._episode_step_count = 0

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))
        else:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)

        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            self._distractor_position_index = self.config.AGENT_0.DISTRACTOR_POSITION_INDEX
            self._current_distractor_sound = self.config.AGENT_0.DISTRACTOR_SOUND_ID
            self._load_single_distractor_sound()

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation()))
        
    def generate_gt_RIRs(self, query_pose_idxs):
        assert query_pose_idxs is not None
        gt_rirs_mags = []
        gt_rirs_phases = []

        for query in query_pose_idxs:
            gt_rir_mag, gt_rir_phase = self.compute_RIR(query)
            gt_rirs_mags.append(gt_rir_mag)
            gt_rirs_phases.append(gt_rir_phase)

        return gt_rirs_mags, gt_rirs_phases

    def compute_RIR(self, query_location):
        receiver_location, source_location, azimuth_angle = query_location
        sampling_rate = self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        if self.config_yaml.USE_RENDERED_OBSERVATIONS:
            binaural_rir_file = os.path.join(self.binaural_rir_dir(), str(azimuth_angle), '{}_{}.wav'.format(
                receiver_location, source_location))
            try:
                sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
            except ValueError:
                logging.warning("{} file is not readable".format(binaural_rir_file))
                binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
                sampling_freq = sampling_rate
            if len(binaural_rir) == 0:
                logging.debug("Empty RIR file at {}".format(binaural_rir_file))
                binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
                sampling_freq = sampling_rate
        else:
            binaural_rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))

        binaural_rir_full_length = np.zeros((sampling_rate, 2))
        if binaural_rir.shape[0] > 128:
            binaural_rir_full_length[: min(binaural_rir.shape[0], sampling_rate) - 128, :] =\
                binaural_rir[128: min(binaural_rir.shape[0], sampling_rate), :]    # remove the first 127 zero samples
        binaural_rir = binaural_rir_full_length
        assert sampling_freq == sampling_rate

        binaural_rir = binaural_rir.T
        fft_windows_l_imp = librosa.stft(np.asfortranarray(binaural_rir[0]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_l_imp, phase_l_imp = librosa.magphase(fft_windows_l_imp)
        phase_l_imp = np.angle(phase_l_imp)

        fft_windows_r_imp = librosa.stft(np.asfortranarray(binaural_rir[1]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_r_imp, phase_r_imp = librosa.magphase(fft_windows_r_imp)
        phase_r_imp = np.angle(phase_r_imp)

        magnitude_imp = np.stack([magnitude_l_imp, magnitude_r_imp], axis=-1)
        phase_imp = np.stack([phase_l_imp, phase_r_imp], axis=-1)

        magnitude_imp = magnitude_imp.astype("float32")
        phase_imp = phase_imp.astype("float32")

        return magnitude_imp, phase_imp
    
    def metric_distance(self, position_a, position_bs):
        a = self.graph.nodes[position_a]['point']
        bs = self.graph.nodes[position_bs]['point']
        return np.linalg.norm(np.array(a)-np.array(bs))
        
    def get_current_bin_spec_mag(self):
        """
        compute IR spectrogram (spect.)
        :param azimuth: pose azimuth angle
        :param receiver_node: pose receiver node
        :return: spect.
        """
        rir_az_dir = os.path.join(self.binaural_rir_dir(), str(self.azimuth_angle()))
        src_rec_dir = '{}_{}.wav'.format(self._receiver_position_index, self._receiver_position_index)
        binaural_rir_file = os.path.join(rir_az_dir, src_rec_dir)

        assert os.path.isfile(binaural_rir_file)

        try:
            fs_imp, sig_imp = wavfile.read(binaural_rir_file)

            assert fs_imp == self.config_yaml.AUDIO.RIR_SAMPLING_RATE, "RIR doesn't have sampling frequency of rir_sampling_rate"
        except ValueError:
            sig_imp = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2)).astype("float32")
            fs_imp = self.config_yaml.AUDIO.RIR_SAMPLING_RATE

        if len(sig_imp) == 0:
            sig_imp = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2)).astype("float32")
            fs_imp = self.config_yaml.AUDIO.RIR_SAMPLING_RATE

        imp_full_length = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2))

        if sig_imp.shape[0] > 128:
            imp_full_length[: min(sig_imp.shape[0], self.config_yaml.AUDIO.RIR_SAMPLING_RATE) - 128, :] =\
                sig_imp[128: min(sig_imp.shape[0], self.config_yaml.AUDIO.RIR_SAMPLING_RATE), :]    # remove the first 127 zero samples
        
        sig_imp = imp_full_length
        assert fs_imp == self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        sig_imp = sig_imp.T
        fft_windows_l_imp = librosa.stft(np.asfortranarray(sig_imp[0]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_l_imp, _ = librosa.magphase(fft_windows_l_imp)
        fft_windows_r_imp = librosa.stft(np.asfortranarray(sig_imp[1]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_r_imp, _ = librosa.magphase(fft_windows_r_imp)
        magnitude_imp = np.stack([magnitude_l_imp, magnitude_r_imp], axis=-1)
        magnitude_imp = magnitude_imp.astype("float32")

        return magnitude_imp        

    def binaural_rir_dir(self):
        return os.path.join(self.config_yaml.AUDIO.RIR_DIR, self.current_scene_name)
    
    def get_receiver_position_idx(self):
        #for use in RelativePoseSensor
        return self._receiver_position_index
    
    def azimuth_angle(self):
        r"""
        get current azimuth of the agent
        :return: current azimuth of the agent
        """
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    def _compute_rotation_from_azimuth(self, azimuth):
        """
        compute rotation angle from azimuth angle
        :param azimuth: azimuth angle
        :return: rotation angle
        """
        # rotation is calculated in the habitat coordinate frame counter-clocwise so -Z is 0 and -X is -90
        return -(azimuth + 0) % 360


@registry.register_simulator()
class SoundSpacesSimActiveRIR(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound = None
        self._offset = None
        self._duration = None
        self._audio_index = None
        self._audio_length = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None
        self._frame_cache = dict()
        self._audiogoal_cache = dict()
        self._spectrogram_cache = dict()
        self._egomap_cache = defaultdict(dict)
        self._scene_observations = None
        self._episode_step_count = None
        self._collected_context = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False
        self._instance2label_mapping = None
        self._house_readers = dict()
        self._use_oracle_planner = True
        self._oracle_actions = list()
        self.scene_to_query_idx_mapping = {}
        self.split = self.config.SPLIT
        self.max_query_length = self.config.MAX_QUERY_LENGTH

        self.points, self.graph = load_metadata(self.metadata_dir)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            self._distractor_position_index = None
            self._current_distractor_sound = None

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim = DummySimulator()
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
        else:
            self._sim = habitat_sim.Simulator(config=self.sim_config)
            self.add_acoustic_config()
            self._last_rir = None
            self._current_sample_index = 0

        self.config_yaml = config
        assert self.config_yaml.SCENE_DATASET in ["mp3d"], "SCENE_DATASET needs to be in ['mp3d']"
        self.temp_scene_dataset = self.config_yaml.SCENE_DATASET

        logging.info('Current scene: {}'.format(self.current_scene_name,))

        self._get_queries_and_RIRs = True
        self.query_poses = None
        self.gt_rirs_mags = None
        self.gt_rirs_phases = None

        #self.split = "seen_eval" # only use if debugging with that one scene

        if self.split == "seen_eval":
            arbitrary_rir_seen_env_eval_query_pose_idxs_path = self.config.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
            assert arbitrary_rir_seen_env_eval_query_pose_idxs_path is not None
            assert os.path.isfile(arbitrary_rir_seen_env_eval_query_pose_idxs_path)
            with open(arbitrary_rir_seen_env_eval_query_pose_idxs_path, "rb") as fi:
                self.arbitrary_rir_query_pose_idxs_from_disk = pickle.load(fi)

            arbitrary_rir_seen_env_eval_scene_names_path = self.config.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH
            assert arbitrary_rir_seen_env_eval_scene_names_path is not None
            assert os.path.isfile(arbitrary_rir_seen_env_eval_scene_names_path)
            with open(arbitrary_rir_seen_env_eval_scene_names_path, "rb") as fi:
                self.arbitrary_rir_scene_names_from_disk = pickle.load(fi)

            arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path = self.config.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
            assert arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path is not None
            assert os.path.isfile(arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path)
            with open(arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path, "rb") as fi:
                self.arbitrary_rir_query_pose_subgraph_idxs_from_disk = pickle.load(fi)
        elif self.split == "unseen_eval":
            arbitrary_rir_unseen_env_eval_query_pose_idxs_path = self.config.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
            assert arbitrary_rir_unseen_env_eval_query_pose_idxs_path is not None
            assert os.path.isfile(arbitrary_rir_unseen_env_eval_query_pose_idxs_path)
            with open(arbitrary_rir_unseen_env_eval_query_pose_idxs_path, "rb") as fi:
                self.arbitrary_rir_query_pose_idxs_from_disk = pickle.load(fi)

            arbitrary_rir_unseen_env_val_scene_names_path = self.config.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH
            assert arbitrary_rir_unseen_env_val_scene_names_path is not None
            assert os.path.isfile(arbitrary_rir_unseen_env_val_scene_names_path)
            with open(arbitrary_rir_unseen_env_val_scene_names_path, "rb") as fi:
                self.arbitrary_rir_scene_names_from_disk = pickle.load(fi)

            arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path = self.config.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
            assert arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path is not None
            assert os.path.isfile(arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path)
            with open(arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path, "rb") as fi:
                self.arbitrary_rir_query_pose_subgraph_idxs_from_disk = pickle.load(fi)

        self._arr_query_poses_per_scene = None
        if self.split == "train":
            assert os.path.isfile(self.config.AUDIO.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH)
            with open(self.config.AUDIO.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)
        elif self.split == "seen_eval":
            assert os.path.isfile(self.config.AUDIO.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH)
            with open(self.config.AUDIO.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)
        elif self.split == "unseen_eval":
            assert os.path.isfile(self.config.AUDIO.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH)
            with open(self.config.AUDIO.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)

    def add_acoustic_config(self):
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = False
        audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        audio_sensor_spec.channelLayout.channelCount = 2
        audio_sensor_spec.acousticsConfig.sampleRate = self.config.AUDIO.RIR_SAMPLING_RATE
        audio_sensor_spec.acousticsConfig.threadCount = 1
        audio_sensor_spec.acousticsConfig.indirectRayCount = 500
        audio_sensor_spec.acousticsConfig.temporalCoherence = True
        audio_sensor_spec.acousticsConfig.transmission = True
        self._sim.add_sensor(audio_sensor_spec)

    def compute_relative_pose(self, current_pose=None, ref_pose=None):
        """
        compute relative pose
        :param current_pose: current pose
        :param ref_pose: reference pose
        :param scene_graph: scene graph
        :return: relative pose
        """
        assert isinstance(current_pose, list)
        assert isinstance(ref_pose, list)
        assert len(ref_pose) == 2
        assert len(current_pose) == 3

        if not isinstance(ref_pose[0], list):
            ref_position_xyz = np.array(list(self.graph.nodes[ref_pose[0]]["point"]), dtype=np.float32)
        else:
            ref_position_xyz = np.array(list(ref_pose[0]), dtype=np.float32)
        rotation_world_ref = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(ref_pose[1])),
                                                  np.array([0, 1, 0]))

        agent_position_xyz = np.array(list(self.graph.nodes[current_pose[0]]["point"]), dtype=np.float32)
        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_ref.inverse(), agent_position_xyz - ref_position_xyz
        )

        audio_source_position_xyz = np.array(list(self.graph.nodes[current_pose[1]]["point"]), dtype=np.float32)
        audio_source_position_xyz = audio_source_position_xyz - ref_position_xyz

        rotation_world_agent = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(current_pose[2])),
                                                    np.array([0, 1, 0]))
        # next 2 lines compute relative rotation in the counter-clockwise direction, i.e. -z to -x
        # rotation_world_agent.inverse() * rotation_world_ref = rotation_world_agent - rotation_world_ref
        heading_vector = quaternion_rotate_vector(rotation_world_agent.inverse() * rotation_world_ref,
                                                  np.array([0, 0, -1]))
        agent_heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        """ return [agent_position_xyz[0], -agent_position_xyz[2], audio_source_position_xyz[0],
                -audio_source_position_xyz[2], agent_heading] """
        
        return [round(-agent_position_xyz[2]), round(agent_position_xyz[0]), round(-audio_source_position_xyz[2]),
                round(audio_source_position_xyz[0]), agent_heading]

    def get_reference_pose(self):
        #for use in RelativePoseSensor
        return [self.config.AGENT_0.START_POSITION, self.azimuth_angle]
 
    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "offset",
                "duration",
                "sound_id",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
                "distractor_sound_id",
                "distractor_position_index",
                "query_position_idxs",
                "subgraph"
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        if self.config.USE_RENDERED_OBSERVATIONS:
            return self._sim.get_agent_state()
        else:
            return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(position, rotation)
        else:
            agent = self._sim.get_agent(agent_id)
            new_state = self.get_agent_state(agent_id)
            new_state.position = position
            new_state.rotation = rotation

            # NB: The agent state also contains the sensor states in _absolute_
            # coordinates. In order to set the agent's body to a specific
            # location and have the sensors follow, we must not provide any
            # state for the sensors. This will cause them to follow the agent's
            # body
            new_state.sensor_states = {}
            agent.set_state(new_state, reset_sensors)
            return True

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.RIR_DIR, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def distractor_sound_dir(self):
        return self.config.AUDIO.DISTRACTOR_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.META_DIR, self.current_scene_name)

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

    @property
    def is_silent(self):
        return self._episode_step_count > self._duration

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        self.config = config
        self.config_yaml = config
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)
        self._audio_index = 0

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self.is_same_scene = False
            self._current_scene = config.SCENE
            logging.debug('Current scene: {}'.format(self.current_scene_name))
            
            if self.config.USE_RENDERED_OBSERVATIONS:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            else:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self.add_acoustic_config()
                audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
            self._instance2label_mapping = None
        else:
            self.is_same_scene = True

        self._update_agents_state()
        audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
        if not self.config.USE_RENDERED_OBSERVATIONS:
            #TO DO: modify to set audio source transform after moving to a new location!!
            audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.START_POSITION) + np.array([0, 1.5, 0]))

        if not is_same_scene:
            self._audiogoal_cache = dict()
            self._spectrogram_cache = dict()

        self._episode_step_count = 0
        self._collected_context = 0
        self._last_rir = None
        self._current_sample_index = np.random.randint(self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME)

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))
        else:
            self.set_agent_state(list(np.array(self.config.AGENT_0.START_POSITION) + np.array([0, 1.5, 0])),
                                 self.config.AGENT_0.START_ROTATION)

        """ logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation())) """

        self.subgraph_index = self.config.AGENT_0.SUBGRAPH
        #self.reference_pose = [self._receiver_position_index, self.azimuth_angle]

    def generate_gt_RIRs(self, query_pose_idxs):
        assert query_pose_idxs is not None
        gt_rirs_mags = []
        gt_rirs_phases = []

        for query in query_pose_idxs:
            gt_rir_mag, gt_rir_phase = self.compute_RIR(query)
            gt_rirs_mags.append(gt_rir_mag)
            gt_rirs_phases.append(gt_rir_phase)

        return gt_rirs_mags, gt_rirs_phases

    def compute_RIR(self, query_location):
        receiver_location, source_location, azimuth_angle = query_location
        sampling_rate = self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        #TODO: Fix this so that we are using SS 2.0 query RIRs (treat setting audio source/receiver transform carefully)
        #if self.config_yaml.USE_RENDERED_OBSERVATIONS:
        binaural_rir_file = os.path.join(self.binaural_rir_dir, str(azimuth_angle), '{}_{}.wav'.format(
            receiver_location, source_location))
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
        except ValueError:
            logging.warning("{} file is not readable".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
            sampling_freq = sampling_rate
        if len(binaural_rir) == 0:
            logging.debug("Empty RIR file at {}".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
            sampling_freq = sampling_rate
        #else:
            #NOTE: If you generate continuous sim query RIRs, then make sure to reset listener and source transform in reconfigure AFTER computing query RIRs.
            """ rec_pos = list(self.graph.nodes[receiver_location]['point'])
            rot = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(azimuth_angle)),
                                                    np.array([0, 1, 0]))
            src_pos = list(self.graph.nodes[source_location]['point'])
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
            audio_sensor.setAudioSourceTransform(np.array(src_pos) + np.array([0, 1.5, 0]))
            audio_sensor.setAudioListenerTransform(
                        np.array(rec_pos),
                        np.array([rot.w, rot.x, rot.y, rot.z]))
            """
            #binaural_rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))

        binaural_rir_full_length = np.zeros((sampling_rate, 2))
        if binaural_rir.shape[0] > 128:
            binaural_rir_full_length[: min(binaural_rir.shape[0], sampling_rate) - 128, :] =\
                binaural_rir[128: min(binaural_rir.shape[0], sampling_rate), :]    # remove the first 127 zero samples
        binaural_rir = binaural_rir_full_length
        #assert sampling_freq == sampling_rate

        binaural_rir = binaural_rir.T
        fft_windows_l_imp = librosa.stft(np.asfortranarray(binaural_rir[0]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_l_imp, phase_l_imp = librosa.magphase(fft_windows_l_imp)
        phase_l_imp = np.angle(phase_l_imp)

        fft_windows_r_imp = librosa.stft(np.asfortranarray(binaural_rir[1]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_r_imp, phase_r_imp = librosa.magphase(fft_windows_r_imp)
        phase_r_imp = np.angle(phase_r_imp)

        magnitude_imp = np.stack([magnitude_l_imp, magnitude_r_imp], axis=-1)
        phase_imp = np.stack([phase_l_imp, phase_r_imp], axis=-1)

        magnitude_imp = magnitude_imp.astype("float32")
        phase_imp = phase_imp.astype("float32")

        return magnitude_imp, phase_imp

    def get_current_bin_spec_mag(self):
        """
        compute IR spectrogram (spect.)
        :param azimuth: pose azimuth angle
        :param receiver_node: pose receiver node
        :return: spect.
        """
        if self.config.USE_RENDERED_OBSERVATIONS:
            rir_az_dir = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle))
            src_rec_dir = '{}_{}.wav'.format(self._receiver_position_index, self._receiver_position_index)
            binaural_rir_file = os.path.join(rir_az_dir, src_rec_dir)

            assert os.path.isfile(binaural_rir_file)

            try:
                fs_imp, sig_imp = wavfile.read(binaural_rir_file)

                assert fs_imp == self.config_yaml.AUDIO.RIR_SAMPLING_RATE, "RIR doesn't have sampling frequency of rir_sampling_rate"
            except ValueError:
                sig_imp = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2)).astype("float32")
                fs_imp = self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        else:
            current_state = self.get_agent_state()
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
            audio_sensor.setAudioSourceTransform(np.array(current_state.position) + np.array([0, 1.5, 0]))
            sig_imp = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))
            fs_imp = self.config_yaml.AUDIO.RIR_SAMPLING_RATE

        if len(sig_imp) == 0:
            sig_imp = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2)).astype("float32")
            fs_imp = self.config_yaml.AUDIO.RIR_SAMPLING_RATE

        imp_full_length = np.zeros((self.config_yaml.AUDIO.RIR_SAMPLING_RATE, 2))

        if sig_imp.shape[0] > 128:
            imp_full_length[: min(sig_imp.shape[0], self.config_yaml.AUDIO.RIR_SAMPLING_RATE) - 128, :] =\
                sig_imp[128: min(sig_imp.shape[0], self.config_yaml.AUDIO.RIR_SAMPLING_RATE), :]    # remove the first 127 zero samples
        
        sig_imp = imp_full_length
        assert fs_imp == self.config_yaml.AUDIO.RIR_SAMPLING_RATE
        sig_imp = sig_imp.T
        fft_windows_l_imp = librosa.stft(np.asfortranarray(sig_imp[0]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_l_imp, _ = librosa.magphase(fft_windows_l_imp)
        fft_windows_r_imp = librosa.stft(np.asfortranarray(sig_imp[1]),
                                            hop_length=self.config_yaml.AUDIO.HOP_LENGTH,
                                            n_fft=self.config_yaml.AUDIO.N_FFT,
                                            win_length=self.config_yaml.AUDIO.WIN_LENGTH if (self.config_yaml.AUDIO.WIN_LENGTH != 0) else None,)
        magnitude_r_imp, _ = librosa.magphase(fft_windows_r_imp)
        magnitude_imp = np.stack([magnitude_l_imp, magnitude_r_imp], axis=-1)
        magnitude_imp = magnitude_imp.astype("float32")

        return magnitude_imp        
    
    
    def compute_semantic_index_mapping(self):
        # obtain mapping from instance id to semantic label id
        if isinstance(self._sim, DummySimulator):
            if self._current_scene not in self._house_readers:
                self._house_readers[self._current_sound] = HouseReader(self._current_scene.replace('.glb', '.house'))
            reader = self._house_readers[self._current_sound]
            instance_id_to_label_id = reader.compute_object_to_category_index_mapping()
        else:
            scene = self._sim.semantic_scene
            instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        self._instance2label_mapping = np.array([instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id))])

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        """ joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            assert not self.config.USE_RENDERED_OBSERVATIONS """
        sim_obs = self._sim.get_sensor_observations()
        for sensor in sim_obs:
            sim_obs[sensor] = sim_obs[sensor]
        #self._frame_cache[joint_index] = sim_obs
        return sim_obs

    def reset(self):
        logging.debug('Reset simulation')
        if self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)
        else:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        #TODO: change this away from SS 1.0 queries, and do not restrict it by subgraph.
        if self._get_queries_and_RIRs:
            #self.subgraph_index = 2 # only use if using that one scene
            if (self.current_scene_name, self.subgraph_index) not in self.scene_to_query_idx_mapping.keys():
                self._arr_query_poses_per_scene[self.current_scene_name][self.subgraph_index] =\
                    np.array(self._arr_query_poses_per_scene[self.current_scene_name][self.subgraph_index])
                num_scene_arbitrary_rir_poses = self._arr_query_poses_per_scene[self.current_scene_name][self.subgraph_index].shape[0]
                if self.split == "train":
                    #generate random pose idxs
                    if self.max_query_length > num_scene_arbitrary_rir_poses:
                        query_pose_idxs = torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                            num_scene_arbitrary_rir_poses).tolist()
                        query_pose_idxs += torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                            self.max_query_length - num_scene_arbitrary_rir_poses,
                                                            replacement=True).tolist()
                    else:
                        query_pose_idxs = torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                            self.max_query_length).tolist()
                else:
                    #load in pose idxs from disk
                    query_pose_idxs = self.arbitrary_rir_query_pose_idxs_from_disk[0][:self.max_query_length] #TODO: Can we fix this as 0?
                query_poses = self._arr_query_poses_per_scene[self.current_scene_name][self.subgraph_index][query_pose_idxs, :].tolist()
                self.scene_to_query_idx_mapping[(self.current_scene_name, self.subgraph_index)] = query_poses
            query_poses = self.scene_to_query_idx_mapping[(self.current_scene_name, self.subgraph_index)]

            self.gt_rirs_mags, self.gt_rirs_phases = self.generate_gt_RIRs(query_poses)
            self.query_poses = []
            for query in query_poses:
                pose = np.array(self.compute_relative_pose(current_pose=query, ref_pose=[self.config.AGENT_0.START_POSITION, self.azimuth_angle])).astype("float32")
                self.query_poses.append(pose)            

        return observations

    def get_RIR_reward_queries_RIRS(self):
        assert self.query_poses is not None
        assert self.gt_rirs_mags is not None
        assert self.gt_rirs_phases is not None
        return self.query_poses, np.array(self.gt_rirs_mags), np.array(self.gt_rirs_phases)
    
    def convert_external_pose_to_relative(self, external_pose):
        assert self.reference_pose is not None
        return np.array(self.compute_relative_pose(current_pose=external_pose, ref_pose=self.reference_pose)).astype("float32") 

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._last_rir = np.transpose(np.array(self._prev_sim_obs["audio_sensor"]))
        #self._previous_step_collided = False
        # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 3
        if self.config_yaml.UNIFORM_SAMPLE:
            if (self._episode_step_count == self.config_yaml.MAX_EPISODE_STEPS):
                self._is_episode_active = False
        else:
            if (self._episode_step_count == self.config_yaml.MAX_EPISODE_STEPS) or (self._collected_context == 19):
                self._is_episode_active = False
        #else:
            """ prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle
            if action == HabitatSimActions.MOVE_FORWARD or action == HabitatSimActions.MOVE_FORWARD_COLLECT:
                # the agent initially faces -Z by default
                self._previous_step_collided = True
                for neighbor in self.graph[self._receiver_position_index]:
                    p1 = self.graph.nodes[self._receiver_position_index]['point']
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._source_position_index = self._receiver_position_index
                        self._previous_step_collided = False
                        break
                if action == HabitatSimActions.MOVE_FORWARD_COLLECT:
                    self._collected_context += 1
            elif action == HabitatSimActions.TURN_LEFT or action == HabitatSimActions.TURN_LEFT_COLLECT:
                # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
                self._rotation_angle = (self._rotation_angle + 90) % 360
                if action == HabitatSimActions.TURN_LEFT_COLLECT:
                    self._collected_context += 1
            elif action == HabitatSimActions.TURN_RIGHT or action == HabitatSimActions.TURN_RIGHT_COLLECT:
                self._rotation_angle = (self._rotation_angle - 90) % 360
                if action == HabitatSimActions.TURN_RIGHT_COLLECT:
                    self._collected_context += 1 """
        if (action == HabitatSimActions.TURN_RIGHT_COLLECT) or (action == HabitatSimActions.TURN_LEFT_COLLECT) or (action == HabitatSimActions.MOVE_FORWARD_COLLECT):
            self._collected_context += 1

            """ if self.config.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations) """

            #self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
            #                     quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        if action == HabitatSimActions.TURN_RIGHT_COLLECT:
            sim_obs = self._sim.step(HabitatSimActions.TURN_RIGHT)
        elif action == HabitatSimActions.TURN_LEFT_COLLECT:
            sim_obs = self._sim.step(HabitatSimActions.TURN_LEFT)
        elif action == HabitatSimActions.MOVE_FORWARD_COLLECT:
            sim_obs = self._sim.step(HabitatSimActions.MOVE_FORWARD)
        else:
            sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs

        # log debugging info
        """ logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point'])) """

        #sim_obs = self._get_sim_observation()

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        #self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        self._episode_step_count += 1
        """ if self.config.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations """

        return observations

    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    @property
    def reaching_goal(self):
        return self._source_position_index == self._receiver_position_index

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sound = sound_file.split('.')[0]
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
            self._audio_length = audio_data.shape[0] // self.config.AUDIO.RIR_SAMPLING_RATE

    def _load_single_distractor_sound(self):
        if self._current_distractor_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.distractor_sound_dir, self._current_distractor_sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[self._current_distractor_sound] = audio_data

    def _load_single_source_sound(self):
        if self._current_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, self._current_sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            if audio_data.shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE == 1:
                audio_data = np.concatenate([audio_data] * 3, axis=0)  # duplicate to be longer than longest RIR
            self._source_sound_dict[self._current_sound] = audio_data
        self._audio_length = self._source_sound_dict[self._current_sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE

    def _compute_euclidean_distance_between_sr_locations(self):
        p1 = self.graph.nodes[self._receiver_position_index]['point']
        p2 = self.graph.nodes[self._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        return d

    def _compute_audiogoal(self):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        if self._episode_step_count > self._duration:
            logging.debug('Step count is greater than duration. Empty spectrogram.')
            audiogoal = np.zeros((2, sampling_rate))
        else:
            if self.config.USE_RENDERED_OBSERVATIONS:
                binaural_rir_file = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                    self._receiver_position_index, self._source_position_index))
                try:
                    sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
                except ValueError:
                    logging.warning("{} file is not readable".format(binaural_rir_file))
                    binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
                if len(binaural_rir) == 0:
                    logging.debug("Empty RIR file at {}".format(binaural_rir_file))
                    binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
            else:
                binaural_rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))

            # by default, convolve in full mode, which preserves the direct sound
            if self.current_source_sound.shape[0] == sampling_rate:
                binaural_convolved = np.array([fftconvolve(self.current_source_sound, binaural_rir[:, channel]
                                                           ) for channel in range(binaural_rir.shape[-1])])
                audiogoal = binaural_convolved[:, :sampling_rate]
            else:
                index = self._audio_index
                self._audio_index = (self._audio_index + 1) % self._audio_length
                if index * sampling_rate - binaural_rir.shape[0] < 0:
                    source_sound = self.current_source_sound[: (index + 1) * sampling_rate]
                    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]
                                                               ) for channel in range(binaural_rir.shape[-1])])
                    audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
                else:
                    # include reverb from previous time step
                    source_sound = self.current_source_sound[index * sampling_rate - binaural_rir.shape[0] + 1
                                                             : (index + 1) * sampling_rate]
                    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel], mode='valid',
                                                               ) for channel in range(binaural_rir.shape[-1])])
                    audiogoal = binaural_convolved

            if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
                binaural_rir_file = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
                    self._receiver_position_index, self._distractor_position_index))
                try:
                    sampling_freq, distractor_rir = wavfile.read(binaural_rir_file)
                except ValueError:
                    logging.warning("{} file is not readable".format(binaural_rir_file))
                    distractor_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)
                if len(distractor_rir) == 0:
                    logging.debug("Empty RIR file at {}".format(binaural_rir_file))
                    distractor_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)

                distractor_convolved = np.array([fftconvolve(self._source_sound_dict[self._current_distractor_sound],
                                                             distractor_rir[:, channel]
                                                             ) for channel in range(distractor_rir.shape[-1])])
                audiogoal += distractor_convolved[:, :sampling_rate]

        return audiogoal

    def get_egomap_observation(self):
        return None

    def cache_egomap_observation(self, egomap):
        self._egomap_cache[self._current_scene][(self._receiver_position_index, self._rotation_angle)] = egomap

    def get_current_audiogoal_observation(self):
        return self._compute_audiogoal()

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        return audiogoal2spectrogram(self.get_current_audiogoal_observation())

    def geodesic_distance(self, position_a, position_bs, episode=None):
        distances = []
        for position_b in position_bs:
            index_a = self._position_to_index(position_a)
            index_b = self._position_to_index(position_b)
            assert index_a is not None and index_b is not None
            path_length = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
            distances.append(path_length)

        return min(distances)

    def get_straight_shortest_path_points(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None

        shortest_path = nx.shortest_path(self.graph, source=index_a, target=index_b)
        points = list()
        for node in shortest_path:
            points.append(self.graph.nodes()[node]['point'])
        return points

    def compute_oracle_actions(self):
        start_node = self._receiver_position_index
        end_node = self._source_position_index
        shortest_path = nx.shortest_path(self.graph, source=start_node, target=end_node)
        assert shortest_path[0] == start_node and shortest_path[-1] == end_node
        logging.debug(shortest_path)

        oracle_actions = []
        orientation = self.get_orientation()
        for i in range(len(shortest_path) - 1):
            prev_node = shortest_path[i]
            next_node = shortest_path[i+1]
            p1 = self.graph.nodes[prev_node]['point']
            p2 = self.graph.nodes[next_node]['point']
            direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
            if direction == orientation:
                pass
            elif (direction - orientation) % 360 == 270:
                orientation = (orientation - 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_LEFT)
            elif (direction - orientation) % 360 == 90:
                orientation = (orientation + 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            elif (direction - orientation) % 360 == 180:
                orientation = (orientation - 180) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            oracle_actions.append(HabitatSimActions.MOVE_FORWARD)
        oracle_actions.append(HabitatSimActions.STOP)
        return oracle_actions

    def get_oracle_action(self):
        return self._oracle_actions[self._episode_step_count]

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def find_nearest_graph_node(self, target_pos):
        from scipy.spatial import cKDTree
        all_points = np.array([self.graph.nodes()[node]['point'] for node in self.graph.nodes()])
        kd_tree = cKDTree(all_points[:, [0, 2]])
        d, ind = kd_tree.query(target_pos[[0, 2]])
        return all_points[ind]

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)

    def write_info_to_obs(self, observations):
        r"""
        write agent location and orientation info, and scene name to observation dict... probably
        redundant
        :param observations: observation dict containing different info about the current observation
        :return: None
        """
        observations["agent node and location"] = (self._receiver_position_index,
                                                   self.graph.nodes[self._receiver_position_index]["point"])
        observations["scene name"] = self.current_scene_name
        observations["orientation"] = self._rotation_angle

    def _compute_rotation_from_azimuth(self, azimuth):
        """
        compute rotation angle from azimuth angle
        :param azimuth: azimuth angle
        :return: rotation angle
        """
        # rotation is calculated in the habitat coordinate frame counter-clocwise so -Z is 0 and -X is -90
        return -(azimuth + 0) % 360

    def get_receiver_position_idx(self):
        #for use in RelativePoseSensor
        return self._receiver_position_index
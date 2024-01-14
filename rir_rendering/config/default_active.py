from typing import List, Optional, Union
import os
import shutil

from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/active_context_sampler/train_active_context_sampler.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ActiveRIRTrainer"
_C.ENV_NAME = "ExploreEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.PARALLEL_GPU_IDS = [0,1,2,3,4,5,6,7]
_C.MODEL_DIR = ''
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_OPTION = []
_C.VIDEO_DIR = ''
_C.TEST_EPISODE_COUNT = 2 
_C.AUDIO_DIR = '' 
_C.VISUALIZATION_OPTION = [] 
_C.SAVE_INTERMEDIATE_RIR_ERRORS = False
_C.SAVE_INTERMEDIATE_FS_RIR_ERRORS = False
_C.TEST_REWARD_ZERO = False
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  
_C.NUM_PROCESSES = 1
_C.SENSORS = ["BIN_SPECT_MAG_SENSOR", "POSE_SENSOR", "RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000 #restore after debugging
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.EXTRA_DEPTH = False
_C.DEBUG = False
_C.UNIFORM_SAMPLE = False
_C.EPS_SCENES = []
_C.EPS_SCENES_N_IDS = []
_C.JOB_ID = 1
_C.TRAIN_OR_EVAL_FOR_SPECIFIC_SCENE = False
_C.SPECIFIC_SCENE_NAME = None
_C.CONTINUOUS = False
_C.DISPLAY_RESOLUTION = 128
_C.FOLLOW_SHORTEST_PATH = False
_C.USE_LAST_CKPT = False

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.DATA_PARALLEL_TRAINING = False

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_NOVELTY_REWARD = False
_C.RL.WITH_COVERAGE_REWARD = False
_C.RL.WITH_RIR_REWARD = False
_C.RL.MEASUREMENT_RIR_REWARD_SCALE = 3000.0
_C.RL.SPARSE_RIR_REWARD_SCALE = 1.0
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
_C.RL.TIME_DIFF = False
_C.RL.USE_EARLY_ANNEALING = False
_C.RL.EARLY_ANNEALING_FRAC = 0.2
_C.RL.USE_COOLDOWN_ANNEALING = False
_C.RL.SAMPLING_COOLDOWN_PERIOD = 30
_C.RL.NOVELTY_GRID_FACTOR = 5
_C.RL.PRETRAINED_RIR_PREDICTOR_PATH = "/vision/asomaya1/active-rir/runs_eval/fs_rir/data/seen_eval_best_ckpt.pth"
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.num_updates_per_cycle = 1
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr_exploration_pol = 1e-3
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.policy_type = None 
_C.RL.PPO.reward_type = None 
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.deterministic_eval = False
_C.RL.PPO.use_ddppo = False
_C.RL.PPO.ddppo_distrib_backend = "NCCL"
_C.RL.PPO.short_rollout_threshold = 1.0
_C.RL.PPO.sync_frac = 0.6
_C.RL.PPO.master_port = 7738 #8738
_C.RL.PPO.master_addr = "127.0.0.9"
# -----------------------------------------------------------------------------
# DEEP DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DDPPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "NCCL"
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# ACTIVE NEURAL SLAM (ANS)
# -----------------------------------------------------------------------------
_C.RL.ANS = CN()
_C.RL.ANS.pyt_random_seed = 123
_C.RL.ANS.planning_step = 0.50  # max distance of local goal from current position
_C.RL.ANS.goal_success_radius = 0.2  # success threshold for reaching a goal
_C.RL.ANS.goal_interval = 25  # goal sampling interval for global policy
_C.RL.ANS.thresh_explored = 0.6  # threshold to classify a cell as explored
_C.RL.ANS.thresh_obstacle = 0.6  # threshold to classify a cell as an obstacle
_C.RL.ANS.overall_map_size = 900  # world map size M
_C.RL.ANS.reward_type = "area_seen"  # Can be area_seen / map_accuracy
_C.RL.ANS.local_slack_reward = -0.3
_C.RL.ANS.local_collision_reward = -1.0
_C.RL.ANS.stop_action_id = 3
_C.RL.ANS.forward_action_id = 0
_C.RL.ANS.left_action_id = 1
_C.RL.ANS.image_scale_hw = [128, 128]
_C.RL.ANS.model_path = ""
_C.RL.ANS.recovery_heuristic = "random_explored_towards_goal"
_C.RL.ANS.crop_map_for_planning = True
# =============================================================================
# Mapper
# =============================================================================
_C.RL.ANS.MAPPER = CN()
_C.RL.ANS.MAPPER.lr = 1e-3
_C.RL.ANS.MAPPER.eps = 1e-5
_C.RL.ANS.MAPPER.max_grad_norm = 0.5
_C.RL.ANS.MAPPER.num_mapper_steps = 100  # number of steps per mapper update
_C.RL.ANS.MAPPER.map_size = 101  # V
_C.RL.ANS.MAPPER.map_scale = 0.05  # s in meters
_C.RL.ANS.MAPPER.projection_unit = "none"
_C.RL.ANS.MAPPER.pose_loss_coef = 30.0
_C.RL.ANS.MAPPER.detach_map = False
_C.RL.ANS.MAPPER.registration_type = "moving_average"
_C.RL.ANS.MAPPER.map_registration_momentum = 0.9
_C.RL.ANS.MAPPER.thresh_explored = 0.6  # threshold to classify a cell as explored
_C.RL.ANS.MAPPER.thresh_entropy = (
    0.5  # entropy threshold to classify a cell as confident
)
_C.RL.ANS.MAPPER.freeze_projection_unit = False
_C.RL.ANS.MAPPER.pose_predictor_inputs = ["ego_map"]
_C.RL.ANS.MAPPER.n_pose_layers = 1
_C.RL.ANS.MAPPER.n_ensemble_layers = 1
_C.RL.ANS.MAPPER.ignore_pose_estimator = False
_C.RL.ANS.MAPPER.label_id = "ego_map_gt_anticipated"
_C.RL.ANS.MAPPER.use_data_parallel = False
_C.RL.ANS.MAPPER.gpu_ids = []  # Set the GPUs for data parallel if necessary
_C.RL.ANS.MAPPER.num_update_batches = 50
_C.RL.ANS.MAPPER.replay_size = 100000
_C.RL.ANS.MAPPER.map_batch_size = 400
# Image normalization
_C.RL.ANS.MAPPER.NORMALIZATION = CN()
_C.RL.ANS.MAPPER.NORMALIZATION.img_mean = [0.485, 0.456, 0.406]
_C.RL.ANS.MAPPER.NORMALIZATION.img_std = [0.229, 0.224, 0.225]
# Image scaling
_C.RL.ANS.MAPPER.image_scale_hw = [128, 128]
# -----------------------------------------------------------------------------
# Uniform context sampler
# -----------------------------------------------------------------------------
_C.UniformContextSampler = CN()

_C.UniformContextSampler.lr = 5.0e-4
_C.UniformContextSampler.eps = 1.0e-5
_C.UniformContextSampler.max_grad_norm = None
_C.UniformContextSampler.betas = [0.9, 0.999]
_C.UniformContextSampler.num_epochs = 1000
_C.UniformContextSampler.batch_size = 64 
_C.UniformContextSampler.num_workers = 64 
_C.UniformContextSampler.set_num_workers_to_one_in_eval = False
_C.UniformContextSampler.num_datapoints_per_scene_train = 1000
_C.UniformContextSampler.num_datapoints_per_scene_eval = 50
_C.UniformContextSampler.predict_in_logspace = True
_C.UniformContextSampler.log_instead_of_log1p_in_logspace = False
_C.UniformContextSampler.log_gt = False
_C.UniformContextSampler.log_gt_eps = 1.0e-8
_C.UniformContextSampler.log1p_gt = False

_C.UniformContextSampler.use_spect_energy_decay_loss = False
_C.UniformContextSampler.spectEnergyDecayLoss = CN()
_C.UniformContextSampler.spectEnergyDecayLoss.type = "l1_loss" 
_C.UniformContextSampler.spectEnergyDecayLoss.weight =  1.0 
_C.UniformContextSampler.spectEnergyDecayLoss.slice_till_direct_signal = False
_C.UniformContextSampler.spectEnergyDecayLoss.direct_signal_len_in_ms = 50
_C.UniformContextSampler.spectEnergyDecayLoss.dont_collapse_across_freq_dim = False

_C.UniformContextSampler.TrainLosses = CN()
_C.UniformContextSampler.TrainLosses.types = ["stft_l1_loss"]
_C.UniformContextSampler.TrainLosses.weights = [1.0]

_C.UniformContextSampler.EvalMetrics = CN()
_C.UniformContextSampler.EvalMetrics.types = ["stft_l1_distance",
											  "diff_rt_startFrom60dB",
											  "diff_drr_3ms"]
_C.UniformContextSampler.EvalMetrics.type_for_ckpt_dump = "stft_l1_distance"

_C.UniformContextSampler.MemoryNet = CN()
_C.UniformContextSampler.MemoryNet.type = "transformer" 
_C.UniformContextSampler.MemoryNet.Transformer = CN()
_C.UniformContextSampler.MemoryNet.Transformer.no_self_attn_in_decoder = False
_C.UniformContextSampler.MemoryNet.Transformer.use_modified_input_size = False
_C.UniformContextSampler.MemoryNet.Transformer.input_size = 1024
_C.UniformContextSampler.MemoryNet.Transformer.hidden_size = 1024
_C.UniformContextSampler.MemoryNet.Transformer.num_encoder_layers = 2
_C.UniformContextSampler.MemoryNet.Transformer.num_decoder_layers = 2
_C.UniformContextSampler.MemoryNet.Transformer.nhead = 2
_C.UniformContextSampler.MemoryNet.Transformer.dropout = 0.1
_C.UniformContextSampler.MemoryNet.Transformer.activation = "relu"

_C.UniformContextSampler.encode_each_modality_as_independent_context_entry = False
_C.UniformContextSampler.append_modality_type_tag_encoding_to_each_modality_encoding = False
_C.UniformContextSampler.modality_type_tag_encoding_size = 8

_C.UniformContextSampler.PositionalEnc = CN()
_C.UniformContextSampler.PositionalEnc.type = "sinusoidal"
_C.UniformContextSampler.PositionalEnc.num_freqs_for_sinusoidal = 8
_C.UniformContextSampler.PositionalEnc.shared_pose_encoder_for_context_n_query = False
#_C.UniformContextSampler.PositionalEnc.n_positional_obs = 5

_C.UniformContextSampler.FusionEnc = CN()
_C.UniformContextSampler.FusionEnc.type = "concatenate"

_C.UniformContextSampler.FusionDec = CN()
_C.UniformContextSampler.FusionDec.type = "concatenate"

_C.UniformContextSampler.dump_audio_waveforms = False

_C.UniformContextSampler.use_gl = False
_C.UniformContextSampler.use_gl_for_gt = False
_C.UniformContextSampler.use_rand_phase = False
_C.UniformContextSampler.use_rand_phase_for_gt = False

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()

########## ACTIONS ###########
# -----------------------------------------------------------------------------
# PAUSE ACTION
# -----------------------------------------------------------------------------
_TC.TASK.ACTIONS.PAUSE = CN()
_TC.TASK.ACTIONS.PAUSE.TYPE = "PauseAction"

########## SENSORS ###########
# -----------------------------------------------------------------------------
# BINAURAL SPECTROGRAM MAGNITUDE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.BIN_SPECT_MAG_SENSOR = CN()
_TC.TASK.BIN_SPECT_MAG_SENSOR.TYPE = "BinSpectMagSensor"
_TC.TASK.BIN_SPECT_MAG_SENSOR.FEATURE_SHAPE = [256, 257, 2] # mp3d (n_fft=511, hop_length=62, win_length=400): [256, 259, 2]; 
# -----------------------------------------------------------------------------
# POSE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.POSE_SENSOR = CN()
_TC.TASK.POSE_SENSOR.TYPE = "ActivePoseSensor"
_TC.TASK.POSE_SENSOR.FEATURE_SHAPE = [5]
# -----------------------------------------------------------------------------
# environment config
# -----------------------------------------------------------------------------
_TC.ENVIRONMENT.MAX_EPISODE_STEPS = 100
_TC.ENVIRONMENT.MAX_CONTEXT_LENGTH = 100
_TC.ENVIRONMENT.MAX_QUERY_LENGTH = 25
_TC.ENVIRONMENT.MAX_QUERY_LENGTH_EVAL_FINETUNE_NAF = 25
_TC.ENVIRONMENT.LOAD_CONTEXT_FROM_DISK = False
_TC.ENVIRONMENT.LOAD_QUERY_FOR_ARBITRARY_RIRS_FROM_DISK = False
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_SCENE_NAMES_PATH = None
_TC.ENVIRONMENT.SEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH = None
_TC.ENVIRONMENT.UNSEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH = None
_TC.ENVIRONMENT.EVAL_CONTEXT_PERCENTAGES_PATH = "data/eval_arbitraryRIRQuery_datasets/mp3d/allEnv_14DatapointsPerEnv/test/60_qry/valid_context_percentages.npy"
# -----------------------------------------------------------------------------
# simulator config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.SEED = -1
_TC.SIMULATOR.SCENE_DATASET = "mp3d"
_TC.SIMULATOR.MAX_EPISODE_STEPS = 100 #
_TC.SIMULATOR.UNIFORM_SAMPLE = False
#_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.RENDERED_OBSERVATIONS = "data/scene_observations/"
_TC.SIMULATOR.DEFAULT_AGENT_ID = 0
_TC.SIMULATOR.SPLIT = _TC.DATASET.SPLIT
# -----------------------------------------------------------------------------
# audio config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.RIR_DIR = "data/binaural_rirs/mp3d"
_TC.SIMULATOR.AUDIO.META_DIR = "data/metadata/mp3d"
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.NUM_WORKER = 4
_TC.SIMULATOR.AUDIO.BATCH_SIZE = 128
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 16000
_TC.SIMULATOR.AUDIO.HOP_LENGTH = 62 
_TC.SIMULATOR.AUDIO.N_FFT = 511 
_TC.SIMULATOR.AUDIO.WIN_LENGTH = 248 
_TC.SIMULATOR.AUDIO.SWEEP_AUDIO_DIR = "data/audio_data/sweep_sounds/visual_echoes/"
_TC.SIMULATOR.AUDIO.SWEEP_AUDIO_FILENAME = "data_sweep_audio_3ms_sweep.wav"
_TC.SIMULATOR.AUDIO.VALID_ECHO_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = False
# -----------------------------------------------------------------------------
# soundspaces config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
_TC.SIMULATOR.STEP_TIME = 1.0
_TC.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"
_TC.SIMULATOR.AUDIO.EVERLASTING = True
_TC.SIMULATOR.AUDIO.CROSSFADE = False
# -----------------------------------------------------------------------------
# DistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.ACOUSTIC_MAP_ERROR = CN()
_TC.TASK.ACOUSTIC_MAP_ERROR.TYPE = "AcousticMapError"
# -----------------------------------------------------------------------------
# EgoMap Sensor
# -----------------------------------------------------------------------------
_TC.TASK.EGOMAP_SENSOR = SIMULATOR_SENSOR.clone()
_TC.TASK.EGOMAP_SENSOR.TYPE = "EgoMap"
_TC.TASK.EGOMAP_SENSOR.MAP_SIZE = 101
_TC.TASK.EGOMAP_SENSOR.MAP_RESOLUTION = 0.1
_TC.TASK.EGOMAP_SENSOR.HEIGHT_THRESH = (0.5, 2.0)
# -----------------------------------------------------------------------------
# GlobalMap Sensor
# -----------------------------------------------------------------------------
_TC.TASK.GLOBALMAP_SENSOR = SIMULATOR_SENSOR.clone()
_TC.TASK.GLOBALMAP_SENSOR.TYPE = "GlobalMap"
_TC.TASK.GLOBALMAP_SENSOR.FEATURE_SHAPE = [2, 241, 241]
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
_TC.DATASET.CONTINUOUS = False

def merge_from_path(config, config_paths):
	"""
	merge config with configs from config paths
	:param config: original unmerged config
	:param config_paths: config paths to merge configs from
	:return: merged config
	"""
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	return config


def get_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None,
		model_dir: Optional[str] = None,
		run_type: Optional[str] = None
) -> CN:
	r"""Create a unified config with default values overwritten by values from
	`config_paths` and overwritten by options from `opts`.
	Args:
		config_paths: List of config paths or string that contains comma
		separated list of config paths.
		opts: Config options (keys, values) in a list (e.g., passed from
		command line into the config. For example, `opts = ['FOO.BAR',
		0.5]`. Argument can be used for parameter sweeping or quick tests.
		model_dir: suffix for output dirs
		run_type: either train or eval
	"""
	config = merge_from_path(_C.clone(), config_paths)
	config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

	if opts:
		config.CMD_TRAILING_OPTS = opts
		config.merge_from_list(opts)

	assert model_dir is not None, "set --model-dir"
	config.MODEL_DIR = model_dir
	config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, config.TENSORBOARD_DIR)
	config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
	config.VIDEO_DIR = os.path.join(config.MODEL_DIR, 'video_dir')
	config.AUDIO_DIR = os.path.join(config.MODEL_DIR, 'audio_dir')
	config.LOG_FILE = os.path.join(config.MODEL_DIR, config.LOG_FILE)
	if config.EVAL_CKPT_PATH_DIR == "data/checkpoints":
		config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, 'data')

	dirs = [config.VIDEO_DIR, config.AUDIO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
	if run_type == 'train':
		# check dirs
		if any([os.path.exists(d) for d in dirs]):
			for d in dirs:
				if os.path.exists(d):
					print('{} exists'.format(d))
			key = input('Output directory already exists! Overwrite the folder? (y/n)')
			if key == 'y':
				for d in dirs:
					if os.path.exists(d):
						shutil.rmtree(d)

	config.TASK_CONFIG.defrost()

	#------------------ modifying SIMULATOR cfg --------------------
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_TRAIN_QUERY_POSE_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_TRAIN_QUERY_POSE_SUBGRAPH_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_SUBGRAPH_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_TRAIN_SCENE_NAMES_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_TRAIN_SCENE_NAMES_PATH
	config.TASK_CONFIG.SIMULATOR.SEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.SEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH
	config.TASK_CONFIG.SIMULATOR.UNSEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.UNSEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
	config.TASK_CONFIG.SIMULATOR.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH = config.TASK_CONFIG.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH
	
	config.TASK_CONFIG.SIMULATOR.MAX_QUERY_LENGTH = config.TASK_CONFIG.ENVIRONMENT.MAX_QUERY_LENGTH

	## setting SIMULATOR'S USE_SYNC_VECENV flag
	config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV
	if config.CONTINUOUS:
		config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 1.0 #TODO: change to 0.25 if using truly continuous later on
		config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 90 #TODO: change to 30 if using truly continuous later on
		config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
		config.TASK_CONFIG.SIMULATOR.STEP_TIME = 0.25
		config.TASK_CONFIG.SIMULATOR.AUDIO.CROSSFADE = True
		config.TASK_CONFIG.DATASET.CONTINUOUS = True
	else:
		config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = config.TASK_CONFIG.SIMULATOR.GRID_SIZE

	## setting max. number of steps of simulator
	config.TASK_CONFIG.SIMULATOR.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
	config.TASK_CONFIG.SIMULATOR.UNIFORM_SAMPLE = config.UNIFORM_SAMPLE

	#-------------------------- modifying cfgs for visualization -------------------
	if len(config.VIDEO_OPTION) > 0:
		config.VISUALIZATION_OPTION = ["top_down_map"]
		config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False

	config.TASK_CONFIG.freeze()

	config.freeze()

	#---------------------------- assertions for metrics --------------------------------
	if (config.TRAINER_NAME == "uniform_context_sampler") and (run_type == "train"):
		assert config.UniformContextSampler.EvalMetrics.type_for_ckpt_dump\
			   in config.UniformContextSampler.EvalMetrics.types
	return config


def get_task_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None
) -> habitat.Config:
	r"""
	get config after merging configs stored in yaml files and command line arguments
	:param config_paths: paths to configs
	:param opts: optional command line arguments
	:return: merged config
	"""
	config = _TC.clone()
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	if opts:
		config.merge_from_list(opts)

	config.freeze()
	return config

import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    max_stages: int = 300
    timesteps_per_stage: int = 50000
    save_dir: str = "rl_models_competitive"
    log_dir: str = "rl_logs_competitive"
    initial_ent_coef_a: float = 0.05
    initial_ent_coef_b: float = 0.1
    ent_coef_increment: float = 0.1
    max_ent_coef: float = 0.5
    eval_episodes_for_competition: int = 200
    gauntlet_eval_episodes_per_color: int = 100
    gauntlet_timesteps: int = 50000
    n_envs_vs_c: int = 2
    n_envs_self_play: int = 2
    best_model_filename: str = "best_model.zip"
    vec_normalize_stats_filename: str = "vec_normalize.pkl"


CONFIG = TrainingConfig()

MAX_STAGES = CONFIG.max_stages
TIMESTEPS_PER_STAGE = CONFIG.timesteps_per_stage
SAVE_DIR = CONFIG.save_dir
LOG_DIR = CONFIG.log_dir
INITIAL_ENT_COEF_A = CONFIG.initial_ent_coef_a
INITIAL_ENT_COEF_B = CONFIG.initial_ent_coef_b
ENT_COEF_INCREMENT = CONFIG.ent_coef_increment
MAX_ENT_COEF = CONFIG.max_ent_coef
EVAL_EPISODES_FOR_COMPETITION = CONFIG.eval_episodes_for_competition
GAUNTLET_EVAL_EPISODES_PER_COLOR = CONFIG.gauntlet_eval_episodes_per_color
GAUNTLET_TIMESTEPS = CONFIG.gauntlet_timesteps
N_ENVS_VS_C = CONFIG.n_envs_vs_c
N_ENVS_SELF_PLAY = CONFIG.n_envs_self_play
BEST_MODEL_FILENAME = CONFIG.best_model_filename
VEC_NORMALIZE_STATS_PATH = os.path.join(SAVE_DIR, CONFIG.vec_normalize_stats_filename)
TRAINING_STATE_FILE = os.path.join(SAVE_DIR, "training_state.npy")

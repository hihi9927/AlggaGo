import os
import time
import re
import numpy as np
import torch
import pygame
import csv
import itertools
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from opponent_c import model_c_action
from opponent import get_regular_action, get_split_shot_action

import gymnasium as gym
from gymnasium import spaces

from env import AlggaGoEnv
from physics import WIDTH, HEIGHT, all_stones_stopped, MARGIN

# --- 하이퍼파라미터 및 설정 ---
MAX_STAGES = 300
TIMESTEPS_PER_STAGE = 50000
SAVE_DIR = "rl_models_competitive"
LOG_DIR = "rl_logs_competitive"
INITIAL_ENT_COEF_A = 0.05
INITIAL_ENT_COEF_B = 0.1
ENT_COEF_INCREMENT = 0.1
MAX_ENT_COEF = 0.5
EVAL_EPISODES_FOR_COMPETITION = 200
GAUNTLET_EVAL_EPISODES_PER_COLOR = 100
CHECKPOINT_SAVE_FREQ = 10_000_000

TIMESTEPS_PER_GAUNTLET_STAGE = 50000  # 예선전 각 스테이지 당 학습할 타임스텝
GAUNTLET_WIN_RATE_THRESHOLD = 0.5

# [수정] 상태 저장 파일 경로 정의
TRAINING_STATE_FILE = os.path.join(SAVE_DIR, "training_state.npy")

# --- 진행률 표시 콜백 클래스 ---
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.start_num = self.model.num_timesteps
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="학습 진행률",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def _on_step(self):
        done_ts = self.model.num_timesteps - self.start_num
        if self.pbar:
            self.pbar.n = min(done_ts, self.total_timesteps)
            self.pbar.refresh()
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close(); self.pbar = None

class StrategyPolicyEnv(gym.Env):
    def __init__(self, agent_side="black"):
        super().__init__()
        self.base = AlggaGoEnv()
        self.agent_side = agent_side
        # 전략만: 0(regular), 1(split)
        self.action_space = spaces.Discrete(2)
        self.observation_space = self.base.observation_space
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        self.base.set_forced_strategy(None)  # reset 시 기본값
        obs, info = self.base.reset(options={"initial_player": self.agent_side})
        # 상대 차례면 한 번 진행 (기존 코드 패턴과 유사)
        if self.base.current_player != self.agent_side:
            # 상대는 그냥 rule-based regular로 돌리거나, 아무 것도 안 해도 됨
            self.base.set_forced_strategy(0)
            dummy_action = np.array([0,0, 0,0,0], dtype=np.float32)
            _,_,_,_,_ = self.base.step(dummy_action)
        self._last_obs = self.base._get_obs()
        return self._last_obs, info

    def step(self, action):
        # 강제 전략 설정
        strat = int(action)
        self.base.set_forced_strategy(strat)
        # 오프셋은 0으로 고정
        a = np.array([0,0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, rew, term, trunc, info = self.base.step(a)
        # 한 턴만 학습하므로 opponent 턴은 스킵(분리학습 단순화)
        self._last_obs = self.base._get_obs()
        return self._last_obs, rew, term, trunc, info

    def close(self):
        self.base.close()

# === 오프셋 전용 환경: 전략은 고정, 오프셋(Box(3))만 학습 ===
class OffsetPolicyEnv(gym.Env):
    def __init__(self, forced_strategy: int, agent_side="black"):
        super().__init__()
        assert forced_strategy in (0,1), "forced_strategy must be 0 or 1"
        self.base = AlggaGoEnv()
        self.agent_side = agent_side
        self.fixed_strategy = forced_strategy
        # 오프셋: index, angle, force (원 env와 범위 동일)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = self.base.observation_space
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(options={"initial_player": self.agent_side})
        # 내 턴이 아니면 상대 한 번 처리(간단화를 위해 regular 강제)
        if self.base.current_player != self.agent_side:
            self.base.set_forced_strategy(0)
            dummy = np.array([0,0, 0,0,0], dtype=np.float32)
            _,_,_,_,_ = self.base.step(dummy)
        self._last_obs = self.base._get_obs()
        return self._last_obs, info

    def step(self, offsets):
        # 전략 고정
        self.base.set_forced_strategy(self.fixed_strategy)
        # offsets는 [idx, ang, force] → 원 env 액션으로 포장
        idx, ang, frc = np.asarray(offsets, dtype=np.float32)
        a = np.array([0,0, idx, ang, frc], dtype=np.float32)
        obs, rew, term, trunc, info = self.base.step(a)
        self._last_obs = self.base._get_obs()
        return self._last_obs, rew, term, trunc, info

    def close(self):
        self.base.close()

# --- Rule-based 초기화 함수 ---
def initialize_to_rule_based(model):
    with torch.no_grad():
        action_net = model.policy.action_net
        action_net.weight.data.fill_(0.0)
        action_net.bias[0].data.fill_(10.0)
        action_net.bias[1].data.fill_(-10.0)
        action_net.bias[2].data.fill_(0.0)
        action_net.bias[3].data.fill_(0.0)
        action_net.bias[4].data.fill_(0.0)
        if isinstance(model.policy.log_std, torch.nn.Parameter):
            model.policy.log_std.data.fill_(-20.0)

# --- 환경 생성 헬퍼 함수 ---
def make_env_fn():
    def _init():
        env = AlggaGoEnv()
        monitored_env = Monitor(env, filename=LOG_DIR)
        return monitored_env
    return _init

# --- 상태 저장/로드 함수 ---
def load_training_state():
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            state = np.load(TRAINING_STATE_FILE, allow_pickle=True).item()
            print(f"[INFO] 이전 학습 상태 로드 성공: {state}")
            return state
        except Exception as e: print(f"[ERROR] 학습 상태 로드 실패: {e}")
    return None
def save_training_state(state_dict):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TRAINING_STATE_FILE, state_dict)

# --- 공정 평가(Fair Evaluation) 함수 ---
def evaluate_fairly(model_A: PPO, model_B: PPO, num_episodes: int):
    games_per_round = num_episodes // 2
    if games_per_round == 0: return 0.5, 0.5, 0.0, 0.0
    print(f"   - 공정한 평가: 총 {num_episodes} 게임 ({games_per_round} 게임/라운드)")

    def _play_round(black_model: PPO, white_model: PPO, num_games: int, round_name: str):
        black_wins = 0
        env = Monitor(AlggaGoEnv())
        for _ in tqdm(range(num_games), desc=round_name, leave=False):
            obs, _ = env.reset(options={"initial_player": "black"})
            done = False
            while not done:
                current_player = env.env.current_player
                action_model = black_model if current_player == 'black' else white_model
                action, _ = action_model.predict(obs, deterministic=True)
                squeezed_action = np.squeeze(action)
                obs, _, done, _, info = env.step(squeezed_action)
            if info.get('winner') == 'black': black_wins += 1
        env.close()
        return black_wins / num_games if num_games > 0 else 0

    original_ent_A, original_ent_B = model_A.ent_coef, model_B.ent_coef
    win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate = 0.5, 0.5, 0.0, 0.0
    try:
        model_A.ent_coef = 0.0
        model_B.ent_coef = 0.0
        r1_black_win_rate = _play_round(model_A, model_B, games_per_round, "1라운드 (A가 흑돌)")
        r2_black_win_rate = _play_round(model_B, model_A, games_per_round, "2라운드 (B가 흑돌)")
        win_rate_A = (r1_black_win_rate + (1 - r2_black_win_rate)) / 2
        win_rate_B = (r2_black_win_rate + (1 - r1_black_win_rate)) / 2
    finally:
        model_A.ent_coef, model_B.ent_coef = original_ent_A, original_ent_B
    return win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate

def evaluate_vs_model_c(ppo_model: PPO, num_episodes_per_color: int):
    env = AlggaGoEnv()
    from pymunk import Vec2d
    from physics import scale_force
    win_rates, total_wins = {}, 0
    strategy_attempts, strategy_successes = {0: 0, 1: 0}, {0: 0, 1: 0}
    for side in ["black", "white"]:
        ppo_wins_on_side = 0
        desc = f"   PPO({side}) vs C"
        for _ in tqdm(range(num_episodes_per_color), desc=desc, leave=False):
            obs, _ = env.reset(options={"initial_player": side})
            done, info = False, {}
            while not done:
                current_player_color = env.current_player
                if current_player_color == side:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    obs, _, done, _, info = env.step(action)
                    strategy = info.get('strategy_choice')
                    if strategy is not None:
                        strategy_attempts[strategy] += 1
                        if info.get('is_regular_success', False) or info.get('is_split_success', False):
                            strategy_successes[strategy] += 1
                else:
                    action_tuple = model_c_action(env.stones, current_player_color)
                    if action_tuple:
                        idx, angle, force = action_tuple
                        player_stones = [s for s in env.stones if s.color[:3] == ((0,0,0) if current_player_color=="black" else (255,255,255))]
                        if 0 <= idx < len(player_stones):
                            stone_to_shoot = player_stones[idx]
                            direction = Vec2d(1, 0).rotated(angle)
                            impulse = direction * scale_force(force)
                            stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)
                    physics_steps = 0
                    while not all_stones_stopped(env.stones) and physics_steps < 600:
                        env.space.step(1/60.0); physics_steps += 1
                    for shape in env.stones[:]:
                        if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                            env.space.remove(shape, shape.body); env.stones.remove(shape)
                    current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
                    current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
                    if current_black == 0: done = True; info['winner'] = 'white'
                    elif current_white == 0: done = True; info['winner'] = 'black'
                    env.current_player = "white" if current_player_color == "black" else "black"
                    obs = env._get_obs()
            if done and info.get('winner') == side:
                ppo_wins_on_side += 1
        win_rates[side] = ppo_wins_on_side / num_episodes_per_color if num_episodes_per_color > 0 else 0
        total_wins += ppo_wins_on_side
    env.close()
    overall_win_rate = total_wins / (num_episodes_per_color * 2) if num_episodes_per_color > 0 else 0
    regular_success_rate = strategy_successes[0] / strategy_attempts[0] if strategy_attempts[0] > 0 else 0
    split_success_rate = strategy_successes[1] / strategy_attempts[1] if strategy_attempts[1] > 0 else 0
    total_strategy_attempts = strategy_attempts[0] + strategy_attempts[1]
    regular_attack_ratio = strategy_attempts[0] / total_strategy_attempts if total_strategy_attempts > 0 else 0
    return (overall_win_rate, win_rates.get("black", 0), win_rates.get("white", 0),
            regular_success_rate, split_success_rate, regular_attack_ratio)

def train_and_eval_separated_policies():
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    os.makedirs("models", exist_ok=True)

    # 1) 전략 정책 학습
    def make_strat_env():
        e = StrategyPolicyEnv(agent_side="black")
        return Monitor(e)
    strat_env = DummyVecEnv([make_strat_env for _ in range(4)])
    strat_model = PPO("MlpPolicy", strat_env, verbose=1, ent_coef=0.02)
    strat_model.learn(total_timesteps=200_000)
    strat_model.save("models/strategy_black.zip")
    strat_env.close()

    # 2) 오프셋(일반) 학습
    def make_off_reg_env():
        e = OffsetPolicyEnv(forced_strategy=0, agent_side="black")
        return Monitor(e)
    off_reg_env = DummyVecEnv([make_off_reg_env for _ in range(4)])
    off_reg_model = PPO("MlpPolicy", off_reg_env, verbose=1, ent_coef=0.01)
    off_reg_model.learn(total_timesteps=200_000)
    off_reg_model.save("models/offset_regular_black.zip")
    off_reg_env.close()

    # 3) 오프셋(스플릿) 학습
    def make_off_split_env():
        e = OffsetPolicyEnv(forced_strategy=1, agent_side="black")
        return Monitor(e)
    off_split_env = DummyVecEnv([make_off_split_env for _ in range(4)])
    off_split_model = PPO("MlpPolicy", off_split_env, verbose=1, ent_coef=0.01)
    off_split_model.learn(total_timesteps=200_000)
    off_split_model.save("models/offset_split_black.zip")
    off_split_env.close()

    # 4) 결합 어댑터 구성 → 기존 평가 함수 재사용
    combined = CombinedPolicy(
        strat_model=strat_model,
        off_reg_model=off_reg_model,
        off_split_model=off_split_model,
        dummy_env_factory=lambda: AlggaGoEnv()
    )
    ppo_like = CombinedPPOAdapter(combined)

    # 기존 평가 루틴을 그대로 사용 (행동 5D 반환만 지켜주면 동작)
    overall, b_win, w_win, reg_succ, split_succ, reg_ratio = evaluate_vs_model_c(
        ppo_like, num_episodes_per_color=50
    )
    print("[EVAL] overall:", overall,
          "black:", b_win, "white:", w_win,
          "reg_succ:", reg_succ, "split_succ:", split_succ, "reg_ratio:", reg_ratio)

class CombinedPolicy:
    """
    전략 모델(Discrete) + 오프셋 모델 2개(regular/split)를 결합하여
    최종 5D 액션을 반환하는 추론기.
    """
    def __init__(self, strat_model, off_reg_model, off_split_model, dummy_env_factory=lambda: AlggaGoEnv()):
        # SB3는 env 필요하므로 더미 VecEnv 부착
        from stable_baselines3.common.vec_env import DummyVecEnv
        self.vec = DummyVecEnv([dummy_env_factory])
        self.strat = strat_model
        self.off_reg = off_reg_model
        self.off_split = off_split_model

    def predict(self, obs, deterministic=True):
        # 1) 전략 선택 (0=regular, 1=split)
        s_action, _ = self.strat.predict(obs, deterministic=deterministic)
        s = int(s_action)

        # 2) 오프셋 예측
        if s == 0:
            o_action, _ = self.off_reg.predict(obs, deterministic=deterministic)
        else:
            o_action, _ = self.off_split.predict(obs, deterministic=deterministic)

        idx, ang, frc = np.asarray(o_action, dtype=np.float32)

        # 3) 최종 5D 액션 (전략 프리퍼런스 슬롯은 0 채움)
        a = np.array([0.0, 0.0, idx, ang, frc], dtype=np.float32)
        return a, None


class CombinedPPOAdapter:
    """
    기존 evaluate_vs_model_c 같은 함수가 PPO 인터페이스를 기대하므로,
    CombinedPolicy를 'predict'만 있는 'PPO처럼' 보이게 감싸는 얇은 어댑터.
    """
    def __init__(self, combined_policy):
        self.combined = combined_policy
        # ent_coef 등 속성을 참조할 수도 있어 보관 (필요 시)
        self.ent_coef = 0.0

    def predict(self, obs, deterministic=True):
        return self.combined.predict(obs, deterministic=deterministic)
    
class VsFixedOpponentEnv(gym.Env):
    def __init__(self, opponent_model: PPO, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv()
        self.opponent = opponent_model
        self.agent_side = agent_side
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self._last_obs = None
    def reset(self, *, seed=None, options=None):
        self._last_obs, info = self.base_env.reset(options={"initial_player": self.agent_side})
        if self.base_env.current_player != self.agent_side:
            self._play_opponent_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info
    def step(self, action):
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated: return obs, reward_agent, terminated, truncated, info
        opp_reward = self._play_opponent_turn()
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_opponent()
        total_reward = (reward_agent - opp_reward) + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info
    def _play_opponent_turn(self):
        if self.base_env.current_player == self.agent_side: return 0.0
        opp_obs = self.base_env._get_obs()
        opp_action, _ = self.opponent.predict(opp_obs, deterministic=True)
        _obs, opp_reward, _terminated, _truncated, _info = self.base_env.step(opp_action)
        return opp_reward
    def set_opponent(self, new_opponent_model: PPO): self.opponent = new_opponent_model
    def _check_terminal_and_penalty_after_opponent(self):
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: winner = "white"; terminated = True
        elif current_white == 0: winner = "black"; terminated = True
        else: return False, 0.0
        if (winner == "white" and self.agent_side == "black") or (winner == "black" and self.agent_side == "white"): return True, -5.0
        return True, 0.0

def make_vs_opponent_env_vec(opponent_model: PPO, n_envs: int = 2):
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsFixedOpponentEnv(opponent_model=opponent_model, agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

# --- [오류 수정] 예선전용 환경(VsModelCEnv)과 헬퍼 함수 복원 ---
class VsModelCEnv(gym.Env):
    def __init__(self, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv()
        self.agent_side = agent_side
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self._last_obs = None
    def reset(self, *, seed=None, options=None):
        self._last_obs, info = self.base_env.reset(options={"initial_player": self.agent_side})
        if self.base_env.current_player != self.agent_side:
            self._play_model_c_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info
    def step(self, action):
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated: return obs, reward_agent, terminated, truncated, info
        self._play_model_c_turn()
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_c_turn()
        total_reward = reward_agent + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info
    def _play_model_c_turn(self):
        current_player_color = self.base_env.current_player
        if current_player_color == self.agent_side: return
        action_tuple = model_c_action(self.base_env.stones, current_player_color)
        if action_tuple:
            from pymunk import Vec2d
            from physics import scale_force
            idx, angle, force = action_tuple
            player_color_tuple = (0, 0, 0) if current_player_color == "black" else (255, 255, 255)
            player_stones = [s for s in self.base_env.stones if s.color[:3] == player_color_tuple]
            if 0 <= idx < len(player_stones):
                stone_to_shoot = player_stones[idx]
                direction = Vec2d(1, 0).rotated(angle)
                impulse = direction * scale_force(force)
                stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)
            physics_steps = 0
            while not all_stones_stopped(self.base_env.stones) and physics_steps < 600:
                self.base_env.space.step(1/60.0); physics_steps += 1
            for shape in self.base_env.stones[:]:
                if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                    if shape in self.base_env.space.shapes: self.base_env.space.remove(shape, shape.body)
                    if shape in self.base_env.stones: self.base_env.stones.remove(shape)
            self.base_env.current_player = "white" if current_player_color == "black" else "black"
    def _check_terminal_and_penalty_after_c_turn(self):
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: winner = "white"; terminated = True
        elif current_white == 0: winner = "black"; terminated = True
        else: return False, 0.0
        if (winner == "white" and self.agent_side == "black") or (winner == "black" and self.agent_side == "white"): return True, -5.0
        return True, 0.0

def make_vs_c_env_vec(n_envs: int = 2):
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsModelCEnv(agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

# --- 시각화 함수 (복원) ---
def visualize_one_game(model_A: PPO, model_B: PPO, ent_A: float, ent_B: float, stage_num: int):
    stage_str = f"스테이지 {stage_num}" if stage_num > 0 else "초기 상태"
    black_model, white_model = (model_A, model_B) if ent_A >= ent_B else (model_B, model_A)
    caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
    print(f"\n--- 시각화 평가: {caption} ---")
    env = AlggaGoEnv()
    obs, _ = env.reset(options={"initial_player": "black"})
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)
    from pymunk import Vec2d
    from physics import scale_force
    done = False
    info = {}
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: break
        current_player = env.current_player
        action_model = black_model if current_player == "black" else white_model
        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)
        action_values, _ = action_model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action_values)
    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()

def visualize_vs_model_c(ppo_model: PPO, round_num: int, ppo_player_side: str):
    stage_str = f"특별 훈련 {round_num}라운드"
    caption = (f"{stage_str}: 모델 A(흑돌) vs 모델 C(백돌)" if ppo_player_side == "black"
               else f"{stage_str}: 모델 C(흑돌) vs 모델 A(백돌)")
    print(f"\n--- {stage_str} 시각화 ({caption}) ---")
    env = AlggaGoEnv()
    obs, _ = env.reset(options={"initial_player": "black"})
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)
    from pymunk import Vec2d
    from physics import scale_force
    done = False
    info = {}
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue
        current_player_color = env.current_player
        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)
        if current_player_color == ppo_player_side:
            action_values, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action_values)
        else:
            action_tuple = model_c_action(env.stones, current_player_color)
            if action_tuple:
                idx, angle, force = action_tuple
                player_color_tuple = (0, 0, 0) if current_player_color == "black" else (255, 255, 255)
                player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
                if 0 <= idx < len(player_stones):
                    stone_to_shoot = player_stones[idx]
                    direction = Vec2d(1, 0).rotated(angle)
                    impulse = direction * scale_force(force)
                    stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)
            physics_steps = 0
            while not all_stones_stopped(env.stones) and physics_steps < 600:
                env.space.step(1/60.0); physics_steps += 1
            for shape in env.stones[:]:
                if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                    env.space.remove(shape, shape.body); env.stones.remove(shape)
            current_black = sum(1 for s in env.stones if s.color[:3] == (0, 0, 0))
            current_white = sum(1 for s in env.stones if s.color[:3] == (255, 255, 255))
            if current_black == 0: done = True; info['winner'] = 'white'
            elif current_white == 0: done = True; info['winner'] = 'black'
            if not done: env.current_player = "white" if current_player_color == "black" else "black"
            obs = env._get_obs()
    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()

def visualize_split_shot_debug(model: PPO):
    print("\n🔬 '틈새 공격' 디버깅 시각화 시작 🔬")
    env = AlggaGoEnv()
    obs, _ = env.reset(options={"initial_player": "black"})
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DEBUG: Split Shot Only")
    from pymunk import Vec2d
    from physics import scale_force
    done = False
    info = {}
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue
        current_player_color = env.current_player
        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(1.0)
        if current_player_color == "black":
            action_values, _ = model.predict(obs, deterministic=True)
            action_values[0] = -1.0; action_values[1] = 1.0 # Force split shot
            obs, _, done, _, info = env.step(action_values)
            if info.get('is_split_success'): print("   [DEBUG] >> 결과: 틈새 공격 성공!")
            else: print("   [DEBUG] >> 결과: 틈새 공격 실패.")
        else: # Model C turn
            action_tuple = model_c_action(env.stones, current_player_color)
            if action_tuple:
                # Apply action (simplified for visualization)
                pass
            env.current_player = "black"
            obs = env._get_obs()
    winner = info.get('winner', 'Draw/Timeout')
    print(f"\n>>> 디버깅 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(3)
    pygame.quit()

# --- 예선전 함수 (수정) ---
def run_gauntlet_training(model_name, initial_timesteps, training_state):
    print(f"\n🥊 최종 규칙 기반 예선전 시작: 모델 {model_name} vs 모델 C 🥊")
    # TIMESTEPS_PER_GAUNTLET_STAGE 와 GAUNTLET_WIN_RATE_THRESHOLD 는 파일 상단에 정의되어 있습니다.
    
    MAX_BACKUPS = 5
    stage_backup_paths = training_state.get("stage_backup_paths", [])
    current_total_timesteps = initial_timesteps
    GAUNTLET_LOG_FILE = os.path.join(LOG_DIR, "gauntlet_log.csv")

    # 실행 시마다 로그 파일 새로 만들기
    with open(GAUNTLET_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Total Timesteps", "Win Rate as Black", "Win Rate as White", "Overall Win Rate", "Regular Success", "Split Success", "Regular Ratio", "Entropy"])
    
    N_ENVS_VS_C = 2
    gauntlet_env_raw = make_vs_c_env_vec(n_envs=N_ENVS_VS_C)
    gauntlet_env = VecNormalize(gauntlet_env_raw, norm_obs=True, norm_reward=True)

    if stage_backup_paths:
        model_path = stage_backup_paths[-1]
        print(f"\n[INFO] 가장 최근 예선전 백업 '{os.path.basename(model_path)}'에서 스테이지를 시작합니다.")
        gauntlet_model = PPO.load(model_path, env=gauntlet_env)
    else:
        print("\n[INFO] 예선전 백업 없음. 규칙 기반으로 초기화하여 시작합니다.")
        gauntlet_model = PPO("MlpPolicy", gauntlet_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A, max_grad_norm=0.5)
        initialize_to_rule_based(gauntlet_model)
        
        print("\n--- [초기 성능 평가] 훈련 시작 전 ---")
        (win_rate, win_as_black, win_as_white, 
         reg_success, split_success, reg_ratio) = evaluate_vs_model_c(
         gauntlet_model, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR
        )
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([0, current_total_timesteps, f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}", f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}", f"{gauntlet_model.ent_coef:.4f}"])
        print("[INFO] 규칙 기반 초기 상태의 평가 결과가 로그에 기록되었습니다.")
    

    # 스테이지 번호를 루프 밖에서 초기화
    current_stage_num = len(stage_backup_paths)
    
    while True:
        # 루프가 시작될 때마다 1씩 증가
        current_stage_num += 1

        print(f"\n--- 예선전 스테이지 {current_stage_num} (탐험 강도: {gauntlet_model.ent_coef:.4f}) ---")

        checkpoint_callback = CheckpointCallback(
          save_freq=CHECKPOINT_SAVE_FREQ,
          save_path=SAVE_DIR,
          name_prefix=f"gauntlet_checkpoint_stage_{current_stage_num}",
          save_replay_buffer=False,
          save_vecnormalize=True
        )

        # learn 호출은 한 번만 실행
        gauntlet_model.learn(
            total_timesteps=TIMESTEPS_PER_GAUNTLET_STAGE, 
            callback=[ProgressCallback(TIMESTEPS_PER_GAUNTLET_STAGE), checkpoint_callback], 
            reset_num_timesteps=False
        )
        
        current_total_timesteps = gauntlet_model.num_timesteps
        win_rate, win_as_black, win_as_white, reg_success, split_success, reg_ratio = evaluate_vs_model_c(gauntlet_model, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR)
        
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_stage_num, current_total_timesteps, f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}", f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}"])
        
        new_backup_filename = f"gauntlet_backup_stage_{current_stage_num}.zip"
        new_backup_path = os.path.join(SAVE_DIR, new_backup_filename)
        gauntlet_model.save(new_backup_path)
        stage_backup_paths.append(new_backup_path)
        
        if len(stage_backup_paths) > MAX_BACKUPS:
            oldest_backup_path = stage_backup_paths.pop(0)
            try: os.remove(oldest_backup_path)
            except OSError: pass

        if win_rate >= GAUNTLET_WIN_RATE_THRESHOLD:
            print(f"\n✅ 최종 예선전 통과! (승률 {win_rate:.2%})")
            training_state["stage_backup_paths"] = stage_backup_paths
            gauntlet_env.save(os.path.join(SAVE_DIR, "gauntlet_vec_normalize.pkl"))
            return gauntlet_model, current_total_timesteps, training_state
        else:
            print(f"\n--- 최종 조건 미달. 다음 스테이지 진행. ---")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    state = load_training_state() or {}

    # --- [복구 로직] ---
    # 상태 파일이 비어있지만, 실제 백업 파일이 폴더에 남아있는 경우 복구를 시도합니다.
    if not state.get("stage_backup_paths") and os.path.exists(SAVE_DIR):
        print("[INFO] 상태 파일이 비어있거나 백업 목록이 없습니다. 폴더에서 직접 백업을 찾아 복구를 시도합니다...")
        found_backups = []
        for f in os.listdir(SAVE_DIR):
            if f.startswith("gauntlet_backup_stage_") and f.endswith(".zip"):
                try:
                    # 파일 이름에서 스테이지 번호를 추출하여 정렬 기준으로 사용
                    stage_num = int(f.split('_')[-1].split('.')[0])
                    found_backups.append((stage_num, os.path.join(SAVE_DIR, f)))
                except (ValueError, IndexError):
                    continue
        
        if found_backups:
            found_backups.sort(key=lambda x: x[0]) # 스테이지 번호 순으로 정렬
            state["stage_backup_paths"] = [path for num, path in found_backups]
            
            # 가장 마지막 백업 모델에서 timesteps 정보를 읽어와 복원 시도
            latest_model_path = state["stage_backup_paths"][-1]
            try:
                print(f"[INFO] 가장 마지막 백업 '{os.path.basename(latest_model_path)}'에서 timesteps를 복구합니다.")
                temp_env = DummyVecEnv([make_env_fn()])
                temp_model = PPO.load(latest_model_path, env=temp_env)
                state["total_timesteps_so_far"] = temp_model.num_timesteps
                temp_env.close()
                print(f"[SUCCESS] 백업 파일 {len(found_backups)}개를 찾아 상태를 성공적으로 복구했습니다. (마지막 Timesteps: {state['total_timesteps_so_far']})")
            except Exception as e:
                print(f"[WARN] 백업 파일에서 timesteps 정보를 읽는 데 실패했습니다: {e}. Timesteps는 0으로 간주합니다.")
                state["total_timesteps_so_far"] = 0
        else:
            print("[INFO] 폴더에서 복구할 예선전 백업 파일을 찾지 못했습니다.")

    # 예선전이 이미 완료되었는지 확인 (상태 파일에 기록됨)
    if state.get("gauntlet_completed"):
        print("\n[INFO] 예선전이 이미 완료된 상태입니다. 프로그램을 종료합니다.")
        return

    # 예선전 시작 또는 이어하기
    total_timesteps_so_far = state.get("total_timesteps_so_far", 0)
    model_A, total_timesteps_so_far, returned_state = run_gauntlet_training(
        model_name="A",
        initial_timesteps=total_timesteps_so_far,
        training_state=state # 복구되었거나 비어있는 상태를 전달
    )

    if model_A is None:
        print("\n[ERROR] 예선전 통과에 실패했습니다. 프로그램을 종료합니다.")
        return

    # 예선전 성공 후 모델 저장 및 종료
    print("\n[INFO] 예선전 통과 완료. 최종 모델을 저장하고 프로그램을 종료합니다.")
    
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.zip")
    LATEST_A_PATH = os.path.join(SAVE_DIR, "latest_model_A.zip")
    LATEST_B_PATH = os.path.join(SAVE_DIR, "latest_model_B.zip")

    model_A.save(BEST_MODEL_PATH)
    model_A.save(LATEST_A_PATH)
    
    temp_env = DummyVecEnv([make_env_fn()])
    model_B = PPO.load(LATEST_A_PATH, env=temp_env)
    model_B.save(LATEST_B_PATH)
    temp_env.close()

    # 최종 상태 저장 (예선전 완료됨을 표시)
    returned_state["gauntlet_completed"] = True
    save_training_state(returned_state)
    
    print(f"\n✅ 예선전 완료. 모델이 '{SAVE_DIR}' 폴더에 저장되었습니다.")
    
if __name__ == "__main__":
    main()   # 기존 파이프라인을 쓸 땐 이걸 사용
    # train_and_eval_separated_policies()  # ← 분리 학습 파이프라인 실행

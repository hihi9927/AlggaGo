import os
import time
import re
import numpy as np
import torch
import pygame
import csv 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
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

# --- 진행률 표시 콜백 클래스 ---
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        # 실제 목표 타임스텝(= learn에 넘긴 total_timesteps)로 표시
        self.start_num = self.model.num_timesteps
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="학습 진행률",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def _on_step(self):
        # 현재 진행 타임스텝 = (모델 누적) - (학습 시작 시점)
        done_ts = self.model.num_timesteps - self.start_num
        # pbar 위치를 직접 맞춰줌
        if self.pbar:
            self.pbar.n = min(done_ts, self.total_timesteps)
            self.pbar.refresh()
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close(); self.pbar = None
# --- Rule-based 초기화 함수 ---
def initialize_to_rule_based(model):
    """
    모델의 정책을 '규칙 기반' 행동을 하도록 초기화합니다.
    - 전략: '일반 공격'을 압도적으로 선호하도록 설정
    - 파라미터: 규칙 기반의 값을 그대로 따르도록 오프셋을 0으로 설정
    """
    with torch.no_grad():
        # 신경망의 마지막 출력 레이어(action_net)를 가져옵니다.
        action_net = model.policy.action_net

        # action_net의 가중치는 모두 0으로 설정하여, 출력이 편향(bias)에 의해서만 결정되도록 합니다.
        action_net.weight.data.fill_(0.0)

        # 모델의 최종 출력은 5개의 값을 가집니다.
        # [0]: 일반 공격 선호도 (Regular Attack Preference)
        # [1]: 틈새 공격 선호도 (Split Shot Preference)
        # [2]: raw_index (돌 선택 오프셋)
        # [3]: raw_angle (각도 오프셋)
        # [4]: raw_force (힘 오프셋)

        # 1. 전략 선택 초기화
        # '일반 공격' 선호도는 매우 높게, '틈새 공격' 선호도는 매우 낮게 설정
        action_net.bias[0].data.fill_(10.0)  # 일반 공격 선호
        action_net.bias[1].data.fill_(-10.0) # 틈새 공격 비선호

        # 2. 파라미터 오프셋 초기화
        # 돌, 각도, 힘에 대한 수정값(오프셋)은 모두 0으로 설정
        action_net.bias[2].data.fill_(0.0)
        action_net.bias[3].data.fill_(0.0)
        action_net.bias[4].data.fill_(0.0)

        # 3. 행동의 분산(log_std)을 매우 작게 만들어, 초기 행동이 거의 일정하게 만듭니다.
        if isinstance(model.policy.log_std, torch.nn.Parameter):
            model.policy.log_std.data.fill_(-20.0)

# --- 모델 파라미터 확인 함수 ---
def print_model_parameters(model: PPO):
    print("\n==== 모델 초기 파라미터 상태 확인 ====")
    params = model.get_parameters()['policy']
    for name, tensor in params.items():
        if name in ["mlp_extractor", "value_net"] and isinstance(tensor, dict):
            for sub_name, sub_tensor in tensor.items():
                if hasattr(sub_tensor, 'cpu'):
                    arr = sub_tensor.cpu().numpy()
                    print(f"policy.{name}.{sub_name}: max={np.max(arr):.6f}, min={np.min(arr):.6f}")
        elif hasattr(tensor, 'cpu'):
            arr = tensor.cpu().numpy()
            print(f"policy.{name}: max={np.max(arr):.6f}, min={np.min(arr):.6f}")
    print("=" * 31, "\n")

# --- 환경 생성 헬퍼 함수 ---
def make_env_fn():
    def _init():
        env = AlggaGoEnv()
        monitored_env = Monitor(env, filename=LOG_DIR)
        return monitored_env
    return _init

# --- 기타 헬퍼 함수 ---
def clean_models(model_A_path, model_B_path, best_model_paths):
    if not os.path.exists(SAVE_DIR): return
    all_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".zip")]
    to_keep_names = {os.path.basename(p) for p in [model_A_path, model_B_path] if p} | {os.path.basename(p) for p in best_model_paths}
    for fname in all_files:
        if fname in to_keep_names: continue
        try:
            file_to_remove = os.path.join(SAVE_DIR, fname)
            if os.path.exists(file_to_remove): os.remove(file_to_remove)
        except OSError as e: print(f"[WARN] 파일 삭제 실패: {e}")
def update_best_models(current_best_models, new_model_path, reward, max_to_keep=5):
    current_best_models.append((new_model_path, reward))
    current_best_models.sort(key=lambda x: x[1], reverse=True)
    return current_best_models[:max_to_keep]
TRAINING_STATE_FILE = os.path.join(SAVE_DIR, "training_state.npy")
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
def print_overall_progress(current_stage, total_stages, current_timesteps, total_timesteps):
    stage_progress = (current_stage / total_stages) * 100
    timestep_progress = (current_timesteps / total_timesteps) * 100
    print(f"\n{'='*60}\n📊 전체 학습 진행률\n   스테이지: {current_stage}/{total_stages} ({stage_progress:.1f}%)\n   타임스텝: {current_timesteps:,}/{total_timesteps:,} ({timestep_progress:.1f}%)\n{'='*60}")

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

    # 평가 전 원래 엔트로피 저장 및 0으로 고정
    original_ent_A = model_A.ent_coef
    original_ent_B = model_B.ent_coef
    win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate = (0.5, 0.5, 0, 0)
    try:
        print("   [INFO] 평가를 위해 두 모델의 엔트로피를 0.0으로 고정합니다.")
        model_A.ent_coef = 0
        model_B.ent_coef = 0

        r1_black_win_rate = _play_round(model_A, model_B, games_per_round, "1라운드 (A가 흑돌)")
        print(f"   ▶ 1라운드 (Model A 흑돌) 승률: {r1_black_win_rate:.2%}")
        r2_black_win_rate = _play_round(model_B, model_A, games_per_round, "2라운드 (B가 흑돌)")
        print(f"   ▶ 2라운드 (Model B 흑돌) 승률: {r2_black_win_rate:.2%}")
        win_rate_A = (r1_black_win_rate + (1 - r2_black_win_rate)) / 2
        win_rate_B = (r2_black_win_rate + (1 - r1_black_win_rate)) / 2
    finally:
        # 평가 후 원래 엔트로피로 복원
        model_A.ent_coef = original_ent_A
        model_B.ent_coef = original_ent_B
        print("   [INFO] 원래 엔트로피 값으로 복원되었습니다.")

    return win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate

def evaluate_vs_model_c(ppo_model: PPO, num_episodes_per_color: int):
    """PPO 모델과 모델 C의 승률 및 각 전략의 성공률을 종합적으로 평가하는 함수"""
    print(f"   - 모델 C와 특별 평가: 총 {num_episodes_per_color * 2} 게임 (흑/백 각 {num_episodes_per_color}판)")
    env = AlggaGoEnv()
    from pymunk import Vec2d
    from physics import scale_force, all_stones_stopped

    win_rates = {}
    total_wins = 0
    
    strategy_attempts = {0: 0, 1: 0}
    strategy_successes = {0: 0, 1: 0}
    
    for side in ["black", "white"]:
        ppo_wins_on_side = 0
        desc = f"   PPO({side}) vs C"
        
        for _ in tqdm(range(num_episodes_per_color), desc=desc, leave=False):
            obs, _ = env.reset(options={"initial_player": side})
            done = False; info = {}
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
                    # ... (모델 C의 턴 로직은 기존과 동일) ...
                    absolute_action = model_c_action(env.stones, current_player_color)
                    if absolute_action:
                        idx, angle, force = absolute_action
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
        
        win_rate = ppo_wins_on_side / num_episodes_per_color if num_episodes_per_color > 0 else 0
        print(f"   ▶ PPO가 {side}일 때 승률: {win_rate:.2%}")
        win_rates[side] = win_rate
        total_wins += ppo_wins_on_side

    env.close()
    
    # --- [✅ 최종 수정] 모든 통계 지표 계산 ---
    overall_win_rate = total_wins / (num_episodes_per_color * 2) if num_episodes_per_color > 0 else 0
    win_rate_as_black = win_rates.get("black", 0)
    win_rate_as_white = win_rates.get("white", 0)
    
    regular_success_rate = strategy_successes[0] / strategy_attempts[0] if strategy_attempts[0] > 0 else 0
    split_success_rate = strategy_successes[1] / strategy_attempts[1] if strategy_attempts[1] > 0 else 0
    
    total_strategy_attempts = strategy_attempts[0] + strategy_attempts[1]
    regular_attack_ratio = strategy_attempts[0] / total_strategy_attempts if total_strategy_attempts > 0 else 0
    
    # --- 콘솔 출력 부분 ---
    print(f"   ▶ 모델 PPO 전체 승률 (vs C): {overall_win_rate:.2%}")
    print(f"   ▶ 일반 공격 선택 비율: {regular_attack_ratio:.2%}")
    print(f"   ▶ 일반 공격 성공률: {regular_success_rate:.2%} ({strategy_successes[0]}/{strategy_attempts[0]})")
    print(f"   ▶ 틈새 공격 성공률: {split_success_rate:.2%} ({strategy_successes[1]}/{strategy_attempts[1]})")
    
    # --- 반환 값 ---
    return (overall_win_rate, win_rate_as_black, win_rate_as_white, 
            regular_success_rate, split_success_rate, regular_attack_ratio)

class VsModelCEnv(gym.Env):
    """
    단일 PPO 에이전트가 고정 상대(Model C)와 번갈아 싸우며 학습하도록 래핑한 환경.
    한 번의 step() 호출에서:
      - PPO(에이전트)의 수를 env.step(action)으로 반영
      - 게임 미종료면, 곧바로 C의 수를 내부에서 실행
      - 다시 PPO 차례가 된 시점의 관측 obs, reward(에이전트 관점), done 등을 반환
    """
    metadata = {"render_modes": []}

    def __init__(self, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv()  # 기존 환경 재사용
        self.agent_side = agent_side  # 'black' or 'white'
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

        # 내부 상태 추적용
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        # PPO가 항상 먼저 두도록 시작 색을 강제
        initial_player = self.agent_side
        self._last_obs, info = self.base_env.reset(options={"initial_player": initial_player})
        # 혹시 시작 플레이어가 PPO가 아닌 경우엔, C가 먼저 한 수 두고 PPO 차례로 맞춰줌
        if self.base_env.current_player != self.agent_side:
            self._play_model_c_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info

    def step(self, action):
        # 1) 에이전트 수
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated:
            return obs, reward_agent, terminated, truncated, info

        # 2) 상대 수 (모델 C)
        self._play_model_c_turn()

        # 3) 종료/패널티 보정 및 다음 관측 반환
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_c_turn()
        total_reward = reward_agent + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info

    # ===== 내부 유틸 =====
    def _play_model_c_turn(self):
        # 현재 차례가 C인지 확인
        current_player_color = self.base_env.current_player
        if current_player_color == self.agent_side:
            return  # 이미 PPO 차례면 아무것도 안 함

        action_tuple = model_c_action(self.base_env.stones, current_player_color)
        if action_tuple:
            from pymunk import Vec2d
            from physics import scale_force, all_stones_stopped, WIDTH, HEIGHT, MARGIN
            idx, angle, force = action_tuple

            player_color_tuple = (0, 0, 0) if current_player_color == "black" else (255, 255, 255)
            player_stones = [s for s in self.base_env.stones if s.color[:3] == player_color_tuple]
            if 0 <= idx < len(player_stones):
                stone_to_shoot = player_stones[idx]
                direction = Vec2d(1, 0).rotated(angle)
                impulse = direction * scale_force(force)
                stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)

            # 물리 진행 (평가 코드와 동일 상한):contentReference[oaicite:2]{index=2}
            from physics import all_stones_stopped, WIDTH, HEIGHT, MARGIN
            physics_steps = 0
            while not all_stones_stopped(self.base_env.stones) and physics_steps < 600:
                self.base_env.space.step(1/60.0)
                physics_steps += 1

            # 바깥으로 나간 돌 제거
            for shape in self.base_env.stones[:]:
                x, y = shape.body.position
                if not (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
                    if shape in self.base_env.space.shapes:
                        self.base_env.space.remove(shape, shape.body)
                    if shape in self.base_env.stones:
                        self.base_env.stones.remove(shape)

            # 턴 전환 및 관측 업데이트
            self.base_env.current_player = "white" if current_player_color == "black" else "black"

    def _check_terminal_and_penalty_after_c_turn(self):
        """
        C 차례 진행 직후 종료 여부와 에이전트 관점 패널티를 계산.
        env.step 내부의 보상은 '수를 둔 쪽' 기준으로 산출되므로,
        C가 이겨서 끝난 경우 에이전트에 약한 패널티를 더해줌(-5.0).
        """
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0 and current_white > 0:
            # 백 승(흑 올아웃)
            winner = "white"
            terminated = True
        elif current_white == 0 and current_black > 0:
            winner = "black"
            terminated = True
        else:
            return False, 0.0

        # 에이전트 패배 시만 작은 패널티
        agent_color = self.agent_side
        if (winner == "white" and agent_color == "black") or (winner == "black" and agent_color == "white"):
            return True, -5.0
        return True, 0.0

class VsFixedOpponentEnv(gym.Env):
    """
    단일 PPO 에이전트가 '고정된 PPO 상대(opponent_model)'와 번갈아 싸우며 학습하는 환경.
    """
    metadata = {"render_modes": []}

    def __init__(self, opponent_model: PPO, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv()
        self.opponent = opponent_model
        self.agent_side = agent_side  # 'black' or 'white'
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        initial_player = self.agent_side
        self._last_obs, info = self.base_env.reset(options={"initial_player": initial_player})
        # 시작 차례가 에이전트가 아니면, 상대가 먼저 한 수 두고 에이전트 차례로 맞춤
        if self.base_env.current_player != self.agent_side:
            self._play_opponent_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info

    def step(self, action):
        # 1) 에이전트 수
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated:
            return obs, reward_agent, terminated, truncated, info

        # 2) 상대 수 (PPO)
        # [✅ 최종 수정] 함수의 반환값을 opp_reward 변수에 저장합니다.
        opp_reward = self._play_opponent_turn()

        # 3) 종료/패널티 보정 및 다음 관측 반환 (다시 에이전트 차례)
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_opponent()
        total_reward = (reward_agent - opp_reward) + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info

    # ===== 내부 유틸 =====
    def _play_opponent_turn(self):
        if self.base_env.current_player == self.agent_side:
            return 0.0  # 상대 턴이 아니면 보상 0 반환
        opp_obs = self.base_env._get_obs()
        opp_action, _ = self.opponent.predict(opp_obs, deterministic=True)
        # 상대방의 step 결과에서 reward 값을 받아옴
        _obs, opp_reward, _terminated, _truncated, _info = self.base_env.step(opp_action)
        return opp_reward
    
    def set_opponent(self, new_opponent_model: PPO):
        self.opponent = new_opponent_model

    def _check_terminal_and_penalty_after_opponent(self):
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0 and current_white > 0:
            winner = "white"; terminated = True
        elif current_white == 0 and current_black > 0:
            winner = "black"; terminated = True
        else:
            return False, 0.0

        # 에이전트가 진 경우만 작은 패널티
        if (winner == "white" and self.agent_side == "black") or \
           (winner == "black" and self.agent_side == "white"):
            return True, -5.0
        return True, 0.0
    
# --- Vs Model C 환경 생성 헬퍼 ---
def make_vs_c_env_vec(n_envs: int = 2):
    """
    PPO가 Model C와 번갈아 싸우며 학습하도록 흑/백을 섞은 VecEnv를 만듭니다.
    짝수 index -> black, 홀수 index -> white
    """
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsModelCEnv(agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

def make_vs_opponent_env_vec(opponent_model: PPO, n_envs: int = 2):
    """
    병렬 env 중 짝수 index는 agent=흑, 홀수 index는 agent=백으로 만들어
    학습 과정에서 색상이 균형되도록 함.
    """
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsFixedOpponentEnv(opponent_model=opponent_model, agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

def train_vs_model_c(total_timesteps=100_000, agent_side="black", ent_coef=0.1, save_name="ppo_vs_c"):
    """
    PPO 하나를 고정 상대(Model C)와 싸우며 학습하는 간단한 학습 루프.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 래퍼 환경 사용
    def _make():
        return VsModelCEnv(agent_side=agent_side)
    env = DummyVecEnv([_make])

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ent_coef)
    # (선택) 기존 규칙 기반 초기화 재사용 가능:contentReference[oaicite:4]{index=4}
    initialize_to_rule_based(model)

    print(f"[INFO] PPO vs Model C 학습 시작: total_timesteps={total_timesteps}, side={agent_side}, ent_coef={ent_coef}")
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(total_timesteps), reset_num_timesteps=False)

    save_path = os.path.join(SAVE_DIR, f"{save_name}_{agent_side}_{total_timesteps}.zip")
    model.save(save_path)
    print(f"[INFO] 학습 완료. 저장: {os.path.basename(save_path)}")
    return model, save_path

# --- 시각화 함수 ---
def visualize_one_game(model_A: PPO, model_B: PPO, ent_A: float, ent_B: float, stage_num: int, force_A_as_black: bool = None):
    """
    한 게임을 시각화합니다. (수정된 버전: PPO 턴의 물리 과정을 프레임별로 렌더링)
    force_A_as_black: True이면 A가 흑돌, False이면 B가 흑돌, None이면 엔트로피 기반으로 결정합니다.
    """
    stage_str = f"스테이지 {stage_num}" if stage_num > 0 else "초기 상태"
    
    if force_A_as_black is True:
        black_model, white_model = model_A, model_B
        caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
    elif force_A_as_black is False:
        black_model, white_model = model_B, model_A
        caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"
    else:
        if ent_A >= ent_B:
            black_model, white_model = model_A, model_B
            caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
        else:
            black_model, white_model = model_B, model_A
            caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"

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
        
        player_color_tuple = (0,0,0) if env.current_player == "black" else (255,255,255)
        player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
        opponent_stones = [s for s in env.stones if s.color[:3] != player_color_tuple]

        if not player_stones or not opponent_stones:
            done = True; continue

        if len(opponent_stones) < 2:
            strategy_choice = 0
        else:
            strategy_preferences = np.asarray(action_values[:2], dtype=np.float32)
            max_pref = float(np.max(strategy_preferences)); exp_p = np.exp(strategy_preferences - max_pref)
            probs = exp_p / (np.sum(exp_p) + 1e-8)
            strategy_choice = int(np.random.choice(2, p=probs)) if np.all(np.isfinite(probs)) and probs.sum() > 0 else int(np.argmax(strategy_preferences))
            
        chosen_str = '일반공격(0)' if strategy_choice == 0 else '스플릿샷(1)'
        print(f"[viz] {current_player} 턴 전략: {chosen_str}")

        rule_action = get_split_shot_action(player_stones, opponent_stones) if strategy_choice == 1 else get_regular_action(player_stones, opponent_stones)
        if rule_action is None: rule_action = get_regular_action(player_stones, opponent_stones)

        if rule_action:
            raw_index, raw_angle, raw_force = action_values[2:]
            raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
            raw_index = np.clip(raw_idx_val, -1.0, 1.0)
            raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
            raw_force = np.clip(raw_force_val, -1.0, 1.0)

            index_weight = raw_index * env.exploration_range['index']
            angle_offset = raw_angle * env.exploration_range['angle']
            force_offset = raw_force * env.exploration_range['force']
            rule_idx, rule_angle, rule_force = rule_action
            
            final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1) if len(player_stones) > 1 else 0
            final_angle = rule_angle + angle_offset
            final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
            
            selected_stone_to_shoot = player_stones[final_idx]
            direction = Vec2d(1, 0).rotated(final_angle)
            impulse = direction * scale_force(final_force)
            selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

            physics_steps = 0
            while not all_stones_stopped(env.stones) and physics_steps < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps += 1
        
        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

        current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
        
        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'

        if not done: env.current_player = "white" if env.current_player == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()


def visualize_vs_model_c(ppo_model: PPO, round_num: int, ppo_player_side: str):
    """
    PPO 모델과 모델 C의 대결을 시각화합니다. (수정된 버전: PPO 턴 렌더링 포함)
    ppo_player_side: PPO 모델이 플레이할 색상 ('black' 또는 'white')
    """
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
    from physics import scale_force, all_stones_stopped

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
            
            player_color_tuple = (0,0,0) if env.current_player == "black" else (255,255,255)
            player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
            opponent_stones = [s for s in env.stones if s.color[:3] != player_color_tuple]

            if not player_stones or not opponent_stones:
                done = True; continue

            if len(opponent_stones) < 2:
                strategy_choice = 0
            else:
                strategy_preferences = np.asarray(action_values[:2], dtype=np.float32)
                max_pref = float(np.max(strategy_preferences)); exp_p = np.exp(strategy_preferences - max_pref)
                probs = exp_p / (np.sum(exp_p) + 1e-8)
                strategy_choice = int(np.random.choice(2, p=probs)) if np.all(np.isfinite(probs)) and probs.sum() > 0 else int(np.argmax(strategy_preferences))
            chosen_str = '일반공격(0)' if strategy_choice == 0 else '스플릿샷(1)'
            print(f"[viz-vsC] PPO 턴 전략: {chosen_str}")

            rule_action = get_split_shot_action(player_stones, opponent_stones) if strategy_choice == 1 else get_regular_action(player_stones, opponent_stones)
            if rule_action is None: rule_action = get_regular_action(player_stones, opponent_stones)

            if rule_action:
                raw_index, raw_angle, raw_force = action_values[2:]

                raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
                raw_index = np.clip(raw_idx_val, -1.0, 1.0)
                raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
                raw_force = np.clip(raw_force_val, -1.0, 1.0)

                index_weight = raw_index * env.exploration_range['index']
                angle_offset = raw_angle * env.exploration_range['angle']
                force_offset = raw_force * env.exploration_range['force']
                rule_idx, rule_angle, rule_force = rule_action
                
                final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1) if len(player_stones) > 1 else 0
                final_angle = rule_angle + angle_offset
                final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
                
                selected_stone_to_shoot = player_stones[final_idx]
                direction = Vec2d(1, 0).rotated(final_angle)
                impulse = direction * scale_force(final_force)
                selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)
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
            env.space.step(1/60.0); env.render(screen=screen)
            pygame.display.flip(); pygame.time.delay(16)
            physics_steps += 1

        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

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
    """
    '틈새 공격' 전략만 사용하여 한 게임을 시각화하고, 매 턴의 성공/실패 여부를 터미널에 출력하는
    디버깅 전용 함수.
    """
    print("\n" + "="*50)
    print("🔬      '틈새 공격' 디버깅 시각화 시작      🔬")
    print("="*50)

    env = AlggaGoEnv()
    obs, _ = env.reset(options={"initial_player": "black"})
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DEBUG: Split Shot Only")

    from pymunk import Vec2d
    from physics import scale_force, all_stones_stopped
    import itertools

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

        if current_player_color == "black": # PPO 모델의 턴 (흑돌 고정)
            action_values, _ = model.predict(obs, deterministic=True)
            
            player_stones = [s for s in env.stones if s.color[:3] == (0,0,0)]
            opponent_stones = [s for s in env.stones if s.color[:3] == (255,255,255)]

            if not player_stones or not opponent_stones:
                done = True; continue

            strategy_choice = 1
            print(f"\n[DEBUG] PPO 턴: '틈새 공격' 강제 실행")

            rule_action = get_split_shot_action(player_stones, opponent_stones)
            if rule_action is None:
                print("[DEBUG] 틈새 공격 가능한 수가 없어 턴을 넘깁니다.")
                env.current_player = "white"
                obs = env._get_obs()
                continue
            
            raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
            raw_index = np.clip(raw_idx_val, -1.0, 1.0)
            raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
            raw_force = np.clip(raw_force_val, -1.0, 1.0)

            index_weight = raw_index * env.exploration_range['index']
            angle_offset = raw_angle * env.exploration_range['angle']
            force_offset = raw_force * env.exploration_range['force']
            rule_idx, rule_angle, rule_force = rule_action
            
            final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1)
            final_angle = rule_angle + angle_offset
            final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
            
            selected_stone_to_shoot = player_stones[final_idx]
            
            direction = Vec2d(1, 0).rotated(final_angle)
            impulse = direction * scale_force(final_force)
            selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

            physics_steps = 0
            while not all_stones_stopped(env.stones) and physics_steps < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps += 1

            moved_stone_final_pos = selected_stone_to_shoot.body.position
            opponent_stones_after_shot = [s for s in env.stones if s.color[:3] == (255,255,255)]
            
            max_wedge_reward = 0.0
            if len(opponent_stones_after_shot) >= 2:
                for o1, o2 in itertools.combinations(opponent_stones_after_shot, 2):
                    p1, p2 = o1.body.position, o2.body.position
                    p3 = moved_stone_final_pos
                    v = p2 - p1; w = p3 - p1
                    t = w.dot(v) / (v.dot(v) + 1e-6)
                    if 0 < t < 1:
                        dist_to_segment = (p3 - (p1 + t * v)).length
                        dist_between_opponents = (p1 - p2).length

                        # [✅ 최종 삭제] '너무 넓은 틈새는 무시'하는 조건문을 삭제합니다.
                        wedge_threshold = dist_between_opponents * 0.15
                        if dist_to_segment < wedge_threshold:
                            current_reward = (1 - (dist_to_segment / wedge_threshold)) * 0.5 
                            if current_reward > max_wedge_reward:
                                max_wedge_reward = current_reward

            if max_wedge_reward > 0:
                print(f"   [DEBUG] >> 결과: 틈새 공격 성공! (보상: {max_wedge_reward:.2f})")
            else:
                print(f"   [DEBUG] >> 결과: 틈새 공격 실패.")

        else: # 모델 C의 턴
            print(f"\n[DEBUG] 모델 C 턴...")
            action_tuple = model_c_action(env.stones, current_player_color)
            if action_tuple:
                idx, angle, force = action_tuple
                player_stones_c = [s for s in env.stones if s.color[:3] == (255,255,255)]
                if 0 <= idx < len(player_stones_c):
                    stone_to_shoot_c = player_stones_c[idx]
                    direction_c = Vec2d(1, 0).rotated(angle)
                    impulse_c = direction_c * scale_force(force)
                    stone_to_shoot_c.body.apply_impulse_at_world_point(impulse_c, stone_to_shoot_c.body.position)
            
            physics_steps_c = 0
            while not all_stones_stopped(env.stones) and physics_steps_c < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps_c += 1

        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

        current_black = sum(1 for s in env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'

        if not done: env.current_player = "white" if current_player_color == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f"\n>>> 디버깅 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(3)
    pygame.quit()

def run_gauntlet_training(model_name, initial_timesteps, training_state):
    """
    지시된 '스테이지-회차-라운드' 규칙에 따라 예선전을 진행하는 새로운 메인 함수.
    '새 학습 시작 시 초기 평가' 로직이 복원되었습니다.
    
    Args:
        model_name (str): 훈련 대상 모델의 이름 ('A' 또는 'B').
        initial_timesteps (int): 현재까지의 누적 타임스텝.
        training_state (dict): 백업 경로 리스트 등을 포함한 전체 훈련 상태.

    Returns:
        tuple: (성공한 모델 객체, 최종 누적 타임스텝, 갱신된 훈련 상태)
               스테이지 최종 실패 시 (None, 최종 누적 타임스텝, 갱신된 훈련 상태) 반환.
    """
    print("\n" + "="*60)
    print(f"🥊      새로운 규칙 기반 예선전 시작: 모델 {model_name} vs 모델 C       🥊")
    print("="*60)

    # --- 1. 상수 및 상태 변수 초기화 ---
    MAX_ATTEMPTS_PER_STAGE = 5
    MAX_ROUNDS_PER_ATTEMPT = 3
    TIMESTEPS_PER_ROUND = 30000
    SUCCESS_THRESHOLD = 0.8
    MAX_BACKUPS = 5

    stage_backup_paths = training_state.get("stage_backup_paths", [])
    current_total_timesteps = initial_timesteps
    
    GAUNTLET_LOG_FILE = os.path.join(LOG_DIR, "gauntlet_log.csv")
    if not os.path.exists(GAUNTLET_LOG_FILE):
        with open(GAUNTLET_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Stage", "Attempt", "Round", "Total Timesteps", "Win Rate as Black", "Win Rate as White", "Overall Win Rate", "Regular Success", "Split Success", "Regular Ratio"])

    N_ENVS_VS_C = 2
    gauntlet_env_raw = make_vs_c_env_vec(n_envs=N_ENVS_VS_C)
    gauntlet_env = VecNormalize(gauntlet_env_raw, norm_obs=True, norm_reward=True)

    # --- 2. 스테이지 시작 모델 결정 및 초기 평가 ---
    if stage_backup_paths:
        stage_start_model_path = stage_backup_paths[-1]
        print(f"\n[INFO] 가장 최근 백업 '{os.path.basename(stage_start_model_path)}'에서 스테이지를 시작합니다.")
    else:
        stage_start_model_path = None
        print("\n[INFO] 사용 가능한 백업이 없습니다. 모델을 규칙 기반으로 초기화하여 처음부터 시작합니다.")
        
        # --- [수정] 새 학습 시작 시, 훈련 전 초기 평가 로직 추가 ---
        print("\n--- 훈련 시작 전, 규칙 기반 초기 상태 종합 평가 시작 ---")
        # 평가를 위한 임시 모델 생성 및 초기화
        temp_model = PPO("MlpPolicy", gauntlet_env, verbose=0)
        initialize_to_rule_based(temp_model)
        temp_model.ent_coef = 0.0

        (win_rate, win_as_black, win_as_white, 
         reg_success, split_success, reg_ratio) = evaluate_vs_model_c(
         temp_model, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR
        )
        
        # 초기 상태 로그 기록 (스테이지 1, 0회차, 0라운드로 기록)
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([1, 0, 0, current_total_timesteps,
                             f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}",
                             f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}"])
        print("[INFO] 규칙 기반 초기 상태의 평가 결과가 로그에 기록되었습니다.")
        del temp_model # 평가에 사용된 임시 모델은 삭제
        # --- [수정] 로직 추가 끝 ---

    stage_succeeded = False
    final_model = None

    # --- 3. 회차(Attempt) 루프 (최대 5회) ---
    for attempt in range(1, MAX_ATTEMPTS_PER_STAGE + 1):
        print(f"\n{'='*20} 스테이지 {len(stage_backup_paths) + 1} / {attempt}회차 시작 {'='*20}")

        if stage_start_model_path:
            attempt_model = PPO.load(stage_start_model_path, env=gauntlet_env, device="auto")
        else:
            attempt_model = PPO("MlpPolicy", gauntlet_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A, max_grad_norm=0.5, learning_rate=0.0001)
            initialize_to_rule_based(attempt_model)

        # --- 4. 라운드(Round) 루프 (최대 3회) ---
        for round_num in range(1, MAX_ROUNDS_PER_ATTEMPT + 1):
            print(f"\n--- {attempt}회차 / {round_num}라운드 훈련 시작 (타임스텝: {TIMESTEPS_PER_ROUND:,}) ---")
            
            attempt_model.ent_coef = INITIAL_ENT_COEF_A # 훈련 시에는 엔트로피 복원
            attempt_model.learn(
                total_timesteps=TIMESTEPS_PER_ROUND,
                callback=ProgressCallback(TIMESTEPS_PER_ROUND),
                reset_num_timesteps=False
            )
            current_total_timesteps = attempt_model.num_timesteps

            print("\n--- 훈련 후 종합 평가 시작 ---")
            attempt_model.ent_coef = 0.0
            (win_rate, win_as_black, win_as_white, 
             reg_success, split_success, reg_ratio) = evaluate_vs_model_c(
             attempt_model, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR
            )
            
            with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([len(stage_backup_paths) + 1, attempt, round_num, current_total_timesteps,
                                 f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}",
                                 f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}"])

            # --- 5. 성공 조건 검사 ---
            if reg_success >= SUCCESS_THRESHOLD and split_success >= SUCCESS_THRESHOLD:
                print("\n" + "🏆" * 25)
                print(f"✅ 스테이지 성공! ({attempt}회차, {round_num}라운드 만에 통과)")
                print(f"   (일반 공격: {reg_success:.2%}, 틈새 공격: {split_success:.2%})")
                print("🏆" * 25)

                stage_succeeded = True
                final_model = attempt_model
                
                new_backup_filename = f"backup_stage_{len(stage_backup_paths) + 1}.zip"
                new_backup_path = os.path.join(SAVE_DIR, new_backup_filename)
                final_model.save(new_backup_path)
                print(f"\n[백업] 새로운 성공 모델을 '{new_backup_filename}'으로 저장했습니다.")

                stage_backup_paths.append(new_backup_path)
                if len(stage_backup_paths) > MAX_BACKUPS:
                    oldest_backup_path = stage_backup_paths.pop(0)
                    try:
                        os.remove(oldest_backup_path)
                        print(f"[백업] 가장 오래된 백업 '{os.path.basename(oldest_backup_path)}'을(를) 삭제했습니다.")
                    except OSError as e:
                        print(f"[WARN] 백업 파일 삭제 실패: {e}")
                
                break
            else:
                print(f"\n--- 성공 조건 미달. 다음 라운드를 진행합니다. (기준: {SUCCESS_THRESHOLD:.0%}) ---")
        
        if stage_succeeded:
            break
        else:
            print(f"\n--- {attempt}회차 실패 (3라운드 내 성공 못함). 학습 내용을 폐기하고 다음 회차를 시작합니다. ---")

    # --- 6. 스테이지 최종 실패 처리 ---
    if not stage_succeeded:
        print("\n" + "❌" * 25)
        print(f"❌ 스테이지 최종 실패! (5회차 내에 통과 못함)")
        print("❌" * 25)
        
        if stage_backup_paths:
            last_backup_to_discard = stage_backup_paths.pop()
            try:
                os.remove(last_backup_to_discard)
                print(f"\n[백업] 스테이지 실패로 인해, 이전 성공 백업이었던 '{os.path.basename(last_backup_to_discard)}'을(를) 폐기합니다.")
            except OSError as e:
                print(f"[WARN] 백업 파일 폐기 실패: {e}")
        else:
            print("\n[백업] 폐기할 이전 백업이 없습니다.")

    # --- 7. 최종 결과 반환 ---
    training_state["stage_backup_paths"] = stage_backup_paths
    
    return final_model, current_total_timesteps, training_state

def reload_with_env(model: PPO, new_env):
    """현재 모델 파라미터를 보존한 채, env 개수를 바꾸기 위해 저장-재로딩."""
    tmp = os.path.join(SAVE_DIR, "_tmp_reload_swap_env.zip")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(tmp)
    new_model = PPO.load(tmp, env=new_env, device=model.device)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return new_model

def run_final_evaluation(champion_model: PPO, env):
    """
    최종 챔피언 모델의 성능을 여러 엔트로피 레벨의 상대와 비교하여 평가합니다.
    """
    print("\n" + "="*35)
    print("🏆      최종 챔피언 모델 성능 평가      🏆")
    print("="*35 + "\n")
    
    opponent_ent_coefs = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = []
    
    # 챔피언 모델을 임시 저장하여 깨끗한 상태의 상대를 로드하기 위함
    champion_path = os.path.join(SAVE_DIR, "champion_model_final.zip")
    champion_model.save(champion_path)

    print(f"[*] 챔피언 모델: {os.path.basename(champion_model.logger.get_dir())} 에서 저장됨")
    print("[*] 평가 상대: 챔피언과 동일한 모델 (엔트로피만 0.0 ~ 0.4로 고정)")
    print("-" * 35)

    for ent_coef in opponent_ent_coefs:
        print(f"\n[평가] vs Opponent (ent_coef: {ent_coef:.1f})")
        
        # 챔피언의 복사본을 상대로 로드
        opponent_model = PPO.load(champion_path, env=env)
        opponent_model.ent_coef = ent_coef
        
        # 100판씩 공정 평가 진행
        win_rate_champion, _, _, _ = evaluate_fairly(
            champion_model, opponent_model, num_episodes=100
        )
        
        results.append((ent_coef, win_rate_champion))
        print(f"  ▶ 챔피언 승률: {win_rate_champion:.2%}")

    print("\n\n" + "="*30)
    print("      📊 최종 평가 결과 요약 📊")
    print("="*30)
    print(f"{'상대 엔트로피':<15} | {'챔피언 승률':<15}")
    print("-" * 30)
    for ent_coef, win_rate in results:
        print(f"{ent_coef:<15.1f} | {win_rate:<15.2%}")
    print("="*30)
    
    # 임시 저장된 챔피언 모델 삭제
    if os.path.exists(champion_path):
        os.remove(champion_path)

# --- 경쟁적 학습 메인 함수 ---
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    temp_env = DummyVecEnv([make_env_fn()])
    
    BEST_MODEL_FILENAME = "best_model.zip"

    # 로그 파일 경로 설정
    MAIN_LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
    GAUNTLET_LOG_FILE = os.path.join(LOG_DIR, "gauntlet_log.csv")

    # 로그 파일 헤더 초기화
    if not os.path.exists(MAIN_LOG_FILE):
        with open(MAIN_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Stage", "Total Timesteps", "Model A Entropy", "Model B Entropy", "Round 1 Win Rate (A Black)", "Round 2 Win Rate (B Black)"])
    if not os.path.exists(GAUNTLET_LOG_FILE):
        with open(GAUNTLET_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Total Timesteps", "Win Rate as Black", "Win Rate as White", "Overall Win Rate", "Neg Strategy Ratio"])
    
    # 상태 로드 또는 새로 시작
    state = load_training_state() or {}
    total_timesteps_so_far = state.get("total_timesteps_so_far", 0)
    current_ent_coef_A = state.get("current_ent_coef_A", INITIAL_ENT_COEF_A)
    current_ent_coef_B = state.get("current_ent_coef_B", INITIAL_ENT_COEF_B)
    best_overall_models = state.get("best_overall_models", [])
    
    # --- [수정] 역대 최고 승률 상태 로드 ---
    best_win_rate = state.get("best_win_rate", 0.0)
    #split_shot_threshold = state.get("split_shot_threshold", 0.5)
    
    model_A, model_B = None, None
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")

    try:
        # 기존 모델 로드 (이 부분은 그대로 둡니다)
        models_found = {"a": [], "b": []}
        if os.path.exists(SAVE_DIR):
            for f in os.listdir(SAVE_DIR):
                match = model_pattern.match(f)
                if match: models_found[match.group(1)].append((int(match.group(2)), os.path.join(SAVE_DIR, f)))
        if not models_found["a"]: raise FileNotFoundError("학습된 A 모델 없음")
        
        latest_a_path = max(models_found["a"], key=lambda i: i[0])[1]
        latest_b_path = max(models_found["b"], key=lambda i: i[0])[1] if models_found["b"] else latest_a_path
        
        print(f"[INFO] 학습 이어하기: Model A({os.path.basename(latest_a_path)}), Model B({os.path.basename(latest_b_path)}) 로드")
        model_A = PPO.load(latest_a_path, env=temp_env)
        model_B = PPO.load(latest_b_path, env=temp_env)

        model_timesteps = max(model_A.num_timesteps, model_B.num_timesteps)
        if model_timesteps > total_timesteps_so_far:
            print(f"[WARN] 모델의 타임스텝({model_timesteps:,})이 상태 파일({total_timesteps_so_far:,})보다 최신입니다. 모델 기준으로 동기화합니다.")
            total_timesteps_so_far = model_timesteps
        
        print("\n[INFO] 현재 로드된 모델의 상태를 시각화합니다...")
        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_num=0)

    except Exception as e:
        print(f"[INFO] 새 학습 시작 ({e}).")
        
        # --- [수정된 부분 시작] ---
        # 새 학습 시작 시, 비어있는 상태(state)로 시작합니다.
        state = {} 
        total_timesteps_so_far = 0

        # run_gauntlet_training 함수를 새로운 규격에 맞게 호출합니다.
        # model_to_train 인자 대신 training_state를 전달합니다.
        model_A, total_timesteps_so_far, state = run_gauntlet_training(
            model_name="A",
            initial_timesteps=total_timesteps_so_far,
            training_state=state
        )

        # 예선전 통과 모델이 없을 경우 (초기화 실패 등) 프로그램 종료
        if model_A is None:
            print("[ERROR] 예선전에서 모델을 생성하지 못했습니다. 프로그램을 종료합니다.")
            return

        # 예선전 통과 후 model_B 생성
        print("\n[INFO] 예선을 통과한 모델 A를 복제하여 모델 B를 다시 동기화합니다...")
        post_gauntlet_a_path = os.path.join(SAVE_DIR, "model_a_post_gauntlet.zip")
        model_A.save(post_gauntlet_a_path)
        
        model_B = PPO.load(post_gauntlet_a_path, env=temp_env)
        print("[INFO] 모델 B 동기화 완료.")
        
        # 예선전 이후의 상태 변수들을 state 딕셔너리에서 가져오거나 초기값으로 설정
        current_ent_coef_A = state.get("current_ent_coef_A", INITIAL_ENT_COEF_A)
        current_ent_coef_B = state.get("current_ent_coef_B", INITIAL_ENT_COEF_B)
        best_win_rate = state.get("best_win_rate", 0.0)
        '''''''''''
        try:
            params = model_A.get_parameters()
            params['policy']['action_net.weight'].data.fill_(0)
            params['policy']['action_net.bias'].data.fill_(0)
            model_A.set_parameters(params)
            print("[INFO] 추가 초기화(action_net->0) 성공.")
        except KeyError:
            print("[경고] 모델 구조를 찾지 못해 추가 초기화에 실패했습니다.")
        print_model_parameters(model_A)
        '''''''''''

    # --- 메인 학습 루프 ---
    start_stage = total_timesteps_so_far // TIMESTEPS_PER_STAGE if TIMESTEPS_PER_STAGE > 0 else 0
    total_expected_timesteps = MAX_STAGES * TIMESTEPS_PER_STAGE

    VEC_NORMALIZE_STATS_PATH = os.path.join(SAVE_DIR, "vec_normalize.pkl")

    # VecNormalize 환경을 생성합니다.
    # 상대 모델은 루프 안에서 계속 바뀌므로, 임시 상대로 먼저 초기화합니다.
    temp_opponent_model = model_B if model_B is not None else model_A
    train_env_raw = make_vs_opponent_env_vec(opponent_model=temp_opponent_model, n_envs=2)
    train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True)

    # 저장된 정규화 상태가 있으면 불러옵니다.
    if os.path.exists(VEC_NORMALIZE_STATS_PATH):
        print(f"[INFO] VecNormalize 상태 로드: {VEC_NORMALIZE_STATS_PATH}")
        train_env = VecNormalize.load(VEC_NORMALIZE_STATS_PATH, train_env_raw)
        # VecEnv는 내부적으로 환경을 다시 만들므로, 래핑을 다시 해줘야 합니다.
        train_env.norm_obs = True
        train_env.norm_reward = True

    for stage_idx in range(start_stage, MAX_STAGES):
        if total_timesteps_so_far >= total_expected_timesteps: break
        print_overall_progress(stage_idx + 1, MAX_STAGES, total_timesteps_so_far, total_expected_timesteps)
        print(f"\n--- 스테이지 {stage_idx + 1}/{MAX_STAGES} 시작 ---")
        stage_start_time = time.time()
        
        current_training_model_name = "A" if current_ent_coef_A >= current_ent_coef_B else "B"
        model_to_train, ent_coef_train = (model_A, current_ent_coef_A) if current_training_model_name == "A" else (model_B, current_ent_coef_B)
        
        model_to_train.ent_coef = ent_coef_train
        opponent_model = model_B if current_training_model_name == "A" else model_A

        # [✅ 최종 수정] 아래 3줄만 남기고 나머지는 삭제합니다.
        train_env.env_method("set_opponent", opponent_model)
        model_to_train.set_env(train_env)
        print(f"   학습 대상: Model {current_training_model_name} (ent_coef: {ent_coef_train:.5f})")
        
        model_to_train.learn(total_timesteps=TIMESTEPS_PER_STAGE, callback=ProgressCallback(TIMESTEPS_PER_STAGE), reset_num_timesteps=False)
        total_timesteps_so_far = model_to_train.num_timesteps
        if current_training_model_name == "A": model_A = model_to_train
        else: model_B = model_to_train

        print(f"\n   --- 경쟁 평가 시작 ---")
        win_rate_A, win_rate_B, r1_win_rate, r2_win_rate = evaluate_fairly(model_A, model_B, num_episodes=EVAL_EPISODES_FOR_COMPETITION)
        
        with open(MAIN_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            log_data = [stage_idx + 1, total_timesteps_so_far, f"{current_ent_coef_A:.5f}", f"{current_ent_coef_B:.5f}", f"{r1_win_rate:.4f}", f"{r2_win_rate:.4f}"]
            writer.writerow(log_data)
        print("   [INFO] 학습 결과가 CSV 로그 파일에 기록되었습니다.")

        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_idx + 1, force_A_as_black=(current_training_model_name == 'A'))
        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_idx + 1, force_A_as_black=(current_training_model_name != 'A'))

        print("\n   --- 엔트로피 및 최고 모델 결정 ---")
        if win_rate_A > win_rate_B: effective_winner, winner_win_rate = "A", win_rate_A
        elif win_rate_B > win_rate_A: effective_winner, winner_win_rate = "B", win_rate_B
        else:
            effective_winner = "B" if current_training_model_name == "A" else "A"
            winner_win_rate = win_rate_B if effective_winner == "B" else win_rate_A
        
        if win_rate_A == win_rate_B: print(f"   경쟁 결과: 무승부. 학습 대상({current_training_model_name})이 패배한 것으로 간주하여 Model {effective_winner} 승리.")
        else: print(f"   경쟁 결과: Model {effective_winner} 승리 (승률: {winner_win_rate:.2%})")

        # --- [수정] 최고 성능 모델 저장 로직 ---
        champion_model = model_A if effective_winner == "A" else model_B
        BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_MODEL_FILENAME)
        if winner_win_rate > best_win_rate:
            print(f"   🚀 새로운 최고 승률 달성! (이전: {best_win_rate:.2%} -> 현재: {winner_win_rate:.2%})")
            print(f"   '{BEST_MODEL_FILENAME}' 파일을 업데이트합니다.")
            champion_model.save(BEST_MODEL_PATH)
            best_win_rate = winner_win_rate  # 최고 승률 갱신
        else:
            print(f"   최고 승률({best_win_rate:.2%})을 넘지 못했습니다. (현재: {winner_win_rate:.2%})")

        FINAL_EVAL_ENT_THRESHOLD = 0.45
        should_terminate = False
        model_to_requalify, model_to_requalify_name = None, ""

        if effective_winner == "A" and current_ent_coef_A > current_ent_coef_B:
            new_ent_coef_B = min(current_ent_coef_B + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            if new_ent_coef_B != current_ent_coef_B:
                print(f"   Model B 엔트로피 증가 → {new_ent_coef_B:.5f}")
                current_ent_coef_B, model_to_requalify, model_to_requalify_name = new_ent_coef_B, model_B, "B"
            if new_ent_coef_B >= FINAL_EVAL_ENT_THRESHOLD:
                run_final_evaluation(champion_model=model_A, env=temp_env); should_terminate = True

        elif effective_winner == "B" and current_ent_coef_B > current_ent_coef_A:
            new_ent_coef_A = min(current_ent_coef_A + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            if new_ent_coef_A != current_ent_coef_A:
                print(f"   Model A 엔트로피 증가 → {new_ent_coef_A:.5f}")
                current_ent_coef_A, model_to_requalify, model_to_requalify_name = new_ent_coef_A, model_A, "A"
            if new_ent_coef_A >= FINAL_EVAL_ENT_THRESHOLD:
                run_final_evaluation(champion_model=model_B, env=temp_env); should_terminate = True
        else:
            print("   엔트로피 조정 없음")

        if model_to_requalify:
            trained_model, total_timesteps_so_far = run_gauntlet_training(
                model_to_train=model_to_requalify, 
                model_name=model_to_requalify_name, 
                initial_timesteps=total_timesteps_so_far
            )
            if model_to_requalify_name == "A": model_A = trained_model
            else: model_B = trained_model

        model_A_path = os.path.join(SAVE_DIR, f"model_a_{total_timesteps_so_far}_{current_ent_coef_A:.3f}.zip")
        model_A.save(model_A_path)
        best_overall_models = update_best_models(best_overall_models, model_A_path, win_rate_A)
        
        model_B_path = os.path.join(SAVE_DIR, f"model_b_{total_timesteps_so_far}_{current_ent_coef_B:.3f}.zip")
        model_B.save(model_B_path)

        # [✅ 추가] VecNormalize 상태 저장
        train_env.save(VEC_NORMALIZE_STATS_PATH)
        print(f" 💾  VecNormalize 상태를 {os.path.basename(VEC_NORMALIZE_STATS_PATH)} 파일에 저장했습니다.")
        
        clean_models(model_A_path, model_B_path, [m[0] for m in best_overall_models])
        
        # --- [수정] 상태 저장 시 최고 승률 포함 ---
        current_state = {
            "total_timesteps_so_far": total_timesteps_so_far, "current_ent_coef_A": current_ent_coef_A,
            "current_ent_coef_B": current_ent_coef_B, "best_overall_models": best_overall_models,
            "best_win_rate": best_win_rate,
            #"split_shot_threshold": split_shot_threshold 
        }
        save_training_state(current_state)
        
        minutes, seconds = divmod(int(time.time() - stage_start_time), 60)
        print(f"\n[스테이지 {stage_idx + 1}] 완료 (소요 시간: {minutes}분 {seconds}초)")

        if should_terminate:
            print("\n--- 최종 평가 완료. 학습을 종료합니다. ---")
            break

    print("\n--- 전체 경쟁적 학습 완료 ---")
    temp_env.close()

if __name__ == "__main__":
    main()
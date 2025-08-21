import os
import time
import re
import numpy as np
import torch
import pygame
import csv 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from opponent_c import model_c_action
from opponent import get_regular_action, get_split_shot_action 

import gymnasium as gym
from gymnasium import spaces

from env import AlggaGoEnv
from physics import WIDTH, HEIGHT, all_stones_stopped, MARGIN

# --- 경로 유틸 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(BASE_DIR, *parts)

# --- 하이퍼파라미터 및 설정 ---
MAX_STAGES = 300
TIMESTEPS_PER_STAGE = 50000
SAVE_DIR = rel_path("rl_models_competitive")
LOG_DIR = rel_path("rl_logs_competitive")
INITIAL_ENT_COEF_A = 0.15
INITIAL_ENT_COEF_B = 0.1
ENT_COEF_INCREMENT = 0.1
MAX_ENT_COEF = 0.5
EVAL_EPISODES_FOR_COMPETITION = 200

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
def initialize_to_rule_based(model, angle_value=-1.57, force_value=1.0):
    with torch.no_grad():
        policy_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net
        for layer in policy_net:
            if hasattr(layer, 'weight'): layer.weight.fill_(0.0)
            if hasattr(layer, 'bias'): layer.bias.fill_(0.0)
        for i in range(action_net.out_features):
            action_net.weight[i].fill_(0.0)
            action_net.bias[i].fill_(angle_value if i == 0 else force_value)
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
        model_A.ent_coef = 0.0
        model_B.ent_coef = 0.0

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
    """PPO 모델과 모델 C의 승률을 흑/백 각각 평가하고 상세 결과를 반환하는 함수"""
    print(f"   - 모델 C와 특별 평가: 총 {num_episodes_per_color * 2} 게임 (흑/백 각 {num_episodes_per_color}판)")
    env = AlggaGoEnv()
    from pymunk import Vec2d
    from physics import scale_force, all_stones_stopped

    win_rates = {}
    total_wins = 0

    ppo_turn_count = 0
    neg_strategy_count = 0
    
    for side in ["black", "white"]:
        ppo_wins_on_side = 0
        desc = f"   PPO({side}) vs C"
        
        for _ in tqdm(range(num_episodes_per_color), desc=desc, leave=False):
            obs, _ = env.reset(options={"initial_player": side})
            done = False
            info = {}
            while not done:
                current_player_color = env.current_player
                
                if current_player_color == side:  # PPO 모델의 턴
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    strategy_choice = np.squeeze(action)[0]
                    if strategy_choice < 0:
                        neg_strategy_count += 1
                    ppo_turn_count += 1
                    obs, _, done, _, info = env.step(action)
                else:  # 모델 C의 턴 (직접 실행)
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
                        env.space.step(1/60.0)
                        physics_steps += 1

                    for shape in env.stones[:]:
                        if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                            env.space.remove(shape, shape.body)
                            env.stones.remove(shape)
                    
                    current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
                    current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
                    
                    if current_black == 0:
                        done = True; info['winner'] = 'white'
                    elif current_white == 0:
                        done = True; info['winner'] = 'black'

                    env.current_player = "white" if env.current_player == "black" else "black"
                    obs = env._get_obs()

            if done and info.get('winner') == side:
                ppo_wins_on_side += 1
        
        win_rate = ppo_wins_on_side / num_episodes_per_color if num_episodes_per_color > 0 else 0
        print(f"   ▶ PPO가 {side}일 때 승률: {win_rate:.2%}")
        win_rates[side] = win_rate
        total_wins += ppo_wins_on_side

    env.close()
    overall_win_rate = total_wins / (num_episodes_per_color * 2) if num_episodes_per_color > 0 else 0
    neg_strategy_ratio = neg_strategy_count / ppo_turn_count if ppo_turn_count > 0 else 0
    print(f"   ▶ 모델 PPO 전체 승률 (vs C): {overall_win_rate:.2%}")
    print(f"   ▶ 일반 공격(-1) 선택 비율: {neg_strategy_ratio:.2%}")
    
    return overall_win_rate, win_rates.get("black", 0), win_rates.get("white", 0), neg_strategy_ratio

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
        """
        입력 action은 항상 PPO(에이전트) 차례의 행동으로 가정.
        이후 C의 차례까지 내부에서 진행하여, 다시 PPO 차례가 되었을 때의 obs를 반환.
        """
        # --- 1) PPO(에이전트) 차례: 기존 env.step 사용 ---
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)

        # 에피소드가 끝났으면 그대로 종료
        if terminated or truncated:
            return obs, reward_agent, terminated, truncated, info

        # --- 2) Model C 차례: 기존 평가 로직을 참고하여 직접 1수 진행 ---
        # (evaluate_vs_model_c / visualize_vs_model_c와 동일 로직을 축약 적용):contentReference[oaicite:1]{index=1}
        self._play_model_c_turn()

        # --- 3) 종료 체크 및 관측 반환(이제 다시 PPO 차례) ---
        # C의 수로 끝났다면 에이전트 관점에서 패배 페널티를 약간 더해도 됨
        # (env.step에서 승/패 보상 일부가 에이전트 턴에만 반영되므로 보정)
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
        self._play_opponent_turn()

        # 3) 종료/패널티 보정 및 다음 관측 반환 (다시 에이전트 차례)
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_opponent()
        total_reward = reward_agent + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info

    # ===== 내부 유틸 =====
    def _play_opponent_turn(self):
        if self.base_env.current_player == self.agent_side:
            return  # 지금은 에이전트 차례면 아무 것도 안 함
        # 현재 상태 관측으로 상대 정책 실행 → 그대로 env.step
        opp_obs = self.base_env._get_obs()
        opp_action, _ = self.opponent.predict(opp_obs, deterministic=True)
        self.base_env.step(opp_action)  # 보상은 상대 기준이므로 여기선 사용하지 않음

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
    한 게임을 시각화합니다.
    force_A_as_black: True이면 A가 흑돌, False이면 B가 흑돌, None이면 엔트로피 기반으로 결정합니다.
    """
    stage_str = f"스테이지 {stage_num}" if stage_num > 0 else "초기 상태"
    
    # 흑돌/백돌 모델 및 캡션 결정
    if force_A_as_black is True:
        black_model, white_model = model_A, model_B
        caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
    elif force_A_as_black is False:
        black_model, white_model = model_B, model_A
        caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"
    else: # 기본 동작 (엔트로피 기반)
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
    from physics import scale_force
    from pymunk import Vec2d
    from opponent import get_regular_action, get_split_shot_action

    done = False; step_count = 0; max_steps = 200; info = {}
    while not done and step_count < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue
        current_player = env.current_player
        action_model = black_model if current_player == "black" else white_model

        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)

        action_values, _ = action_model.predict(obs.reshape(1, -1), deterministic=True)
        strategy_choice, index_weight, angle_offset, force_offset = np.squeeze(action_values)

        player_stones = [s for s in env.stones if s.color[:3] == ((0,0,0) if current_player == "black" else (255,255,255))]
        opponent_stones = [s for s in env.stones if s.color[:3] != ((0,0,0) if current_player == "black" else (255,255,255))]

        if not player_stones or not opponent_stones: break

        if strategy_choice >= 0:
            rule_action = get_split_shot_action(player_stones, opponent_stones)
        else:
            rule_action = get_regular_action(player_stones, opponent_stones)
        
        if rule_action is None:
            rule_action = get_regular_action(player_stones, opponent_stones)
        
        if rule_action is None: break

        rule_idx, rule_angle, rule_force = rule_action
        
        if len(player_stones) > 1:
            idx_offset = int(np.round(index_weight * (len(player_stones) - 1) / 2))
            final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
        else: final_idx = 0
        
        final_angle = rule_angle + angle_offset
        final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
        
        selected_stone = player_stones[final_idx]
        direction = Vec2d(1, 0).rotated(final_angle)
        impulse = direction * scale_force(final_force)
        
        selected_stone.body.apply_impulse_at_world_point(impulse, selected_stone.body.position)
        
        physics_steps = 0
        while not all_stones_stopped(env.stones) and physics_steps < 300:
            env.space.step(1/60.0)
            env.render(screen=screen)
            pygame.display.flip()
            time.sleep(1/60.0)
            physics_steps += 1
        
        current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'
        
        env.current_player = "white" if current_player == "black" else "black"
        obs = env._get_obs()
        step_count += 1
        
    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()

def visualize_vs_model_c(ppo_model: PPO, round_num: int, ppo_player_side: str):
    """
    PPO 모델과 모델 C의 대결을 시각화합니다.
    ppo_player_side: PPO 모델이 플레이할 색상 ('black' 또는 'white')
    """
    stage_str = f"특별 훈련 {round_num}라운드"
    
    if ppo_player_side == "black":
        caption = f"{stage_str}: 모델 A(흑돌) vs 모델 C(백돌)"
    else:
        caption = f"{stage_str}: 모델 C(흑돌) vs 모델 A(백돌)"

    print(f"\n--- {stage_str} 시각화 ({caption}) ---")

    env = AlggaGoEnv()
    # 게임은 항상 흑돌부터 시작하도록 고정합니다.
    obs, _ = env.reset(options={"initial_player": "black"})
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)
    
    from pymunk import Vec2d
    from physics import scale_force, all_stones_stopped
    from opponent import get_regular_action, get_split_shot_action

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

        action_tuple = None
        # 현재 턴 플레이어가 PPO 모델이 플레이해야 하는 색상과 일치하는지 확인합니다.
        if current_player_color == ppo_player_side: # PPO 모델의 턴
            action_values, _ = ppo_model.predict(obs, deterministic=True)
            strategy_choice, index_weight, angle_offset, force_offset = np.squeeze(action_values)

            player_color_tuple = (0,0,0) if ppo_player_side == "black" else (255,255,255)
            opponent_color_tuple = (255,255,255) if ppo_player_side == "black" else (0,0,0)

            player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
            opponent_stones = [s for s in env.stones if s.color[:3] == opponent_color_tuple]

            if not player_stones or not opponent_stones: break
            
            if strategy_choice >= 0:
                rule_action = get_split_shot_action(player_stones, opponent_stones)
            else:
                rule_action = get_regular_action(player_stones, opponent_stones)

            if rule_action is None:
                rule_action = get_regular_action(player_stones, opponent_stones)

            if rule_action:
                rule_idx, rule_angle, rule_force = rule_action
                if len(player_stones) > 1:
                    idx_offset = int(np.round(index_weight * (len(player_stones) - 1) / 2))
                    final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
                else: final_idx = 0
                
                final_angle = rule_angle + angle_offset
                final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
                action_tuple = (final_idx, final_angle, final_force)

        else: # 모델 C의 턴
            action_tuple = model_c_action(env.stones, current_player_color)

        if action_tuple:
            idx, angle, force = action_tuple
            player_color_tuple = (0,0,0) if current_player_color == "black" else (255,255,255)
            player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
            
            if 0 <= idx < len(player_stones):
                stone_to_shoot = player_stones[idx]
                direction = Vec2d(1, 0).rotated(angle)
                impulse = direction * scale_force(force)
                stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)
        
        physics_steps = 0
        while not all_stones_stopped(env.stones) and physics_steps < 600:
            env.space.step(1/60.0)
            env.render(screen=screen)
            pygame.display.flip()
            time.sleep(1/60.0)
            physics_steps += 1
        
        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes:
                    env.space.remove(shape, shape.body)
                if shape in env.stones:
                    env.stones.remove(shape)
        
        current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
        if current_black == 0:
            done = True; info['winner'] = 'white'
        elif current_white == 0:
            done = True; info['winner'] = 'black'
        
        env.current_player = "white" if current_player_color == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()

def run_gauntlet_training(model_to_train, model_name, initial_timesteps):
    """
    주어진 모델이 모델 C를 이길 때까지 훈련하는 예선전 함수.
    """
    print("\n" + "="*50)
    print(f"🥊      특별 예선 시작: 모델 {model_name} vs 모델 C       🥊")
    print("="*50)

    GAUNTLET_LOG_FILE = os.path.join(LOG_DIR, "gauntlet_log.csv")
    GAUNTLET_SAVE_PATH = os.path.join(SAVE_DIR, f"model_{model_name.lower()}_gauntlet_in_progress.zip")

    if os.path.exists(GAUNTLET_SAVE_PATH):
        print(f"\n[INFO] 진행 중이던 예선전 모델({os.path.basename(GAUNTLET_SAVE_PATH)})을 로드하여 이어갑니다.")
        # 현재 env와 device 설정을 그대로 사용하여 모델을 로드합니다.
        model_to_train = PPO.load(GAUNTLET_SAVE_PATH, env=model_to_train.get_env(), device=model_to_train.device)
        initial_timesteps = model_to_train.num_timesteps
        print(f"[INFO] 로드된 모델의 누적 타임스텝: {initial_timesteps:,}")
    else:
        print(f"\n[INFO] 모델 {model_name}에 대한 새 예선전을 시작합니다.")
    
    original_ent_coef = model_to_train.ent_coef
    model_to_train.ent_coef = 0.0 # 평가 시에는 엔트로피 0으로 고정
    print(f"\n[INFO] 예선 평가를 위해 모델 {model_name}의 엔트로피를 0으로 설정합니다.")

    GAUNTLET_TIMESTEPS = 50000
    GAUNTLET_EVAL_EPISODES_PER_COLOR = 100
    N_ENVS_VS_C = 2  # 흑/백 번갈아

    # 필요시 원래 env를 복원할 수 있도록 백업(선택)
    original_env = getattr(model_to_train, "env", None)
    
    gauntlet_round = 1
    current_total_timesteps = initial_timesteps

    while True:
        # [수정] 함수가 반환하는 3개의 값을 각각의 변수로 받도록 수정
        overall_win_rate, win_rate_as_black, win_rate_as_white, neg_strategy_ratio = evaluate_vs_model_c(model_to_train, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR)
        
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{model_name}_Round_{gauntlet_round}", current_total_timesteps, f"{win_rate_as_black:.4f}", f"{win_rate_as_white:.4f}", f"{overall_win_rate:.4f}", f"{neg_strategy_ratio:.4f}"])
        print("   [INFO] 예선전 결과가 CSV 로그 파일에 기록되었습니다.")

        # --- 듀얼 시각화 평가 (vs Model C) ---
        print("\n[INFO] 예선전 시각화 평가를 시작합니다 (흑돌/백돌 각각 1회).")
        # 1. PPO 모델이 흑돌일 때
        visualize_vs_model_c(model_to_train, round_num=gauntlet_round, ppo_player_side="black")
        # 2. PPO 모델이 백돌일 때
        visualize_vs_model_c(model_to_train, round_num=gauntlet_round, ppo_player_side="white")

        if overall_win_rate > 0.5:
            print(f"\n🏆 모델 {model_name}이(가) 모델 C를 상대로 승리했습니다! 예선을 통과합니다. 🏆")
            model_to_train.ent_coef = original_ent_coef
            # (선택) 원래 env 복원
            if original_env is not None:
                model_to_train.set_env(original_env)
            if os.path.exists(GAUNTLET_SAVE_PATH):
                os.remove(GAUNTLET_SAVE_PATH)
            break
        else:
            print(f"   - 전체 승률({overall_win_rate:.2%})이 50% 미만입니다. 모델 {model_name}을(를) **Model C와 싸우며** 추가 훈련합니다.")
            # 학습 때는 엔트로피 복구
            model_to_train.ent_coef = original_ent_coef

            # ✅ 여기! 학습 환경을 VsModelCEnv로 바꿉니다.
            train_env = make_vs_c_env_vec(n_envs=N_ENVS_VS_C)
            model_to_train = reload_with_env(model_to_train, train_env)

            model_to_train.learn(
                total_timesteps=GAUNTLET_TIMESTEPS,
                callback=ProgressCallback(GAUNTLET_TIMESTEPS),
                reset_num_timesteps=False
            )
            current_total_timesteps = model_to_train.num_timesteps
            model_to_train.save(GAUNTLET_SAVE_PATH)
            print(f" 💾  예선 훈련 진행 상황을 {os.path.basename(GAUNTLET_SAVE_PATH)} 파일에 덮어썼습니다.")
            model_to_train.ent_coef = 0.0 # 다음 평가를 위해 다시 0으로 설정
        gauntlet_round += 1
    
    return model_to_train, current_total_timesteps

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
    model_A, model_B = None, None
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")

    try:
        # 기존 모델 로드
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
    except Exception as e:
        # [수정] 원래의 정상 작동하던 초기화 순서로 복원
        print(f"[INFO] 새 학습 시작 ({e}).")
        model_A = PPO("MlpPolicy", temp_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A, max_grad_norm=0.5)
        print("[INFO] 모델을 Rule-based 정책으로 초기화합니다...")
        initialize_to_rule_based(model_A)
        print("[INFO] 정책을 rule-based 형태로 강제 초기화 완료")
        try:
            params = model_A.get_parameters()
            params['policy']['action_net.weight'].data.fill_(0)
            params['policy']['action_net.bias'].data.fill_(0)
            model_A.set_parameters(params)
            print("[INFO] 추가 초기화(action_net->0) 성공.")
        except KeyError:
            print("[경고] 모델 구조를 찾지 못해 추가 초기화에 실패했습니다.")
        print_model_parameters(model_A)
        
        total_timesteps_so_far = 0

        # 초기 예선전은 A모델만 진행
        model_A, total_timesteps_so_far = run_gauntlet_training(model_A, "A", total_timesteps_so_far)
        
        # 예선 통과한 A를 다시 저장하고 B를 그것으로 업데이트 (선택사항, 더 공정한 시작을 위해)
        print("\n[INFO] 예선을 통과한 모델 A를 복제하여 모델 B를 다시 동기화합니다...")
        post_gauntlet_a_path = os.path.join(SAVE_DIR, "model_a_post_gauntlet.zip")
        model_A.save(post_gauntlet_a_path)
        model_B = PPO.load(post_gauntlet_a_path, env=temp_env)
        print("[INFO] 모델 B 동기화 완료.")


    # --- 메인 학습 루프 ---
    start_stage = total_timesteps_so_far // TIMESTEPS_PER_STAGE if TIMESTEPS_PER_STAGE > 0 else 0
    total_expected_timesteps = MAX_STAGES * TIMESTEPS_PER_STAGE

    for stage_idx in range(start_stage, MAX_STAGES):
        if total_timesteps_so_far >= total_expected_timesteps: break
        print_overall_progress(stage_idx + 1, MAX_STAGES, total_timesteps_so_far, total_expected_timesteps)
        print(f"\n--- 스테이지 {stage_idx + 1}/{MAX_STAGES} 시작 ---")
        stage_start_time = time.time()
        
        current_training_model_name = "A" if current_ent_coef_A >= current_ent_coef_B else "B"
        model_to_train, ent_coef_train = (model_A, current_ent_coef_A) if current_training_model_name == "A" else (model_B, current_ent_coef_B)
        
        model_to_train.ent_coef = ent_coef_train
        opponent_model = model_B if current_training_model_name == "A" else model_A
        train_env = make_vs_opponent_env_vec(opponent_model=opponent_model, n_envs=2)
        model_to_train = reload_with_env(model_to_train, train_env)
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

        # --- 듀얼 시각화 평가 ---
        # 이번 스테이지에서 학습한 모델이 A인지 B인지 결정
        trained_model_is_A = True if current_training_model_name == "A" else False

        # 1. 학습 모델이 흑돌로 플레이
        print("\n[INFO] 시각화 (1/2): 학습 모델이 흑돌일 때 플레이")
        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_idx + 1, force_A_as_black=trained_model_is_A)

        # 2. 학습 모델이 백돌로 플레이
        print("\n[INFO] 시각화 (2/2): 학습 모델이 백돌일 때 플레이")
        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_idx + 1, force_A_as_black=(not trained_model_is_A))

        print("\n   --- 엔트로피 조정 ---")
        if win_rate_A > win_rate_B: effective_winner = "A"
        elif win_rate_B > win_rate_A: effective_winner = "B"
        else: effective_winner = "B" if current_training_model_name == "A" else "A"
        
        if win_rate_A == win_rate_B: print(f"   경쟁 결과: 무승부. 학습 대상({current_training_model_name})이 패배한 것으로 간주하여 Model {effective_winner} 승리.")
        else: print(f"   경쟁 결과: Model {effective_winner} 승리")

        FINAL_EVAL_ENT_THRESHOLD = 0.45
        should_terminate = False

        model_to_requalify = None
        model_to_requalify_name = ""

        if effective_winner == "A" and current_ent_coef_A > current_ent_coef_B:
            new_ent_coef_B = min(current_ent_coef_B + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            if new_ent_coef_B != current_ent_coef_B:
                print(f"   Model B 엔트로피 증가 → {new_ent_coef_B:.5f}")
                current_ent_coef_B = new_ent_coef_B
                model_to_requalify = model_B
                model_to_requalify_name = "B"
            if new_ent_coef_B >= FINAL_EVAL_ENT_THRESHOLD:
                run_final_evaluation(champion_model=model_A, env=temp_env)
                should_terminate = True

        elif effective_winner == "B" and current_ent_coef_B > current_ent_coef_A:
            new_ent_coef_A = min(current_ent_coef_A + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            if new_ent_coef_A != current_ent_coef_A:
                print(f"   Model A 엔트로피 증가 → {new_ent_coef_A:.5f}")
                current_ent_coef_A = new_ent_coef_A
                model_to_requalify = model_A
                model_to_requalify_name = "A"
            if new_ent_coef_A >= FINAL_EVAL_ENT_THRESHOLD:
                run_final_evaluation(champion_model=model_B, env=temp_env)
                should_terminate = True
        else:
            print("   엔트로피 조정 없음")

        if model_to_requalify:
            trained_model, total_timesteps_so_far = run_gauntlet_training(model_to_requalify, model_to_requalify_name, total_timesteps_so_far)
            if model_to_requalify_name == "A":
                model_A = trained_model
            else:
                model_B = trained_model

        model_A_path = os.path.join(SAVE_DIR, f"model_a_{total_timesteps_so_far}_{current_ent_coef_A:.3f}.zip")
        model_A.save(model_A_path)
        best_overall_models = update_best_models(best_overall_models, model_A_path, win_rate_A)
        
        model_B_path = os.path.join(SAVE_DIR, f"model_b_{total_timesteps_so_far}_{current_ent_coef_B:.3f}.zip")
        model_B.save(model_B_path)
        
        clean_models(model_A_path, model_B_path, [m[0] for m in best_overall_models])
        
        current_state = {
            "total_timesteps_so_far": total_timesteps_so_far, "current_ent_coef_A": current_ent_coef_A,
            "current_ent_coef_B": current_ent_coef_B, "best_overall_models": best_overall_models
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
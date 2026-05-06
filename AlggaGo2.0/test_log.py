import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# 필요한 기존 파일들
from env import AlggaGoEnv
from opponent_c import model_c_action
from physics import all_stones_stopped, MARGIN, WIDTH, HEIGHT
import gymnasium as gym

# --- train.py에서 가져온 핵심 초기화 함수 ---
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

# --- [최종 수정] 로직 결함을 완전히 수정한 콜백 ---
class GameLogAndStopCallback(BaseCallback):
    def __init__(self, log_file, max_games_to_log, verbose=0):
        super(GameLogAndStopCallback, self).__init__(verbose)
        self.log_file = log_file
        self.max_games_to_log = max_games_to_log
        self.games_logged = 0
        self.log_header = ['경우의 수', '게임 기록', '총보상', '결과']
        self.seen_games = set()
        
        with open(self.log_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(self.log_header)

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self.agent_logs = [[] for _ in range(num_envs)]
        self.agent_rewards = [0.0 for _ in range(num_envs)]

    def _on_step(self) -> bool:
        for i in range(self.training_env.num_envs):
            # --- 1. 매 스텝마다 에이전트의 행동과 보상을 임시 기록 ---
            info = self.locals['infos'][i]
            reward = self.locals['rewards'][i]
            agent_side = self.training_env.get_attr('agent_side', i)[0]
            
            # 콜백은 에이전트의 행동 직후에만 호출되므로, 항상 에이전트의 로그를 기록
            log_entry = self._create_log_entry(info, reward)
            self.agent_logs[i].append(log_entry)
            self.agent_rewards[i] += reward
            
            # --- 2. 게임이 끝났는지(done) 확인 ---
            if self.locals['dones'][i]:
                # 흑돌 에이전트의 게임이었는지 확인하고, 로그 수집 목표에 도달하지 않았는지 확인
                if agent_side == 'black' and self.games_logged < self.max_games_to_log:
                    total_reward = self.agent_rewards[i]
                    log_tuple = tuple(self.agent_logs[i])
                    game_signature = (log_tuple, round(total_reward, 2))

                    # 중복되지 않은 새로운 기록일 경우에만 파일에 저장
                    if game_signature not in self.seen_games:
                        self.seen_games.add(game_signature)
                        
                        winner = info.get('winner')
                        outcome = "승리" if winner == agent_side else "패배"
                        full_log_str = " / ".join(log_tuple)
                        
                        with open(self.log_file, 'a', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(['', full_log_str, f"{total_reward:.2f}", outcome])
                        
                        self.games_logged += 1
                
                # 다음 게임을 위해 해당 환경의 임시 로그와 보상을 초기화
                self.agent_logs[i] = []
                self.agent_rewards[i] = 0.0

        # 목표한 게임 수를 모두 기록했으면 훈련 중단
        if self.games_logged >= self.max_games_to_log:
            print(f"\n[INFO] 목표한 {self.max_games_to_log}개의 고유한 흑돌 로그 기록 완료. 훈련을 중단합니다.")
            return False
        return True
        
    def _create_log_entry(self, info, reward):
        strategy_num = info.get('strategy_choice')
        strategy_text = "틈새 공격" if strategy_num == 1 else "일반 공격"
        is_success = info.get('is_regular_success', False) or info.get('is_split_success', False)
        outcome_text = "성공" if is_success else "실패"
        return f"{strategy_text}({outcome_text}: {reward:.2f})"

# --- 학습 환경 정의 (train.py 기반) ---
class VsModelCEnv(gym.Env):
    def __init__(self, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv()
        self.agent_side = agent_side
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
    @property
    def current_player(self): return self.base_env.current_player
    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(options={"initial_player": self.agent_side})
        if self.base_env.current_player != self.agent_side:
            self._play_model_c_turn()
            obs = self.base_env._get_obs()
        return obs, info
    def step(self, action):
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated: return obs, reward_agent, terminated, truncated, info
        self._play_model_c_turn()
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_c_turn()
        total_reward = reward_agent + loser_penalty
        obs = self.base_env._get_obs()
        return obs, total_reward, terminated_now, False, info
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
        self.base_env.current_player = "white" if self.base_env.current_player == "black" else "black"
    def _check_terminal_and_penalty_after_c_turn(self):
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: winner = "white"; terminated = True
        elif current_white == 0: winner = "black"; terminated = True
        else: return False, 0.0
        if (winner == "white" and self.agent_side == "black") or (winner == "black" and self.agent_side == "white"): return True, -5.0
        return True, 0.0

def make_vs_c_env_vec(n_envs: int = 4):
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: Monitor(VsModelCEnv(agent_side=side))
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

# --- 메인 실행 함수 ---
def main(num_games_to_log):
    LOG_FILENAME = "game_log.csv"
    N_ENVS = 4

    env = make_vs_c_env_vec(n_envs=N_ENVS)
    model = PPO("MlpPolicy", env, verbose=0)
    
    print("[INFO] 모델을 생성하고 규칙 기반으로 초기화합니다 (train.py와 동일한 초기 성능)")
    initialize_to_rule_based(model)
    
    log_and_stop_callback = GameLogAndStopCallback(log_file=LOG_FILENAME, max_games_to_log=num_games_to_log)
    
    try:
        print(f"\n[INFO] {num_games_to_log}개의 고유한 흑돌 게임 로그 기록을 시작합니다...")
        model.learn(total_timesteps=10_000_000, callback=log_and_stop_callback, progress_bar=True)
    finally:
        if os.path.exists(LOG_FILENAME):
            print(f"\n[INFO] '{LOG_FILENAME}' 파일의 내용을 총보상 높은 순으로 정렬합니다...")
            try:
                df = pd.read_csv(LOG_FILENAME)
                if not df.empty:
                    df['총보상'] = pd.to_numeric(df['총보상'], errors='coerce')
                    df.dropna(subset=['총보상'], inplace=True)
                    df_sorted = df.sort_values(by='총보상', ascending=False)
                    df_sorted['경우의 수'] = np.arange(1, len(df_sorted) + 1)
                    df_sorted.to_csv(LOG_FILENAME, index=False, encoding='utf-8-sig')
                    print("[INFO] 정렬 완료.")
                else:
                    print("[INFO] 기록된 로그가 없어 정렬을 건너뜁니다.")
            except Exception as e:
                print(f"[ERROR] CSV 파일 정렬 중 오류 발생: {e}")

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("[오류] pandas 라이브러리가 필요합니다. 'pip install pandas'를 실행해주세요.")
        sys.exit(1)

    DEFAULT_NUM_GAMES = 1000
    num_games = DEFAULT_NUM_GAMES
    if len(sys.argv) > 1:
        try: num_games = int(sys.argv[1])
        except ValueError: print(f"[경고] 잘못된 숫자 입력. 기본값 {DEFAULT_NUM_GAMES}판을 기록합니다.")
    
    main(num_games)
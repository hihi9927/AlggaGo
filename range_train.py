# range_training.py

import os
import math
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 로컬 파일 임포트
from env import AlggaGoEnv

# 하이퍼파라미터 및 설정
MAX_STAGES = 200
TIMESTEPS_PER_STAGE = 50000
SAVE_DIR = "range_rl_models"
EVAL_EPISODES_FOR_COMPETITION = 100

# 행동 범위 설정
INITIAL_ANGLE_RANGE = 0 #0.05
ANGLE_RANGE_INCREMENT = 0 #0.05
MAX_ANGLE_RANGE = np.pi / 2
INITIAL_FORCE_RANGE = 0 #0.1
FORCE_RANGE_INCREMENT = 0 #0.05
MAX_FORCE_RANGE = 0.5
INITIAL_INDEX_RANGE = 0 #0.2
INDEX_RANGE_INCREMENT = 0 #0.1
MAX_INDEX_RANGE = 1.0

# 환경 생성 헬퍼
def make_env_fn():
    def _init():
        return AlggaGoEnv()
    return _init

# 공정 평가
def evaluate_fairly(model_A: PPO, model_B: PPO, env: DummyVecEnv, num_episodes: int):
    env.envs[0].set_exploration_range(MAX_INDEX_RANGE, MAX_ANGLE_RANGE, MAX_FORCE_RANGE)
    model_A_wins = 0
    num_episodes_per_role = math.ceil(num_episodes / 2)

    for _ in range(num_episodes_per_role):
        obs = env.reset()
        done = False
        while not done:
            action, _ = (model_A if env.envs[0].current_player == 'white' else model_B).predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
        if infos[0].get('winner') == 'white': model_A_wins += 1
    
    for _ in range(num_episodes_per_role):
        obs = env.reset()
        done = False
        while not done:
            action, _ = (model_A if env.envs[0].current_player == 'black' else model_B).predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
        if infos[0].get('winner') == 'black': model_A_wins += 1

    return model_A_wins / num_episodes

# 메인 학습 함수
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = DummyVecEnv([make_env_fn()])

    model_A_path = os.path.join(SAVE_DIR, "model_a.zip")
    model_B_path = os.path.join(SAVE_DIR, "model_b.zip")

    if os.path.exists(model_A_path) and os.path.exists(model_B_path):
        print("[INFO] 저장된 모델 로드")
        model_A = PPO.load(model_A_path, env=env)
        model_B = PPO.load(model_B_path, env=env)
    else:
        print("[INFO] 새 모델 생성")
        model_A = PPO("MlpPolicy", env, verbose=0)
        model_B = PPO("MlpPolicy", env, verbose=0)
        model_B.set_parameters(model_A.get_parameters())

    ranges_A = {"index": INITIAL_INDEX_RANGE, "angle": INITIAL_ANGLE_RANGE, "force": INITIAL_FORCE_RANGE}
    ranges_B = {"index": INITIAL_INDEX_RANGE, "angle": INITIAL_ANGLE_RANGE, "force": INITIAL_FORCE_RANGE}

    for stage_idx in range(MAX_STAGES):
        print(f"\n--- 스테이지 {stage_idx + 1}/{MAX_STAGES} 시작 ---")

        high_range_model_is_A = ranges_A['angle'] >= ranges_B['angle']
        if high_range_model_is_A:
            model_to_train, ranges_to_use, model_name = model_A, ranges_A, "A"
        else:
            model_to_train, ranges_to_use, model_name = model_B, ranges_B, "B"

        print(f"  학습 대상: Model {model_name}")
        print(f"  적용 범위: Idx(±{ranges_to_use['index']:.2f}), Angle(±{ranges_to_use['angle']:.2f}), Force(±{ranges_to_use['force']:.2f})")
        env.envs[0].set_exploration_range(ranges_to_use['index'], ranges_to_use['angle'], ranges_to_use['force'])
        model_to_train.learn(total_timesteps=TIMESTEPS_PER_STAGE, reset_num_timesteps=False)

        # 평가
        win_rate_A = evaluate_fairly(model_A, model_B, env, num_episodes=EVAL_EPISODES_FOR_COMPETITION)
        print(f"  - 평가 결과: Model A 승률 {win_rate_A:.2%}")

        model_A_won = win_rate_A > 0.5

        if (high_range_model_is_A and model_A_won) or (not high_range_model_is_A and not model_A_won):
            # 규칙 1: 범위가 넓은 모델이 이긴 경우
            if high_range_model_is_A: # A가 넓고 A가 이김
                print("  규칙 1 적용: 넓은 범위(A) 승리 -> 좁은 범위(B) 확장")
                ranges_B['index'] = min(ranges_A['index'] + INDEX_RANGE_INCREMENT, MAX_INDEX_RANGE)
                ranges_B['angle'] = min(ranges_A['angle'] + ANGLE_RANGE_INCREMENT, MAX_ANGLE_RANGE)
                ranges_B['force'] = min(ranges_A['force'] + FORCE_RANGE_INCREMENT, MAX_FORCE_RANGE)
            else: # B가 넓고 B가 이김
                print("  규칙 1 적용: 넓은 범위(B) 승리 -> 좁은 범위(A) 확장")
                ranges_A['index'] = min(ranges_B['index'] + INDEX_RANGE_INCREMENT, MAX_INDEX_RANGE)
                ranges_A['angle'] = min(ranges_B['angle'] + ANGLE_RANGE_INCREMENT, MAX_ANGLE_RANGE)
                ranges_A['force'] = min(ranges_B['force'] + FORCE_RANGE_INCREMENT, MAX_FORCE_RANGE)
        else:
            # 규칙 2: 범위가 좁은 모델이 이긴 경우
            print("  규칙 2 적용: 좁은 범위 승리 -> 범위 변경 없음")
            pass
        
        model_A.save(model_A_path)
        model_B.save(model_B_path)

if __name__ == "__main__":
    main()
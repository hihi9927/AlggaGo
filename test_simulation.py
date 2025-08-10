import os
import pygame
import numpy as np
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from test_env import AlggaGoEnv
from physics import all_stones_stopped, WIDTH, HEIGHT, MARGIN

TOTAL_CYCLES = 100
STEPS_PER_CYCLE = 10000
EVAL_GAMES_PER_CYCLE = 100
MODEL_A_ENTROPY = 0.3

def initialize_to_rule_based(model, angle_value=-1.57, force_value=1.0):
    with torch.no_grad():
        policy_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net

        for layer in policy_net:
            if hasattr(layer, 'weight'):
                layer.weight.fill_(0.0)
            if hasattr(layer, 'bias'):
                layer.bias.fill_(0.0)

        for i in range(action_net.out_features):
            action_net.weight[i].fill_(0.0)
            action_net.bias[i].fill_(
                angle_value if i == 0 else force_value
            )

        if isinstance(model.policy.log_std, torch.nn.Parameter):
            model.policy.log_std.data.fill_(-20.0)


def evaluate_model(model, num_games, render=False, warmup_mode=False):
    eval_env = DummyVecEnv([lambda: AlggaGoEnv()])
    results = []
    MAX_STEPS_PER_EPISODE = 200

    for game_idx in tqdm(range(num_games), desc="Evaluating"):
        obs = eval_env.reset()
        done = False
        step_count = 0

        while not done:
            current_player = eval_env.envs[0].current_player
            if current_player == 'black':
                action, _ = model.predict(obs, deterministic=True) if not warmup_mode else np.array([[0.0, 0.0, 0.0]])
            else:
                action = np.array([[0.0, 0.0, 0.0]])

            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            step_count += 1

            if render:
                eval_env.envs[0].render()
                pygame.time.wait(100)

            if step_count >= MAX_STEPS_PER_EPISODE:
                done = True

        winner = infos[0].get('winner')
        if winner == 'black':
            results.append('B')
        elif winner == 'white':
            results.append('W')
        else:
            results.append('-')

    eval_env.close()

    win_count = results.count('B')
    return win_count

def main():
    print("=" * 50)
    print("누적 학습 및 주기적 평가를 시작합니다.")
    print(f"총 사이클: {TOTAL_CYCLES}, 사이클 당 학습: {STEPS_PER_CYCLE} 스텝, 사이클 당 평가: {EVAL_GAMES_PER_CYCLE} 게임")
    print("=" * 50)

    train_env = DummyVecEnv([lambda: AlggaGoEnv()])
    final_model_path = "final_model.zip"

    if os.path.exists(final_model_path):
        print(f"[INFO] 저장된 모델 '{final_model_path}'을 로드하여 학습을 이어갑니다.")
        model_A = PPO.load(final_model_path, env=train_env)
    else:
        print("[INFO] 새 학습을 시작합니다.")
        model_A = PPO("MlpPolicy", train_env, verbose=0, ent_coef=MODEL_A_ENTROPY)
        print("[INFO] 모델을 Rule-based 정책(모든 오차 0)으로 초기화합니다...")
        model_A = PPO("MlpPolicy", train_env, verbose=0, ent_coef=MODEL_A_ENTROPY)

    initialize_to_rule_based(model_A)
    print("[INFO] 정책을 rule-based 형태로 강제 초기화 완료")

    print("==== 모델 초기 파라미터 상태 확인 ====")
    params = model_A.get_parameters()

    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            else:
                flat[full_key] = v
        return flat

    flat_params = flatten_params(params)

    for name, tensor in flat_params.items():
        if hasattr(tensor, 'cpu') and hasattr(tensor, 'detach'):
            arr = tensor.cpu().detach().numpy()
            print(f"{name}: max={np.max(arr):.6f}, min={np.min(arr):.6f}")
        else:
            print(f"{name}: (non-numeric type: {type(tensor)})")

    try:
        params = model_A.get_parameters()
        params['policy']['action_net.weight'].data.fill_(0)
        params['policy']['action_net.bias'].data.fill_(0)
        model_A.set_parameters(params)
        print("[INFO] 초기화 성공.")
    except KeyError:
        print("[경고] 모델 구조를 찾지 못해 초기화에 실패했습니다.")


    for cycle in range(TOTAL_CYCLES):

        total_steps = model_A.num_timesteps + STEPS_PER_CYCLE
        print(f"\n--- Cycle {cycle + 1}/{TOTAL_CYCLES} (Total Timesteps: {total_steps}) ---")


        model_A.learn(total_timesteps=STEPS_PER_CYCLE,
                      reset_num_timesteps=False)

        wins = evaluate_model(model_A, EVAL_GAMES_PER_CYCLE, render=False)
        win_rate = (wins / EVAL_GAMES_PER_CYCLE) * 100
        print(f"Evaluation Result: {wins}/{EVAL_GAMES_PER_CYCLE} wins ({win_rate:.1f}%)")

    model_A.save(final_model_path)
    print("\n" + "=" * 50)
    print("모든 학습 및 평가가 완료되었습니다.")
    print(f"최종 학습된 모델이 '{final_model_path}'에 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
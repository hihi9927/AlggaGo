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

from env import AlggaGoEnv
from physics import WIDTH, HEIGHT, all_stones_stopped, MARGIN

# 하이퍼파라미터 및 설정
MAX_STAGES = 300
TIMESTEPS_PER_STAGE = 10000
SAVE_DIR = "rl_models_competitive"
LOG_DIR = "rl_logs_competitive"
INITIAL_ENT_COEF_A = 0.05
INITIAL_ENT_COEF_B = 0
ENT_COEF_INCREMENT = 0.1
MAX_ENT_COEF = 0.5
EVAL_EPISODES_FOR_COMPETITION = 200

# 진행률 표시 콜백 클래스
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="학습 진행률", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    def _on_step(self):
        if self.pbar: self.pbar.update(1)
        return True
    def _on_training_end(self):
        if self.pbar: self.pbar.close(); self.pbar = None

# Rule-based 초기화 
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

# 모델 파라미터 확인
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

# 환경 생성 헬퍼
def make_env_fn():
    def _init():
        env = AlggaGoEnv()
        monitored_env = Monitor(env, filename=LOG_DIR)
        return monitored_env
    return _init

# 기타 헬퍼 함수
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

# 공정 평가(Fair Evaluation)
def evaluate_fairly(model_A: PPO, model_B: PPO, num_episodes: int):
    games_per_round = num_episodes // 2
    if games_per_round == 0: return 0.5, 0.5, 0.0, 0.0
    print(f"  - 공정한 평가: 총 {num_episodes} 게임 ({games_per_round} 게임/라운드)")
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
                obs, _, done, _, info = env.step(action)
            if info.get('winner') == 'black': black_wins += 1
        env.close()
        return black_wins / num_games if num_games > 0 else 0
    r1_black_win_rate = _play_round(model_A, model_B, games_per_round, "1라운드 (A가 흑돌)")
    print(f"  ▶ 1라운드 (Model A 흑돌) 승률: {r1_black_win_rate:.2%}")
    r2_black_win_rate = _play_round(model_B, model_A, games_per_round, "2라운드 (B가 흑돌)")
    print(f"  ▶ 2라운드 (Model B 흑돌) 승률: {r2_black_win_rate:.2%}")
    win_rate_A = (r1_black_win_rate + (1 - r2_black_win_rate)) / 2
    win_rate_B = (r2_black_win_rate + (1 - r1_black_win_rate)) / 2
    
    return win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate

# 시각화
def visualize_one_game(model_A: PPO, model_B: PPO, ent_A: float, ent_B: float, stage_num: int):
    stage_str = f"스테이지 {stage_num}" if stage_num > 0 else "초기 상태"
    print(f"\n--- {stage_str} 시각화 평가 시작 ---")
    if ent_A >= ent_B:
        black_model, white_model = model_A, model_B
        caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
    else:
        black_model, white_model = model_B, model_A
        caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"
    print(f"  ({caption})")
    env = AlggaGoEnv()
    obs, _ = env.reset(options={"initial_player": "black"})
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)
    from opponent import rule_based_action
    from physics import scale_force
    from pymunk import Vec2d
    done = False; step_count = 0; max_steps = 200; info = {}
    while not done and step_count < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue
        current_player = env.current_player
        action_model = black_model if current_player == "black" else white_model
        action_offsets, _ = action_model.predict(obs.reshape(1, -1), deterministic=True)
        index_weight, angle_offset, force_offset = np.squeeze(action_offsets)
        rule_action = rule_based_action(env.stones, current_player)
        if rule_action is None: break
        rule_idx, rule_angle, rule_force = rule_action
        player_color_tuple = (0, 0, 0) if current_player == "black" else (255, 255, 255)
        player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
        if not player_stones: break
        if len(player_stones) > 1:
            idx_offset = int(np.round(index_weight * (len(player_stones) - 1) / 2))
            final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
        else: final_idx = 0
        final_angle = rule_angle + angle_offset
        final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
        selected_stone = player_stones[final_idx]
        direction = Vec2d(1, 0).rotated(final_angle)
        impulse = direction * scale_force(final_force)
        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)
        selected_stone.body.apply_impulse_at_world_point(impulse, selected_stone.body.position)
        physics_steps = 0
        while physics_steps < 300:
            env.space.step(1/60.0)
            for shape in env.stones[:]:
                if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                    env.space.remove(shape, shape.body); env.stones.remove(shape)
            env.render(screen=screen)
            pygame.display.flip()
            time.sleep(1/60.0)
            physics_steps += 1
            if all_stones_stopped(env.stones): break
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

# 경쟁적 학습 메인
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    temp_env = DummyVecEnv([make_env_fn()])

    LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
    file_exists = os.path.exists(LOG_FILE)
    csv_header = ["Stage", "Model A Entropy", "Model B Entropy", "Round 1 Win Rate (A Black)", "Round 2 Win Rate (B Black)"]
    if not file_exists:
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    state = load_training_state() or {}
    total_timesteps_so_far = state.get("total_timesteps_so_far", 0)
    current_ent_coef_A = state.get("current_ent_coef_A", INITIAL_ENT_COEF_A)
    current_ent_coef_B = state.get("current_ent_coef_B", INITIAL_ENT_COEF_B)
    best_overall_models = state.get("best_overall_models", [])

    model_A, model_B = None, None
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")

    try:
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
        print(f"[INFO] 새 학습 시작 ({e}).")
        model_A = PPO("MlpPolicy", temp_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A)
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
        print("[INFO] 초기화된 Model A를 복제하여 Model B를 생성합니다.")
        initial_A_path = os.path.join(SAVE_DIR, f"model_a_0_{INITIAL_ENT_COEF_A:.3f}.zip")
        model_A.save(initial_A_path)
        model_B = PPO.load(initial_A_path, env=temp_env)
        total_timesteps_so_far = 0
        print("\n[초기 상태 시각화] Rule-based로 초기화된 두 모델의 대결을 시작합니다.")
        visualize_one_game(model_A, model_B, INITIAL_ENT_COEF_A, INITIAL_ENT_COEF_B, stage_num=0)

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
        model_to_train.set_env(temp_env)
        print(f"  학습 대상: Model {current_training_model_name} (ent_coef: {ent_coef_train:.5f})")
        print(f"  (현재 엔트로피: Model A={current_ent_coef_A:.5f}, Model B={current_ent_coef_B:.5f})")
        
        model_to_train.learn(total_timesteps=TIMESTEPS_PER_STAGE, callback=ProgressCallback(TIMESTEPS_PER_STAGE), reset_num_timesteps=False)
        total_timesteps_so_far = model_to_train.num_timesteps
        if current_training_model_name == "A": model_A = model_to_train
        else: model_B = model_to_train

        print(f"\n  --- 경쟁 평가 시작 ---")
        win_rate_A, win_rate_B, r1_win_rate, r2_win_rate = evaluate_fairly(model_A, model_B, num_episodes=EVAL_EPISODES_FOR_COMPETITION)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            log_data = [stage_idx + 1, f"{current_ent_coef_A:.5f}", f"{current_ent_coef_B:.5f}", f"{r1_win_rate:.4f}", f"{r2_win_rate:.4f}"]
            writer.writerow(log_data)
        print("  [INFO] 학습 결과가 CSV 로그 파일에 기록되었습니다.")

        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_idx + 1)

        print("\n  --- 엔트로피 조정 ---")
        print(f"  현재 엔트로피: A={current_ent_coef_A:.5f}, B={current_ent_coef_B:.5f}")
        
        if win_rate_A > win_rate_B: effective_winner = "A"
        elif win_rate_B > win_rate_A: effective_winner = "B"
        else: effective_winner = "B" if current_training_model_name == "A" else "A"
        
        if win_rate_A == win_rate_B: print(f"  경쟁 결과: 무승부. 학습 대상({current_training_model_name})이 패배한 것으로 간주하여 Model {effective_winner} 승리.")
        else: print(f"  경쟁 결과: Model {effective_winner} 승리")

        if effective_winner == "A" and current_ent_coef_A > current_ent_coef_B:
            current_ent_coef_B = min(current_ent_coef_B + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            print(f"  Model B 엔트로피 증가 → {current_ent_coef_B:.5f}")
        elif effective_winner == "B" and current_ent_coef_B > current_ent_coef_A:
            current_ent_coef_A = min(current_ent_coef_A + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            print(f"  Model A 엔트로피 증가 → {current_ent_coef_A:.5f}")
        else: print("  엔트로피 조정 없음")

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

    print("\n--- 전체 경쟁적 학습 완료 ---")
    temp_env.close()

if __name__ == "__main__":
    main()
import csv
import os
import re
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alggago.visualize import visualize_one_game, visualize_vs_model_c, visualize_split_shot_debug
from .callbacks import ProgressCallback, print_overall_progress
from .config import (
    BEST_MODEL_FILENAME,
    EVAL_EPISODES_FOR_COMPETITION,
    ENT_COEF_INCREMENT,
    GAUNTLET_EVAL_EPISODES_PER_COLOR,
    GAUNTLET_TIMESTEPS,
    INITIAL_ENT_COEF_A,
    INITIAL_ENT_COEF_B,
    LOG_DIR,
    MAX_ENT_COEF,
    MAX_STAGES,
    N_ENVS_SELF_PLAY,
    N_ENVS_VS_C,
    SAVE_DIR,
    TIMESTEPS_PER_STAGE,
    VEC_NORMALIZE_STATS_PATH,
)
from .env_wrappers import VsModelCEnv, make_vs_c_env_vec, make_vs_opponent_env_vec
from .evaluation import evaluate_fairly, evaluate_vs_model_c, run_final_evaluation
from .model_utils import (
    clean_models,
    initialize_to_rule_based,
    load_training_state,
    make_env_fn,
    save_training_state,
    update_best_models,
)


def train_vs_model_c(total_timesteps=100_000, agent_side="black", ent_coef=0.1, save_name="ppo_vs_c"):
    """
    PPO 하나를 고정 상대(Model C)와 싸우며 학습하는 간단한 학습 루프.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 더미 환경 사용
    def _make():
        return VsModelCEnv(agent_side=agent_side)
    env = DummyVecEnv([_make])

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ent_coef)
    # (선택) 기존 규칙 기반 초기화 사용 가능
    initialize_to_rule_based(model)

    print(f"[INFO] PPO vs Model C 학습 시작: total_timesteps={total_timesteps}, side={agent_side}, ent_coef={ent_coef}")
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(total_timesteps), reset_num_timesteps=False)

    save_path = os.path.join(SAVE_DIR, f"{save_name}_{agent_side}_{total_timesteps}.zip")
    model.save(save_path)
    print(f"[INFO] 학습 완료. 저장: {os.path.basename(save_path)}")
    return model, save_path

def run_gauntlet_training(model_to_train, model_name, initial_timesteps):
    """
    주어진 모델이 모델 C를 이길 때까지 훈련하는 예선전 함수. (최종 버전)
    """
    print("\n" + "="*50)
    print(f"⚔️      특별 예선 시작: 모델 {model_name} vs 모델 C       ⚔️")
    print("="*50)

    GAUNTLET_LOG_FILE = os.path.join(LOG_DIR, "gauntlet_log.csv")
    if not os.path.exists(GAUNTLET_LOG_FILE):
        with open(GAUNTLET_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Total Timesteps", "Win Rate as Black", "Win Rate as White", "Overall Win Rate", "Regular Success", "Split Success", "Regular Ratio"])

    GAUNTLET_SAVE_PATH = os.path.join(SAVE_DIR, f"model_{model_name.lower()}_gauntlet_in_progress.zip")
    
    N_ENVS_VS_C = 2 
    gauntlet_env_raw = make_vs_c_env_vec(n_envs=N_ENVS_VS_C)
    gauntlet_env = VecNormalize(gauntlet_env_raw, norm_obs=True, norm_reward=True)

    if os.path.exists(GAUNTLET_SAVE_PATH):
        print(f"\n[INFO] 진행 중이던 예선전 모델({os.path.basename(GAUNTLET_SAVE_PATH)})을 로드하여 이어갑니다.")
        model_to_train = PPO.load(GAUNTLET_SAVE_PATH, env=gauntlet_env, device="auto")
        initial_timesteps = model_to_train.num_timesteps
        print(f"[INFO] 로드된 모델의 누적 타임스텝: {initial_timesteps:,}")
    else:
        print(f"\n[INFO] 모델 {model_name}에 대한 예선전을 시작합니다.")
        if model_to_train is None:
            print("   - 전달된 모델이 없어 새로 생성하고 규칙 기반으로 초기화합니다.")
            model_to_train = PPO("MlpPolicy", gauntlet_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A, max_grad_norm=0.5, learning_rate=0.0001)
            initialize_to_rule_based(model_to_train)
            visualize_split_shot_debug(model_to_train)
        else:
            print(f"   - 전달된 모델(Model {model_name})의 학습 상태를 유지하며 예선전을 시작합니다.")
            model_to_train.set_env(gauntlet_env)

        print("\n--- 훈련 시작 전 초기 상태 종합 평가 시작 ---")
        model_to_train.ent_coef = 0.0
        (win_rate, win_as_black, win_as_white, 
         reg_success, split_success, reg_ratio) = evaluate_vs_model_c(model_to_train, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR)
        
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{model_name}_Round_0_Initial", initial_timesteps,
                             f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}",
                             f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}"])
        print("   [INFO] 초기 상태 평가 결과가 CSV 로그 파일에 기록되었습니다.")

        visualize_vs_model_c(model_to_train, round_num=0, ppo_player_side="black")
        visualize_vs_model_c(model_to_train, round_num=0, ppo_player_side="white")

        if win_rate > 0.5:
            print(f"\n🏆 초기 모델이 이미 전체 승률 50%를 넘었습니다! 예선전을 통과합니다. 🏆")
            return model_to_train, model_to_train.num_timesteps
            
        print(f"   - 현재 모델을 첫 체크포인트로 저장합니다: {os.path.basename(GAUNTLET_SAVE_PATH)}")
        model_to_train.save(GAUNTLET_SAVE_PATH)
    
    original_ent_coef = model_to_train.ent_coef
    gauntlet_round = 1
    current_total_timesteps = model_to_train.num_timesteps

    while True:
        print(f"\n--- 예선 {gauntlet_round}라운드 훈련 시작 ---")
        model_to_train.ent_coef = original_ent_coef
        
        model_to_train.learn(
            total_timesteps=GAUNTLET_TIMESTEPS,
            callback=ProgressCallback(GAUNTLET_TIMESTEPS),
            reset_num_timesteps=False
        )
        current_total_timesteps = model_to_train.num_timesteps

        print("\n--- 훈련 후 종합 평가 시작 ---")
        model_to_train.ent_coef = 0.0
        (win_rate, win_as_black, win_as_white, 
         reg_success, split_success, reg_ratio) = evaluate_vs_model_c(model_to_train, num_episodes_per_color=GAUNTLET_EVAL_EPISODES_PER_COLOR)
        
        with open(GAUNTLET_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{model_name}_Round_{gauntlet_round}", current_total_timesteps,
                             f"{win_as_black:.4f}", f"{win_as_white:.4f}", f"{win_rate:.4f}",
                             f"{reg_success:.4f}", f"{split_success:.4f}", f"{reg_ratio:.4f}"])
        print("   [INFO] 예선전 결과가 CSV 로그 파일에 기록되었습니다.")

        print("\n[INFO] 예선전 시각화 평가를 시작합니다.(흑돌/백돌 각각 1판)")
        visualize_vs_model_c(model_to_train, round_num=gauntlet_round, ppo_player_side="black")
        visualize_vs_model_c(model_to_train, round_num=gauntlet_round, ppo_player_side="white")

        print("\n[INFO] 다음 학습 전략 설정...")
        
        # 성공률에 따라 각 보너스 모드를 개별적으로 활성화
        regular_bonus_active = (reg_success < 0.8)
        split_bonus_active = (split_success < 0.8)

        if regular_bonus_active:
            print(f"   📉 일반 공격 성공률 미달({reg_success:.2%}). 다음 학습에 '명중 보너스'를 활성화합니다.")
        if split_bonus_active:
            print(f"   📉 틈새 공격 성공률 미달({split_success:.2%}). 다음 학습에 '틈새 보너스'를 활성화합니다.")
        if not regular_bonus_active and not split_bonus_active:
            print(f"   👍 모든 성공률 달성. 다음 학습엔 보너스를 적용하지 않습니다.")

        # 플래그를 환경에 각각 전달
        gauntlet_env.env_method("set_bonus_modes", 
                                regular_active=regular_bonus_active, 
                                split_active=split_bonus_active)


        model_to_train.save(GAUNTLET_SAVE_PATH)
        print(f"   - 현재 모델 진행상황 저장: {os.path.basename(GAUNTLET_SAVE_PATH)}")

        if win_rate > 0.5:
            print(f"\n🎉 모델 {model_name}이(가) 전체 승률 50%를 달성했습니다! 예선전을 통과합니다. 🎉")
            if os.path.exists(GAUNTLET_SAVE_PATH):
                os.remove(GAUNTLET_SAVE_PATH)
            break
        else:
            print(f"   - 현재 승률({win_rate:.2%})이 50% 미만입니다. 다음 라운드로 계속 학습합니다.")

        gauntlet_round += 1

    return model_to_train, current_total_timesteps

# --- 경쟁적 학습 메인 함수 ---
def run_competitive_training():
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
    
    # --- 최고 승률 상태 로드 ---
    best_win_rate = state.get("best_win_rate", 0.0)
    #split_shot_threshold = state.get("split_shot_threshold", 0.5)
    
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

        model_timesteps = max(model_A.num_timesteps, model_B.num_timesteps)
        if model_timesteps > total_timesteps_so_far:
            print(f"[WARN] 모델의 타임스텝({model_timesteps:,})이 상태 파일({total_timesteps_so_far:,})보다 최신입니다. 모델 기준으로 동기화합니다.")
            total_timesteps_so_far = model_timesteps
        
        print("\n[INFO] 현재 로드된 모델의 상태를 시각화합니다...")
        visualize_one_game(model_A, model_B, current_ent_coef_A, current_ent_coef_B, stage_num=0)

    except Exception as e:
        print(f"[INFO] 새 학습 시작 ({e}).")

        # 예선전용 VecNormalize 환경을 먼저 생성합니다.
        gauntlet_env_raw = make_vs_c_env_vec(n_envs=N_ENVS_VS_C)
        gauntlet_env = VecNormalize(gauntlet_env_raw, norm_obs=True, norm_reward=True)

        # 모델을 만들 때부터 예선전용 환경(gauntlet_env)을 사용합니다.
        model_A = PPO("MlpPolicy", gauntlet_env, verbose=0, ent_coef=INITIAL_ENT_COEF_A, max_grad_norm=0.5, learning_rate=0.0001)

        print("[INFO] 모델을 Rule-based 정책으로 초기화합니다...")
        initialize_to_rule_based(model_A)
        print("[INFO] 정책을 rule-based 형태로 강제 초기화 완료")
        
        total_timesteps_so_far = 0

        # 초기 예선전은 A모델만 진행
        model_A, total_timesteps_so_far = run_gauntlet_training(
            model_to_train=None, 
            model_name="A", 
            initial_timesteps=0
        )
        
        print("\n[INFO] 예선전을 통과한 모델 A를 복제하여 모델 B를 다시 동기화합니다...")
        post_gauntlet_a_path = os.path.join(SAVE_DIR, "model_a_post_gauntlet.zip")
        model_A.save(post_gauntlet_a_path)
        
        # 모델 B를 로드할 때도 임시 환경(temp_env)을 사용합니다.
        model_B = PPO.load(post_gauntlet_a_path, env=temp_env)
        print("[INFO] 모델 B 동기화 완료.")
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
    train_env_raw = make_vs_opponent_env_vec(opponent_model=temp_opponent_model, n_envs=N_ENVS_SELF_PLAY)
    train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True)

    # 저장된 정규화 상태가 있으면 불러옵니다.
    if os.path.exists(VEC_NORMALIZE_STATS_PATH):
        print(f"[INFO] VecNormalize 상태 로드: {VEC_NORMALIZE_STATS_PATH}")
        train_env = VecNormalize.load(VEC_NORMALIZE_STATS_PATH, train_env_raw)
        # VecEnv가 내부적으로 환경을 다시 만들므로, 세팅을 다시 켜주어야 합니다.
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

        # 아래 3줄만 남기고 나머지는 삭제합니다.
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

        # --- 최고 성능 모델 저장 로직 ---
        champion_model = model_A if effective_winner == "A" else model_B
        BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_MODEL_FILENAME)
        if winner_win_rate > best_win_rate:
            print(f"   🏆 새로운 최고 승률 달성! (이전: {best_win_rate:.2%} -> 현재: {winner_win_rate:.2%})")
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
                print(f"   Model B 엔트로피 증가 -> {new_ent_coef_B:.5f}")
                current_ent_coef_B, model_to_requalify, model_to_requalify_name = new_ent_coef_B, model_B, "B"
            if new_ent_coef_B >= FINAL_EVAL_ENT_THRESHOLD:
                run_final_evaluation(champion_model=model_A, env=temp_env); should_terminate = True

        elif effective_winner == "B" and current_ent_coef_B > current_ent_coef_A:
            new_ent_coef_A = min(current_ent_coef_A + ENT_COEF_INCREMENT, MAX_ENT_COEF)
            if new_ent_coef_A != current_ent_coef_A:
                print(f"   Model A 엔트로피 증가 -> {new_ent_coef_A:.5f}")
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

        # VecNormalize 상태 저장
        train_env.save(VEC_NORMALIZE_STATS_PATH)
        print(f" 💾  VecNormalize 상태를 {os.path.basename(VEC_NORMALIZE_STATS_PATH)} 파일에 저장했습니다.")
        
        clean_models(model_A_path, model_B_path, [m[0] for m in best_overall_models])
        
        # --- 상태 저장에 최고 승률 포함 ---
        current_state = {
            "total_timesteps_so_far": total_timesteps_so_far,
            "current_ent_coef_A": current_ent_coef_A,
            "current_ent_coef_B": current_ent_coef_B,
            "best_overall_models": best_overall_models,
            "best_win_rate": best_win_rate,
        }
        save_training_state(current_state)

        minutes, seconds = divmod(int(time.time() - stage_start_time), 60)
        print(f"\n[스테이지 {stage_idx + 1}] 완료 (소요 시간: {minutes}분 {seconds}초)")

        if should_terminate:
            print("\n--- 최종 평가 완료. 학습을 종료합니다. ---")
            break

    print("\n--- 전체 경쟁 학습 완료 ---")
    temp_env.close()

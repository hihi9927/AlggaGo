import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from alggago.env import AlggaGoEnv
from alggago.reward import reward_fn
from .config import LOG_DIR, SAVE_DIR, TRAINING_STATE_FILE


# --- Rule-based 초기화 함수 ---
def initialize_to_rule_based(model):
    """
    모델의 정책을 '규칙 기반' 행동으로 초기화합니다.
    - 목적: '일반 공격'을 선호하도록 편향값으로 설정
    - 파라미터: 규칙 기반의 값을 그대로 따르도록 오프셋을 0으로 설정
    """
    with torch.no_grad():
        # 정책의 마지막 출력 레이어(action_net)를 접근합니다.
        action_net = model.policy.action_net

        # action_net의 가중치를 모두 0으로 설정하여, 출력이 편향(bias)에만 의존하도록 만듭니다.
        action_net.weight.data.fill_(0.0)

        # 모델의 최종 출력은 5개의 값을 가집니다.
        # [0]: 일반 공격 선호도 (Regular Attack Preference)
        # [1]: 틈새 공격 선호도 (Split Shot Preference)
        # [2]: raw_index (돌 선택 오프셋)
        # [3]: raw_angle (각도 오프셋)
        # [4]: raw_force (힘 오프셋)

        # 1. 전략 선택 초기화
        # '일반 공격' 선호도는 매우 높게, '틈새 공격' 선호도는 매우 낮게 설정
        action_net.bias[0].data.fill_(5.0)  # 일반 공격 선호
        action_net.bias[1].data.fill_(-3.0) # 틈새 공격 억제

        # 2. 파라미터 오프셋 초기화
        # 돌 인덱스, 각도, 힘에 대한 오프셋을 모두 0으로 설정
        action_net.bias[2].data.fill_(0.0)
        action_net.bias[3].data.fill_(0.0)
        action_net.bias[4].data.fill_(0.0)

        # 3. 행동의 분산(log_std)을 매우 낮게 설정. 초기 행동이 거의 결정적으로 만듭니다.
        if isinstance(model.policy.log_std, torch.nn.Parameter):
            model.policy.log_std.data.fill_(-3.0)

# --- 모델 파라미터 확인 함수 ---
def print_model_parameters(model: PPO):
    print("\n==== 모델 초기화 파라미터 상태 확인 ====")
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

# --- 환경 생성 팩토리 함수 ---
def make_env_fn():
    def _init():
        env = AlggaGoEnv(reward_fn=reward_fn)
        monitored_env = Monitor(env, filename=LOG_DIR)
        return monitored_env
    return _init

# --- 구형 모델 삭제 함수 ---
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


def load_training_state():
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            state = np.load(TRAINING_STATE_FILE, allow_pickle=True).item()
            print(f"[INFO] 저장된 학습 상태 로드 완료: {state}")
            return state
        except Exception as e:
            print(f"[ERROR] 학습 상태 로드 실패: {e}")
    return None


def save_training_state(state_dict):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TRAINING_STATE_FILE, state_dict)


def reload_with_env(model: PPO, new_env):
    """저장된 모델을 불러오되, env를 새로운 것으로 교체하는 함수."""
    tmp = os.path.join(SAVE_DIR, "_tmp_reload_swap_env.zip")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(tmp)
    new_model = PPO.load(tmp, env=new_env, device=model.device)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return new_model

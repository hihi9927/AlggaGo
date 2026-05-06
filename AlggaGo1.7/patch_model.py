import os
import torch
from stable_baselines3 import PPO

# --- [사용자 설정] ---
# 1. 초기화할 모델 파일의 전체 경로를 지정하세요.
MODEL_TO_PATCH_PATH = "rl_models_competitive/model_a_gauntlet_in_progress.zip"

# 2. 초기화 후 새로 저장할 모델 파일의 경로를 지정하세요.
PATCHED_MODEL_SAVE_PATH = "rl_models_competitive/model_a_gauntlet_in_progress_patched.zip"
# --------------------


def patch_model_force_parameter(load_path, save_path):
    """
    지정된 PPO 모델을 로드하여 'force' 파라미터를 초기화하고 새 파일로 저장합니다.
    """
    if not os.path.exists(load_path):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {load_path}")
        return

    print(f"[1/4] 모델 로드 중... ({os.path.basename(load_path)})")
    model = PPO.load(load_path)
    params = model.get_parameters()

    print("[2/4] 모델의 'force' 파라미터를 초기화합니다...")
    
    # 모델의 최종 출력은 5개의 값을 가집니다.
    # [0]: 일반 공격 선호도
    # [1]: 틈새 공격 선호도
    # [2]: raw_index (돌 선택)
    # [3]: raw_angle (각도)
    # [4]: raw_force (힘)  <-- 이 부분을 초기화합니다.
    
    try:
        # '힘'을 출력하는 action_net의 가중치(weight)와 편향(bias)을 0으로 설정
        force_output_index = 4
        
        # 가중치 텐서에서 5번째 행 전체를 0으로 만듭니다.
        params['policy']['action_net.weight'][force_output_index].data.fill_(0)
        
        # 편향 텐서에서 5번째 값을 0으로 만듭니다.
        params['policy']['action_net.bias'][force_output_index].data.fill_(0)
        
        print("   - 'force' 파라미터 초기화 성공.")

    except (KeyError, IndexError) as e:
        print(f"[오류] 모델 파라미터 구조를 찾거나 수정하는 데 실패했습니다: {e}")
        return

    print("[3/4] 수정된 파라미터를 모델에 적용합니다...")
    model.set_parameters(params)
    
    print(f"[4/4] 초기화된 모델을 새 파일로 저장 중... ({os.path.basename(save_path)})")
    model.save(save_path)
    
    print("\n✅ 작업 완료!")
    print(f"이제부터 '{os.path.basename(save_path)}' 파일을 사용하여 학습을 이어가세요.")


if __name__ == "__main__":
    patch_model_force_parameter(MODEL_TO_PATCH_PATH, PATCHED_MODEL_SAVE_PATH)
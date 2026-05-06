import numpy as np
import os
from stable_baselines3 import PPO
from pymunk import Vec2d
from physics import scale_force
import re

MODEL_SAVE_DIR = "rl_models_competitive"

class MainRLAgent:
    """
    메인 강화 학습 모델의 로딩 및 추론 로직을 담당하는 클래스.
    이 모델은 흑돌과 백돌 역할을 모두 수행할 수 있도록 훈련됩니다.
    """
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                print(f"[MainRLAgent] 모델 로드 성공: {model_path}")
            except Exception as e:
                print(f"[MainRLAgent] 모델 로드 실패: {model_path} - {e}")
                self.model = None
        elif model_path:
            print(f"[MainRLAgent] 지정된 모델 경로를 찾을 수 없음: {model_path}")

    def select_action(self, observation: np.ndarray, num_current_player_stones: int):
        """
        주어진 관측을 바탕으로 현재 턴 플레이어의 rule-based 행동에 대한 오차를 예측합니다.
        """
        if num_current_player_stones <= 0:
            print("[MainRLAgent] 행동 불가: 선택할 돌이 없습니다.")
            return None
        if self.model is None:
            print(f"[MainRLAgent] 모델 없음: 순수 rule-based 행동")
            return (0.0, 0.0, 0.0)

        observation_reshaped = np.expand_dims(observation, axis=0)
        action_raw, _states = self.model.predict(observation_reshaped, deterministic=True)
        action_raw = action_raw[0] 
        
        index_weight = np.clip(action_raw[0], -1.0, 1.0)
        angle_offset = np.clip(action_raw[1], -np.pi, np.pi)
        force_offset = np.clip(action_raw[2], -0.5, 0.5)

        print(f"[MainRLAgent] AI 오차 예측 - index_weight: {index_weight:.3f}, angle_offset: {angle_offset:.3f}, force_offset: {force_offset:.3f}")
        
        return (index_weight, angle_offset, force_offset)

def apply_action_to_stone(action_tuple: tuple, stones: list, target_color_tuple: tuple):
    """
    Rule-based 행동에 오차를 적용하여 해당 색상의 돌에 impulse를 적용합니다.
    """
    from opponent import rule_based_action
    
    if action_tuple is None:
        return

    index_weight, angle_offset, force_offset = action_tuple
    current_player = "white" if target_color_tuple == (255, 255, 255) else "black"
    
    rule_action = rule_based_action(stones, current_player)
    if rule_action is None:
        print(f"[ERROR] apply_action_to_stone: Rule-based 행동 계산 실패")
        return
    
    rule_idx, rule_angle, rule_force = rule_action
    target_stones = [s for s in stones if s.color[:3] == target_color_tuple]
    
    if not target_stones:
        print(f"[ERROR] apply_action_to_stone: 대상 돌이 없습니다.")
        return
    
    if len(target_stones) > 1:
        idx_offset = int(np.round(index_weight * (len(target_stones) - 1) / 2))
        final_idx = np.clip(rule_idx + idx_offset, 0, len(target_stones) - 1)
    else:
        final_idx = 0
    
    final_angle = rule_angle + angle_offset
    final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
    
    print(f"[apply_action] Rule: idx={rule_idx}, angle={rule_angle:.2f}, force={rule_force:.2f}")
    print(f"[apply_action] Final: idx={final_idx}, angle={final_angle:.2f}, force={final_force:.2f}")
    
    stone_to_move = target_stones[final_idx]
    direction = Vec2d(1, 0).rotated(final_angle)
    scaled_force = scale_force(final_force)
    impulse = direction * scaled_force
    stone_to_move.body.apply_impulse_at_world_point(impulse, stone_to_move.body.position)

def choose_ai():
    """
    사용자 입력을 받아 메인 에이전트 모델을 선택합니다.
    Returns:
        MainRLAgent 인스턴스.
    """
    print("🤖 AI 행동 방식을 선택하세요:")
    
    # 파일명 규칙(타임스텝 + 엔트로피)에 맞는 정규 표현식
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")
    available_models = []

    if os.path.exists(MODEL_SAVE_DIR):
        for filename in sorted(os.listdir(MODEL_SAVE_DIR)):
            match = model_pattern.match(filename)
            if match:
                # 정규 표현식 그룹 순서에 맞게 데이터 추출
                model_char = match.group(1).upper() # 'a' 또는 'b'
                timesteps = int(match.group(2))     # 타임스텝
                entropy = float(match.group(3))     # 엔트로피
                
                full_path = os.path.join(MODEL_SAVE_DIR, filename)
                available_models.append({
                    "char": model_char,
                    "timesteps": timesteps,
                    "entropy": entropy, # 엔트로피 정보 추가
                    "path": full_path
                })
    
    # 모델 목록을 타임스텝 순으로 정렬
    available_models.sort(key=lambda x: x["timesteps"])

    # 0번 옵션은 항상 순수 Rule-based
    print("   0) 순수 Rule-based (AI 없음)")
    
    model_options = {"0": None} # 0번은 모델 없음
    
    if not available_models:
        print("   (사용 가능한 강화학습 모델이 없습니다.)")
    else:
        for i, model_info in enumerate(available_models):
            option_num = i + 1
            print(f"   {option_num}) 강화학습 모델 (Model {model_info['char']} - {model_info['timesteps']} steps, ent={model_info['entropy']:.3f})")
            model_options[str(option_num)] = model_info["path"]

    # 사용자 입력 받기
    choice = input("선택 (0" + "".join(f"/{i+1}" for i in range(len(available_models))) + "): ").strip()

    model_path = model_options.get(choice)
    
    if choice in model_options:
        if model_path:
            print(f"[AI 선택] 강화학습 모델 '{os.path.basename(model_path)}'을(를) 사용합니다.")
        else:
            print("[AI 선택] 순수 Rule-based를 사용합니다.")
    else:
        print("[AI 선택] 유효하지 않은 선택입니다. 순수 Rule-based를 사용합니다.")
        model_path = None

    return MainRLAgent(model_path=model_path)
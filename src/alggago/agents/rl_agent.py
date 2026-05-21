import numpy as np
import os
import traceback
from stable_baselines3 import PPO
from pymunk import Vec2d
from alggago.physics import scale_force
import re
from alggago.agents.rule_base import get_regular_action, get_split_shot_action

# numpy pickle 호환성 패치
def _patch_numpy_compat():
    """
    cloudpickle.loads 를 Python-level 커스텀 Unpickler 로 교체한다.
    C-level _pickle.Unpickler 는 패치 불가이므로,
    cloudpickle.loads 자체를 래핑해서 모듈 경로를 리매핑한다.
    """
    import io
    import pickle
    import cloudpickle

    # numpy._core.X 를 numpy.core.X 로 리매핑하는 Unpickler
    class _NumpyCompatUnpickler(pickle.Unpickler):
        _REMAP = {
            'numpy._core': 'numpy.core',
        }

        def find_class(self, module, name):
            for old, new in self._REMAP.items():
                if module == old or module.startswith(old + '.'):
                    module = module.replace(old, new, 1)
                    break
            return super().find_class(module, name)

    _orig_loads = cloudpickle.loads

    def _compat_loads(data, *args, **kwargs):
        try:
            return _NumpyCompatUnpickler(io.BytesIO(data)).load()
        except Exception:
            # 패치 실패 시 원본 시도
            return _orig_loads(data, *args, **kwargs)

    cloudpickle.loads = _compat_loads

_patch_numpy_compat()

# --- [추가] 파일 기준 경로 세팅 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
def rel_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

MODEL_SAVE_DIR = rel_path("rl_models_competitive")

class MainRLAgent:
    """
    메인 강화 학습 모델의 로딩 및 추론 로직을 담당하는 클래스.
    """
    def __init__(self, model_path=None):
        self.model = None
        # --- [수정] 모델 경로를 클래스 변수로 저장 ---
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                print(f"[MainRLAgent] 모델 로드 성공: {model_path}")
            except Exception as e:
                import traceback
                print(f"[MainRLAgent] 모델 로드 실패: {model_path} - {e}")
                traceback.print_exc()
                self.model = None
        elif model_path:
            print(f"[MainRLAgent] 지정된 모델 경로를 찾을 수 없음: {model_path}")

    def select_action(self, observation: np.ndarray):
        """
        주어진 관측을 바탕으로 전체 행동 벡터(5개 값)를 예측합니다.
        """
        if self.model is None:
            print(f"[MainRLAgent] 모델 없음: 순수 rule-based 행동")
            return np.array([10.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)

        try:
            # --- 1. 관측 공간(Observation Space) 하위 호환성 보정 ---
            # 과거 모델(1.0 버전 등)이 24차원으로 학습된 경우를 대비해 잘라줍니다.
            obs_shape = getattr(self.model.observation_space, 'shape', (25,))
            expected_obs_shape = obs_shape[0] if obs_shape else 25
            if expected_obs_shape == 24 and observation.shape[0] == 25:
                observation_to_use = observation[:24]
            else:
                observation_to_use = observation

            observation_reshaped = np.expand_dims(observation_to_use, axis=0)
            action_raw, _states = self.model.predict(observation_reshaped, deterministic=True)
            action_vector = action_raw[0]
            
            # --- 2. 행동 공간(Action Space) 하위 호환성 보정 ---
            if action_vector.size == 3:
                print("[MainRLAgent] AlggaGo 1.0 (3-dim) 모델 감지 -> 5-dim 변환")
                pref_regular, pref_split = 10.0, -10.0
                raw_index, raw_angle, raw_force = action_vector
                return np.array([pref_regular, pref_split, raw_index, raw_angle, raw_force], dtype=np.float32)
            elif action_vector.size == 4:
                print("[MainRLAgent] 과거 4-dim 모델 감지 -> 5-dim 변환")
                strat_val = action_vector[0]
                pref_regular, pref_split = -strat_val, strat_val
                raw_index, raw_angle, raw_force = action_vector[1:]
                return np.array([pref_regular, pref_split, raw_index, raw_angle, raw_force], dtype=np.float32)

            return action_vector

        except Exception as e:
            print(f"[MainRLAgent] 예측(predict) 중 에러 발생: {e}")
            traceback.print_exc()
            # 에러 시 멈추지 않고 룰베이스(일반공격)로 안전하게 진행
            return np.array([10.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)

def apply_action_to_stone(full_action: np.ndarray, stones: list, target_color_tuple: tuple):
    """
    모델이 예측한 전체 행동 벡터를 해석하여 돌에 impulse를 적용합니다.
    """
    if full_action is None:
        return

    player_stones = [s for s in stones if s.color[:3] == target_color_tuple]
    opponent_stones = [s for s in stones if s.color[:3] != target_color_tuple]

    if not player_stones:
        print(f"[ERROR] apply_action_to_stone: 대상 돌이 없습니다.")
        return

    # --- argmax를 softmax 샘플링 로직으로 교체 ---
    strategy_preferences = np.asarray(full_action[:2], dtype=np.float32)
    
    # 안정화된 소프트맥스 확률 계산 (train.py와 동일한 로직)
    max_pref = float(np.max(strategy_preferences))
    exp_p = np.exp(strategy_preferences - max_pref)
    probs = exp_p / (np.sum(exp_p) + 1e-8)

    # 수치 이상 시 안전하게 argmax로 폴백
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        strategy_choice = int(np.argmax(strategy_preferences))
    else:
        # 계산된 확률에 따라 행동을 랜덤하게 선택
        strategy_choice = int(np.random.choice(2, p=probs))
    # --- 수정 끝 ---

    if strategy_choice == 1: # 1번 전략: 틈새 공격
        rule_action = get_split_shot_action(player_stones, opponent_stones)
        if rule_action is None:
            rule_action = get_regular_action(player_stones, opponent_stones)
            print("[Action Strategy] 모델이 '틈새 공격'을 원했으나, 불가능하여 '일반 공격'으로 전환")
        else:
            print("[Action Strategy] 모델이 '틈새 공격(Split Shot)'을 선택")
    else: # 0번 전략: 일반 공격
        rule_action = get_regular_action(player_stones, opponent_stones)
        print("[Action Strategy] 모델이 '일반 공격(Regular Action)'을 선택")

    if rule_action is None:
        print(f"[ERROR] apply_action_to_stone: Rule-based 행동 계산 실패")
        return
        
    raw_index, raw_angle, raw_force = full_action[2:]
    
    index_weight = np.clip(raw_index, -1.0, 1.0)
    angle_offset = np.clip(raw_angle, -1.0, 1.0)
    force_offset = np.clip(raw_force, -1.0, 1.0)
    
    exploration_range = {"index": 1.0, "angle": np.pi / 4, "force": 0.5}

    final_index_weight = index_weight * exploration_range['index']
    final_angle_offset = angle_offset * exploration_range['angle']
    final_force_offset = force_offset * exploration_range['force']

    print(f"[MainRLAgent] AI 오차 예측 - index_w: {final_index_weight:.3f}, angle_o: {final_angle_offset:.3f}, force_o: {final_force_offset:.3f}")

    rule_idx, rule_angle, rule_force = rule_action

    if len(player_stones) > 1:
        # 2. train.py와 동일하게 scaling 부분 삭제
        idx_offset = int(np.round(final_index_weight))
        final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
    else:
        final_idx = 0

    final_angle = rule_angle + final_angle_offset
    final_force = np.clip(rule_force + final_force_offset, 0.0, 1.0)

    print(f"[apply_action] Rule: idx={rule_idx}, angle={rule_angle:.2f}, force={rule_force:.2f}")
    print(f"[apply_action] Final: idx={final_idx}, angle={final_angle:.2f}, force={final_force:.2f}")

    stone_to_move = player_stones[final_idx]
    direction = Vec2d(1, 0).rotated(final_angle)
    scaled_force = scale_force(final_force)
    impulse = direction * scaled_force
    stone_to_move.body.apply_impulse_at_world_point(impulse, stone_to_move.body.position)

def choose_ai():
    """
    사용자의 입력을 받아 메인 에이전트 모델을 선택합니다.
    (이 함수는 수정할 필요가 없습니다)
    """
    print("🤖 AI 행동 방식을 선택하세요.")
    
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")
    available_models = []

    if os.path.exists(MODEL_SAVE_DIR):
        for filename in sorted(os.listdir(MODEL_SAVE_DIR)):
            match = model_pattern.match(filename)
            if match:
                model_char = match.group(1).upper()
                timesteps = int(match.group(2))
                entropy = float(match.group(3))
                full_path = os.path.join(MODEL_SAVE_DIR, filename)
                available_models.append({
                    "char": model_char, "timesteps": timesteps,
                    "entropy": entropy, "path": full_path
                })
    
    available_models.sort(key=lambda x: x["timesteps"])

    print("   0) 순수 Rule-based (AI 없음)")
    model_options = {"0": None}
    
    if not available_models:
        print("   (사용 가능한 강화학습 모델이 없습니다.)")
    else:
        for i, model_info in enumerate(available_models):
            option_num = i + 1
            print(f"   {option_num}) 강화학습 모델 (Model {model_info['char']} - {model_info['timesteps']} steps, ent={model_info['entropy']:.3f})")
            model_options[str(option_num)] = model_info["path"]

    choice = input("선택 (0" + "".join(f"/{i+1}" for i in range(len(available_models))) + "): ").strip()
    model_path = model_options.get(choice)
    
    if choice in model_options:
        if model_path:
            print(f"[AI 선택] 강화학습 모델 '{os.path.basename(model_path)}' 을(를) 사용합니다.")
        else:
            print("[AI 선택] 순수 Rule-based를 사용합니다.")
    else:
        print("[AI 선택] 유효하지 않은 선택입니다. 순수 Rule-based를 사용합니다.")
        model_path = None

    return MainRLAgent(model_path=model_path)

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from alggago.agents.model_c import model_c_action
from alggago.env import AlggaGoEnv
from alggago.reward import reward_fn
from alggago.physics import WIDTH, HEIGHT, MARGIN, all_stones_stopped
from .config import SAVE_DIR


# --- 공정 평가(Fair Evaluation) 함수 ---
def evaluate_fairly(model_A: PPO, model_B: PPO, num_episodes: int):
    games_per_round = num_episodes // 2
    if games_per_round == 0: return 0.5, 0.5, 0.0, 0.0
    print(f"   - 공정한 평가: 총 {num_episodes} 게임 ({games_per_round} 게임/라운드)")

    def _play_round(black_model: PPO, white_model: PPO, num_games: int, round_name: str):
        black_wins = 0
        env = Monitor(AlggaGoEnv(reward_fn=reward_fn))
        for _ in tqdm(range(num_games), desc=round_name, leave=False):
            obs, _ = env.reset(options={"initial_player": "black"})
            done = False
            info: dict = {}
            while not done:
                current_player = env.unwrapped.current_player  # type: ignore[attr-defined]
                action_model = black_model if current_player == 'black' else white_model
                obs_shape = getattr(action_model.observation_space, 'shape', (25,))
                expected_dim = obs_shape[0] if obs_shape else 25
                obs_to_use = obs[:24] if expected_dim == 24 and len(obs) == 25 else obs
                action, _ = action_model.predict(obs_to_use, deterministic=True)
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
        model_A.ent_coef = 0
        model_B.ent_coef = 0

        r1_black_win_rate = _play_round(model_A, model_B, games_per_round, "1라운드 (A가 흑돌)")
        print(f"   🚩 1라운드 (Model A 흑돌) 승률: {r1_black_win_rate:.2%}")
        r2_black_win_rate = _play_round(model_B, model_A, games_per_round, "2라운드 (B가 흑돌)")
        print(f"   🚩 2라운드 (Model B 흑돌) 승률: {r2_black_win_rate:.2%}")
        win_rate_A = (r1_black_win_rate + (1 - r2_black_win_rate)) / 2
        win_rate_B = (r2_black_win_rate + (1 - r1_black_win_rate)) / 2
    finally:
        # 평가 후 원래 엔트로피로 복원
        model_A.ent_coef = original_ent_A
        model_B.ent_coef = original_ent_B
        print("   [INFO] 원래 엔트로피 값으로 복원했습니다.")

    return win_rate_A, win_rate_B, r1_black_win_rate, r2_black_win_rate

def evaluate_vs_model_c(ppo_model: PPO, num_episodes_per_color: int):
    """PPO 모델과 모델 C의 승률 및 각 전략의 성공률을 종합적으로 평가하는 함수"""
    print(f"   - 모델 C와 특별 평가: 총 {num_episodes_per_color * 2} 게임 (각 진영당 {num_episodes_per_color}판)")
    env = AlggaGoEnv(reward_fn=reward_fn)
    from pymunk import Vec2d
    from alggago.physics import scale_force, all_stones_stopped

    win_rates = {}
    total_wins = 0
    
    strategy_attempts = {0: 0, 1: 0}
    strategy_successes = {0: 0, 1: 0}
    
    for side in ["black", "white"]:
        ppo_wins_on_side = 0
        desc = f"   PPO({side}) vs C"
        
        for _ in tqdm(range(num_episodes_per_color), desc=desc, leave=False):
            obs, _ = env.reset(options={"initial_player": side})
            done = False; info = {}
            while not done:
                current_player_color = env.current_player
                
                if current_player_color == side:
                    obs_shape = getattr(ppo_model.observation_space, 'shape', (25,))
                    expected_dim = obs_shape[0] if obs_shape else 25
                    obs_to_use = obs[:24] if expected_dim == 24 and len(obs) == 25 else obs
                    action, _ = ppo_model.predict(obs_to_use, deterministic=True)
                    obs, _, done, _, info = env.step(action)
                    
                    strategy = info.get('strategy_choice')
                    if strategy is not None:
                        strategy_attempts[strategy] += 1
                        if info.get('is_regular_success', False) or info.get('is_split_success', False):
                            strategy_successes[strategy] += 1
                else:
                    # ... (모델 C의 턴 로직은 기존과 동일) ...
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
                        env.space.step(1/60.0); physics_steps += 1

                    for shape in env.stones[:]:
                        if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                            env.space.remove(shape, shape.body); env.stones.remove(shape)
                    
                    current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
                    current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))
                    if current_black == 0: done = True; info['winner'] = 'white'
                    elif current_white == 0: done = True; info['winner'] = 'black'
                    env.current_player = "white" if current_player_color == "black" else "black"
                    obs = env._get_obs()

            if done and info.get('winner') == side:
                ppo_wins_on_side += 1
        
        win_rate = ppo_wins_on_side / num_episodes_per_color if num_episodes_per_color > 0 else 0
        print(f"   ▶ PPO가 {side}일 때 승률: {win_rate:.2%}")
        win_rates[side] = win_rate
        total_wins += ppo_wins_on_side

    env.close()
    
    # --- 모든 통계 지표 계산 ---
    overall_win_rate = total_wins / (num_episodes_per_color * 2) if num_episodes_per_color > 0 else 0
    win_rate_as_black = win_rates.get("black", 0)
    win_rate_as_white = win_rates.get("white", 0)
    
    regular_success_rate = strategy_successes[0] / strategy_attempts[0] if strategy_attempts[0] > 0 else 0
    split_success_rate = strategy_successes[1] / strategy_attempts[1] if strategy_attempts[1] > 0 else 0
    
    total_strategy_attempts = strategy_attempts[0] + strategy_attempts[1]
    regular_attack_ratio = strategy_attempts[0] / total_strategy_attempts if total_strategy_attempts > 0 else 0
    
    # --- 콘솔 출력 부분 ---
    print(f"   🏆 모델 PPO 전체 승률 (vs C): {overall_win_rate:.2%}")
    print(f"   📊 일반 공격 선택 비율: {regular_attack_ratio:.2%}")
    print(f"   🎯 일반 공격 성공률: {regular_success_rate:.2%} ({strategy_successes[0]}/{strategy_attempts[0]})")
    print(f"   🎯 틈새 공격 성공률: {split_success_rate:.2%} ({strategy_successes[1]}/{strategy_attempts[1]})")

    
    return (overall_win_rate, win_rate_as_black, win_rate_as_white,
            regular_success_rate, split_success_rate, regular_attack_ratio)

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

    log_dir = champion_model.logger.get_dir() or "unknown"
    print(f"[*] Champion model log dir: {os.path.basename(log_dir)}")
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

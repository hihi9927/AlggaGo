import time
import numpy as np
import pygame
from stable_baselines3 import PPO

from alggago.env import AlggaGoEnv
from alggago.reward import reward_fn
from alggago.physics import WIDTH, HEIGHT, MARGIN, all_stones_stopped
from alggago.agents.model_c import model_c_action
from alggago.agents.rule_base import get_regular_action, get_split_shot_action


# --- 시각화 함수 ---
def visualize_one_game(model_A: PPO, model_B: PPO, ent_A: float, ent_B: float, stage_num: int, force_A_as_black: bool | None = None):
    """
    한 게임을 시각화합니다. (수정된 버전: PPO 이외의 물리 과정을 프레임별로 렌더링)
    force_A_as_black: True이면 A가 흑돌, False이면 B가 흑돌, None이면 엔트로피 기반으로 결정합니다.
    """
    stage_str = f"스테이지 {stage_num}" if stage_num > 0 else "초기 상태"

    if force_A_as_black is True:
        black_model, white_model = model_A, model_B
        caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
    elif force_A_as_black is False:
        black_model, white_model = model_B, model_A
        caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"
    else:
        if ent_A >= ent_B:
            black_model, white_model = model_A, model_B
            caption = f"{stage_str} Eval: A(Black, ent={ent_A:.3f}) vs B(White, ent={ent_B:.3f})"
        else:
            black_model, white_model = model_B, model_A
            caption = f"{stage_str} Eval: B(Black, ent={ent_B:.3f}) vs A(White, ent={ent_A:.3f})"

    print(f"\n--- 시각화 평가: {caption} ---")

    env = AlggaGoEnv(reward_fn=reward_fn)
    obs, _ = env.reset(options={"initial_player": "black"})
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)

    from pymunk import Vec2d
    from alggago.physics import scale_force

    done = False
    info = {}

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: break

        current_player = env.current_player
        action_model = black_model if current_player == "black" else white_model

        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)

        # --- 관측 크기 보정 (24-dim 호환) ---
        obs_shape = getattr(action_model.observation_space, 'shape', (25,))
        expected_dim = obs_shape[0] if obs_shape else 25
        obs_to_use = obs[:24] if expected_dim == 24 and len(obs) == 25 else obs

        action_values, _ = action_model.predict(obs_to_use, deterministic=True)

        # --- AlggaGo1.0 모델 (3-val) 호환성 처리 ---
        if len(action_values) == 3:
            action_values = np.array([1.0, -1.0, action_values[0], action_values[1], action_values[2]], dtype=np.float32)

        player_color_tuple = (0,0,0) if env.current_player == "black" else (255,255,255)
        player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
        opponent_stones = [s for s in env.stones if s.color[:3] != player_color_tuple]

        if not player_stones or not opponent_stones:
            done = True; continue

        if len(opponent_stones) < 2:
            strategy_choice = 0
        else:
            strategy_preferences = np.asarray(action_values[:2], dtype=np.float32)
            max_pref = float(np.max(strategy_preferences)); exp_p = np.exp(strategy_preferences - max_pref)
            probs = exp_p / (np.sum(exp_p) + 1e-8)
            strategy_choice = int(np.random.choice(2, p=probs)) if np.all(np.isfinite(probs)) and probs.sum() > 0 else int(np.argmax(strategy_preferences))

        chosen_str = '일반공격(0)' if strategy_choice == 0 else '스플릿샷(1)'
        print(f"[viz] {current_player} 의 전략: {chosen_str}")

        rule_action = get_split_shot_action(player_stones, opponent_stones) if strategy_choice == 1 else get_regular_action(player_stones, opponent_stones)
        if rule_action is None: rule_action = get_regular_action(player_stones, opponent_stones)

        if rule_action:
            raw_index, raw_angle, raw_force = action_values[2:]
            raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
            raw_index = np.clip(raw_idx_val, -1.0, 1.0)
            raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
            raw_force = np.clip(raw_force_val, -1.0, 1.0)

            index_weight = raw_index * env.exploration_range['index']
            angle_offset = raw_angle * env.exploration_range['angle']
            force_offset = raw_force * env.exploration_range['force']
            rule_idx, rule_angle, rule_force = rule_action

            final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1) if len(player_stones) > 1 else 0
            final_angle = rule_angle + angle_offset
            final_force = np.clip(rule_force + force_offset, 0.0, 1.0)

            selected_stone_to_shoot = player_stones[final_idx]
            direction = Vec2d(1, 0).rotated(final_angle)
            impulse = direction * scale_force(final_force)
            selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

            physics_steps = 0
            while not all_stones_stopped(env.stones) and physics_steps < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps += 1

        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

        current_black = sum(1 for s in env.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255,255,255))

        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'

        if not done: env.current_player = "white" if env.current_player == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()


def visualize_vs_model_c(ppo_model: PPO, round_num: int, ppo_player_side: str):
    """
    PPO 모델과 모델 C의 대결을 시각화합니다. (수정된 버전: PPO 턴 렌더링 포함)
    ppo_player_side: PPO 모델이 플레이할 색상 ('black' 또는 'white')
    """
    stage_str = f"예선전 {round_num}라운드"
    caption = (f"{stage_str}: 모델 A(흑돌) vs 모델 C(백돌)" if ppo_player_side == "black"
               else f"{stage_str}: 모델 C(흑돌) vs 모델 A(백돌)")

    print(f"\n--- {stage_str} 시각화 ({caption}) ---")

    env = AlggaGoEnv(reward_fn=reward_fn)
    obs, _ = env.reset(options={"initial_player": "black"})

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(caption)

    from pymunk import Vec2d
    from alggago.physics import scale_force, all_stones_stopped

    done = False
    info = {}
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue

        current_player_color = env.current_player

        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(0.5)

        if current_player_color == ppo_player_side:
            # --- 관측 크기 보정 (24-dim 호환) ---
            obs_shape = getattr(ppo_model.observation_space, 'shape', (25,))
            expected_dim = obs_shape[0] if obs_shape else 25
            obs_to_use = obs[:24] if expected_dim == 24 and len(obs) == 25 else obs

            action_values, _ = ppo_model.predict(obs_to_use, deterministic=True)

            # --- AlggaGo1.0 모델 (3-val) 호환성 처리 ---
            if len(action_values) == 3:
                action_values = np.array([1.0, -1.0, action_values[0], action_values[1], action_values[2]], dtype=np.float32)

            player_color_tuple = (0,0,0) if env.current_player == "black" else (255,255,255)
            player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
            opponent_stones = [s for s in env.stones if s.color[:3] != player_color_tuple]

            if not player_stones or not opponent_stones:
                done = True; continue

            if len(opponent_stones) < 2:
                strategy_choice = 0
            else:
                strategy_preferences = np.asarray(action_values[:2], dtype=np.float32)
                max_pref = float(np.max(strategy_preferences)); exp_p = np.exp(strategy_preferences - max_pref)
                probs = exp_p / (np.sum(exp_p) + 1e-8)
                strategy_choice = int(np.random.choice(2, p=probs)) if np.all(np.isfinite(probs)) and probs.sum() > 0 else int(np.argmax(strategy_preferences))
            chosen_str = '일반공격(0)' if strategy_choice == 0 else '스플릿샷(1)'
            print(f"[viz-vsC] PPO 의 전략: {chosen_str}")

            rule_action = get_split_shot_action(player_stones, opponent_stones) if strategy_choice == 1 else get_regular_action(player_stones, opponent_stones)
            if rule_action is None: rule_action = get_regular_action(player_stones, opponent_stones)

            if rule_action:
                raw_index, raw_angle, raw_force = action_values[2:]

                raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
                raw_index = np.clip(raw_idx_val, -1.0, 1.0)
                raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
                raw_force = np.clip(raw_force_val, -1.0, 1.0)

                index_weight = raw_index * env.exploration_range['index']
                angle_offset = raw_angle * env.exploration_range['angle']
                force_offset = raw_force * env.exploration_range['force']
                rule_idx, rule_angle, rule_force = rule_action

                final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1) if len(player_stones) > 1 else 0
                final_angle = rule_angle + angle_offset
                final_force = np.clip(rule_force + force_offset, 0.0, 1.0)

                selected_stone_to_shoot = player_stones[final_idx]
                direction = Vec2d(1, 0).rotated(final_angle)
                impulse = direction * scale_force(final_force)
                selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)
        else:
            action_tuple = model_c_action(env.stones, current_player_color)
            if action_tuple:
                idx, angle, force = action_tuple
                player_color_tuple = (0, 0, 0) if current_player_color == "black" else (255, 255, 255)
                player_stones = [s for s in env.stones if s.color[:3] == player_color_tuple]
                if 0 <= idx < len(player_stones):
                    stone_to_shoot = player_stones[idx]
                    direction = Vec2d(1, 0).rotated(angle)
                    impulse = direction * scale_force(force)
                    stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)

        physics_steps = 0
        while not all_stones_stopped(env.stones) and physics_steps < 600:
            env.space.step(1/60.0); env.render(screen=screen)
            pygame.display.flip(); pygame.time.delay(16)
            physics_steps += 1

        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

        current_black = sum(1 for s in env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'

        if not done: env.current_player = "white" if current_player_color == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f">>> 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(2)
    pygame.quit()


def visualize_split_shot_debug(model: PPO):
    """
    '틈새 공격' 전략만 사용하여 한 게임을 시각화하고, 매 턴의 성공/실패 여부를 터미널에 출력하는 디버깅 전용 함수.
    """
    print("\n" + "="*50)
    print("🔍      '틈새 공격' 디버깅 시각화 시작      🔍")
    print("="*50)

    env = AlggaGoEnv(reward_fn=reward_fn)
    obs, _ = env.reset(options={"initial_player": "black"})

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DEBUG: Split Shot Only")

    from pymunk import Vec2d
    from alggago.physics import scale_force, all_stones_stopped
    import itertools

    done = False
    info = {}
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
        if done: continue

        current_player_color = env.current_player
        env.render(screen=screen)
        pygame.display.flip()
        time.sleep(1.0)

        if current_player_color == "black": # PPO 모델 턴 (흑돌 고정)
            # --- 관측 크기 보정 (24-dim 호환) ---
            obs_shape = getattr(model.observation_space, 'shape', (25,))
            expected_dim = obs_shape[0] if obs_shape else 25
            obs_to_use = obs[:24] if expected_dim == 24 and len(obs) == 25 else obs

            action_values, _ = model.predict(obs_to_use, deterministic=True)

            # --- AlggaGo1.0 모델 (3-val) 호환성 처리 ---
            if len(action_values) == 3:
                action_values = np.array([1.0, -1.0, action_values[0], action_values[1], action_values[2]], dtype=np.float32)

            player_stones = [s for s in env.stones if s.color[:3] == (0,0,0)]
            opponent_stones = [s for s in env.stones if s.color[:3] == (255,255,255)]

            if not player_stones or not opponent_stones:
                done = True; continue

            strategy_choice = 1
            print(f"\n[DEBUG] PPO 턴: '틈새 공격' 강제 실행")

            rule_action = get_split_shot_action(player_stones, opponent_stones)
            if rule_action is None:
                print("[DEBUG] 틈새 공격 가능한 수가 없어 턴을 넘깁니다.")
                env.current_player = "white"
                obs = env._get_obs()
                continue

            raw_idx_val, raw_angle_val, raw_force_val = action_values[2:]
            raw_index = np.clip(raw_idx_val, -1.0, 1.0)
            raw_angle = np.clip(raw_angle_val, -1.0, 1.0)
            raw_force = np.clip(raw_force_val, -1.0, 1.0)

            index_weight = raw_index * env.exploration_range['index']
            angle_offset = raw_angle * env.exploration_range['angle']
            force_offset = raw_force * env.exploration_range['force']
            rule_idx, rule_angle, rule_force = rule_action

            final_idx = np.clip(rule_idx + int(np.round(index_weight)), 0, len(player_stones)-1)
            final_angle = rule_angle + angle_offset
            final_force = np.clip(rule_force + force_offset, 0.0, 1.0)

            selected_stone_to_shoot = player_stones[final_idx]

            direction = Vec2d(1, 0).rotated(final_angle)
            impulse = direction * scale_force(final_force)
            selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

            physics_steps = 0
            while not all_stones_stopped(env.stones) and physics_steps < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps += 1

            moved_stone_final_pos = selected_stone_to_shoot.body.position
            opponent_stones_after_shot = [s for s in env.stones if s.color[:3] == (255,255,255)]

            max_wedge_reward = 0.0
            if len(opponent_stones_after_shot) >= 2:
                for o1, o2 in itertools.combinations(opponent_stones_after_shot, 2):
                    p1, p2 = o1.body.position, o2.body.position
                    p3 = moved_stone_final_pos
                    v = p2 - p1; w = p3 - p1
                    t = w.dot(v) / (v.dot(v) + 1e-6)
                    if 0 < t < 1:
                        dist_to_segment = (p3 - (p1 + t * v)).length
                        dist_between_opponents = (p1 - p2).length

                        wedge_threshold = dist_between_opponents * 0.15
                        if dist_to_segment < wedge_threshold:
                            current_reward = (1 - (dist_to_segment / wedge_threshold)) * 0.5
                            if current_reward > max_wedge_reward:
                                max_wedge_reward = current_reward

            if max_wedge_reward > 0:
                print(f"   [DEBUG] >> 결과: 틈새 공격 성공! (보상: {max_wedge_reward:.2f})")
            else:
                print(f"   [DEBUG] >> 결과: 틈새 공격 실패.")

        else: # 모델 C 턴
            print(f"\n[DEBUG] 모델 C 턴..")
            action_tuple = model_c_action(env.stones, current_player_color)
            if action_tuple:
                idx, angle, force = action_tuple
                player_stones_c = [s for s in env.stones if s.color[:3] == (255,255,255)]
                if 0 <= idx < len(player_stones_c):
                    stone_to_shoot_c = player_stones_c[idx]
                    direction_c = Vec2d(1, 0).rotated(angle)
                    impulse_c = direction_c * scale_force(force)
                    stone_to_shoot_c.body.apply_impulse_at_world_point(impulse_c, stone_to_shoot_c.body.position)

            physics_steps_c = 0
            while not all_stones_stopped(env.stones) and physics_steps_c < 600:
                env.space.step(1/60.0); env.render(screen=screen)
                pygame.display.flip(); pygame.time.delay(16)
                physics_steps_c += 1

        for shape in env.stones[:]:
            if not (MARGIN < shape.body.position.x < WIDTH - MARGIN and MARGIN < shape.body.position.y < HEIGHT - MARGIN):
                if shape in env.space.shapes: env.space.remove(shape, shape.body)
                if shape in env.stones: env.stones.remove(shape)

        current_black = sum(1 for s in env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0: done = True; info['winner'] = 'white'
        elif current_white == 0: done = True; info['winner'] = 'black'

        if not done: env.current_player = "white" if current_player_color == "black" else "black"
        obs = env._get_obs()

    winner = info.get('winner', 'Draw/Timeout')
    print(f"\n>>> 디버깅 시각화 종료: 최종 승자 {winner} <<<")
    time.sleep(3)
    pygame.quit()

import numpy as np
import random

def get_last_stone_on_path(shooter_stone, target_stone, all_stones, player_color_tuple, stone_radius=20):
    s_pos = np.array(shooter_stone.body.position)
    t_pos = np.array(target_stone.body.position)
    shot_vector = t_pos - s_pos
    shot_length_sq = np.dot(shot_vector, shot_vector)
    if shot_length_sq == 0: return None

    stones_on_path = []
    for stone in all_stones:
        if stone is shooter_stone or stone is target_stone: continue
        stone_pos = np.array(stone.body.position)
        stone_vector = stone_pos - s_pos
        dot_product = np.dot(stone_vector, shot_vector)
        if dot_product <= 0: continue
        
        # ZeroDivisionError 방지를 위해 shot_length_sq가 0에 가까운 경우 처리
        if np.isclose(shot_length_sq, 0): continue
        perpendicular_dist = np.abs(np.cross(shot_vector, stone_vector)) / np.sqrt(shot_length_sq)

        if perpendicular_dist < stone_radius * 2:
            stones_on_path.append((dot_product, stone))

    if not stones_on_path: return None
    stones_on_path.sort(key=lambda x: x[0], reverse=True)
    return stones_on_path[0][1]

def get_intelligent_random_shot(player_stones, opponent_stones, all_stones):
    player_color_tuple = player_stones[0].color[:3]
    safe_shots = []
    for p_idx, shooter in enumerate(player_stones):
        for target in opponent_stones:
            last_stone = get_last_stone_on_path(shooter, target, all_stones, player_color_tuple)
            if last_stone is None or last_stone.color[:3] != player_color_tuple:
                safe_shots.append({'p_idx': p_idx, 'shooter': shooter, 'target': target})

    if not safe_shots: return None

    chosen_shot = random.choice(safe_shots)
    p_idx = chosen_shot['p_idx']
    diff = chosen_shot['target'].body.position - chosen_shot['shooter'].body.position
    angle = diff.angle
    return (p_idx, angle, 1.0)

def model_c_action(stones, player_color):
    if player_color == "black":
        player_stones = [s for s in stones if s.color[:3] == (0, 0, 0)]
        opponent_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]
    else:
        player_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]
        opponent_stones = [s for s in stones if s.color[:3] == (0, 0, 0)]

    if not player_stones or not opponent_stones: return None

    # 1 vs 2 스플릿 샷 로직
    if len(player_stones) == 1 and len(opponent_stones) == 2:
        try:
            stone1, stone2 = opponent_stones[0], opponent_stones[1]
            target_pos = (stone1.body.position + stone2.body.position) / 2
            stone_to_shoot = player_stones[0]
            direction_vector = target_pos - stone_to_shoot.body.position
            distance_to_target = direction_vector.length

            if distance_to_target > 0:
                rule_idx = 0
                rule_angle = direction_vector.angle
                
                # [수정] 거리에 비례하여 힘을 계산하는 로직으로 복원
                DAMPING = 0.1
                LN_DAMPING = np.log(DAMPING)
                MIN_IMPULSE = 20.0
                MAX_IMPULSE = 2000.0
                required_impulse = -distance_to_target * LN_DAMPING
                normalized_force = (required_impulse - MIN_IMPULSE) / (MAX_IMPULSE - MIN_IMPULSE)
                rule_force = np.clip(normalized_force, 0.0, 1.0)
                
                return (rule_idx, rule_angle, rule_force)

        except (ValueError, IndexError):
            pass
    
    # 그 외의 경우, 지능적인 무작위 공격
    best_shot = get_intelligent_random_shot(player_stones, opponent_stones, stones)
    if best_shot:
        return best_shot

    # 안전한 샷이 하나도 없을 경우의 비상 로직 (완전 무작위)
    p_idx = random.randrange(len(player_stones))
    target = random.choice(opponent_stones)
    diff = target.body.position - player_stones[p_idx].body.position
    angle = diff.angle
    return (p_idx, angle, 1.0)
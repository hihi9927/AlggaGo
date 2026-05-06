import numpy as np
import random
from pymunk import Vec2d
from physics import WIDTH, STONE_RADIUS

def is_path_blocked(shooter_stone, target_pos, all_stones, ignore_list):
    """슈터와 타겟 사이에 다른 돌이 있는지 확인하는 경로 탐색 함수"""
    s_pos = shooter_stone.body.position
    t_pos = target_pos
    shot_vector = t_pos - s_pos
    shot_length_sq = shot_vector.dot(shot_vector)

    if shot_length_sq == 0:
        return True

    for stone in all_stones:
        if stone in ignore_list:
            continue

        stone_pos = stone.body.position
        stone_vector = stone_pos - s_pos
        
        projection = stone_vector.dot(shot_vector)
        if projection < 0 or projection > shot_length_sq:
            continue

        perpendicular_dist_sq = stone_vector.dot(stone_vector) - (projection**2 / shot_length_sq)
        
        if perpendicular_dist_sq < (STONE_RADIUS * 2)**2:
            return True

    return False

def get_regular_action(player_stones, opponent_stones):
    """[✅ 최종 수정] '일반 공격' 기준 행동: 랜덤 돌 선택, 랜덤 상대 공격"""
    if not player_stones or not opponent_stones:
        return None

    # 1. 쏠 돌을 무작위로 선택
    shooter_stone = random.choice(player_stones)
    # 선택된 돌의 원래 인덱스를 찾음
    shooter_index = player_stones.index(shooter_stone)
    
    # 2. 공격할 상대 돌을 무작위로 선택
    target_stone = random.choice(opponent_stones)
    
    # 3. 샷 계산
    direction_vec = (target_stone.body.position - shooter_stone.body.position).normalized()
    target_pos = target_stone.body.position + direction_vec * STONE_RADIUS
    
    final_vec = target_pos - shooter_stone.body.position
    angle = final_vec.angle
    force = 1.0 # 일반 공격은 항상 최대 힘
    
    return (shooter_index, angle, force)

def get_split_shot_action(player_stones, opponent_stones):
    if len(opponent_stones) < 2:
        return None

    best_shot = None
    min_dist_sum = float('inf')
    best_shot_index = -1
    all_stones = player_stones + opponent_stones

    for i, p_stone in enumerate(player_stones):
        opp_stones_sorted = sorted(opponent_stones, key=lambda o: (o.body.position - p_stone.body.position).length)
        o1, o2 = opp_stones_sorted[0], opp_stones_sorted[1]

        # 1. 틈새 너비 확인
        gap_distance = (o1.body.position - o2.body.position).length
        if gap_distance <= STONE_RADIUS * 4:
            continue
        
        target_pos = (o1.body.position + o2.body.position) / 2
        
        # 2. [수정] 모든 장애물(목표물 포함)을 대상으로 경로 검사
        # ignore_list에서 o1, o2를 제거하여, 목표물이 경로를 막는 경우도 감지하도록 수정
        path_blocked = is_path_blocked(p_stone, target_pos, all_stones, ignore_list=[p_stone])
        if path_blocked:
            continue

        # 모든 검사를 통과한 경우에만 최적의 샷을 계산
        direction_vec = target_pos - p_stone.body.position
        dist_sum = (p_stone.body.position - o1.body.position).length + \
                   (p_stone.body.position - o2.body.position).length
        
        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            angle = direction_vec.angle
            distance_to_target = direction_vec.length

            DAMPING = 0.1; LN_DAMPING = np.log(DAMPING)
            MIN_IMPULSE = 20.0; MAX_IMPULSE = 2000.0
            required_impulse = -distance_to_target * LN_DAMPING
            normalized_force = (required_impulse - MIN_IMPULSE) / (MAX_IMPULSE - MIN_IMPULSE)
            force = np.clip(normalized_force, 0.0, 1.0)
            
            best_shot_index = i
            best_shot_params = (angle, force)

    if best_shot_index != -1:
        angle, force = best_shot_params
        best_shot = (best_shot_index, angle, force)

    return best_shot
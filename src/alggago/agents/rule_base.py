import numpy as np
import random
from alggago.physics import STONE_RADIUS


def _is_path_blocked(shooter_stone, target_pos, all_stones, ignore_list):
    s_pos = shooter_stone.body.position
    shot_vector = target_pos - s_pos
    shot_length_sq = shot_vector.dot(shot_vector)
    if shot_length_sq == 0:
        return True
    for stone in all_stones:
        if stone in ignore_list:
            continue
        stone_vector = stone.body.position - s_pos
        projection = stone_vector.dot(shot_vector)
        if projection < 0 or projection > shot_length_sq:
            continue
        perp_sq = stone_vector.dot(stone_vector) - (projection ** 2 / shot_length_sq)
        if perp_sq < (STONE_RADIUS * 2) ** 2:
            return True
    return False


def _is_own_stone_blocking(shooter, target, own_stones):
    """경로 위에 내 돌이 있으면 True"""
    s_pos = shooter.body.position
    t_pos = target.body.position
    shot_vec = t_pos - s_pos
    shot_len_sq = shot_vec.dot(shot_vec)
    if shot_len_sq == 0:
        return True
    for stone in own_stones:
        if stone is shooter:
            continue
        sv = stone.body.position - s_pos
        proj = sv.dot(shot_vec)
        if proj <= 0 or proj >= shot_len_sq:
            continue
        perp_sq = sv.dot(sv) - (proj * proj / shot_len_sq)
        if perp_sq < (STONE_RADIUS * 2) ** 2:
            return True
    return False


def get_regular_action(player_stones, opponent_stones):
    if not player_stones or not opponent_stones:
        return None

    # 내 돌이 경로를 막지 않는 유효한 (idx, shooter, target) 조합 수집
    valid = [
        (i, shooter, target)
        for i, shooter in enumerate(player_stones)
        for target in opponent_stones
        if not _is_own_stone_blocking(shooter, target, player_stones)
    ]

    # 유효한 샷이 없으면 전체 조합에서 랜덤 선택 (폴백)
    if not valid:
        shooter = random.choice(player_stones)
        target  = random.choice(opponent_stones)
        valid   = [(player_stones.index(shooter), shooter, target)]

    idx, shooter, target = random.choice(valid)
    angle = (target.body.position - shooter.body.position).angle
    return (idx, angle, 1.0)


def get_split_shot_action(player_stones, opponent_stones):
    if len(opponent_stones) < 2:
        return None
    all_stones = player_stones + opponent_stones
    best_shot = None
    min_dist_sum = float('inf')
    for i, p_stone in enumerate(player_stones):
        opp_sorted = sorted(opponent_stones, key=lambda o: (o.body.position - p_stone.body.position).length)
        o1, o2 = opp_sorted[0], opp_sorted[1]
        if (o1.body.position - o2.body.position).length <= STONE_RADIUS * 4:
            continue
        target_pos = (o1.body.position + o2.body.position) / 2
        if _is_path_blocked(p_stone, target_pos, all_stones, ignore_list=[p_stone]):
            continue
        direction_vec = target_pos - p_stone.body.position
        dist_sum = (p_stone.body.position - o1.body.position).length + \
                   (p_stone.body.position - o2.body.position).length
        if dist_sum < min_dist_sum:
            min_dist_sum = dist_sum
            required_impulse = -direction_vec.length * np.log(0.1)
            force = float(np.clip((required_impulse - 20.0) / (2000.0 - 20.0), 0.0, 1.0))
            best_shot = (i, direction_vec.angle, force)
    return best_shot

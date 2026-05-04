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


def get_regular_action(player_stones, opponent_stones):
    if not player_stones or not opponent_stones:
        return None
    shooter = random.choice(player_stones)
    idx = player_stones.index(shooter)
    target = random.choice(opponent_stones)
    direction = (target.body.position - shooter.body.position).normalized()
    target_pos = target.body.position + direction * STONE_RADIUS
    angle = (target_pos - shooter.body.position).angle
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

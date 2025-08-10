import numpy as np

def rule_based_black_action(stones):
    """
    가장 가까운 백돌을 향해 흑돌이 정확하게 조준해서 때림
    """
    black_stones = [s for s in stones if s.color[:3] == (0, 0, 0)]
    white_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]

    if not black_stones or not white_stones:
        return None

    best_shot = None
    min_dist = float('inf')

    for b_idx, b in enumerate(black_stones):
        b_pos = b.body.position

        for w in white_stones:
            w_pos = w.body.position
            diff = w_pos - b_pos
            dist = diff.length

            if dist < min_dist:
                min_dist = dist
                angle = np.arctan2(diff.y, diff.x)
                best_shot = (b_idx, angle, 1.0)

    return best_shot

def rule_based_white_action(stones):
    """
    가장 가까운 흑돌을 향해 백돌이 정확하게 조준해서 때림
    """
    black_stones = [s for s in stones if s.color[:3] == (0, 0, 0)]
    white_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]

    if not black_stones or not white_stones:
        return None

    best_shot = None
    min_dist = float('inf')

    for w_idx, w in enumerate(white_stones):
        w_pos = w.body.position

        for b in black_stones:
            b_pos = b.body.position
            diff = b_pos - w_pos
            dist = diff.length

            if dist < min_dist:
                min_dist = dist
                angle = np.arctan2(diff.y, diff.x)
                best_shot = (w_idx, angle, 1.0)

    return best_shot

def rule_based_action(stones, player_color):
    """
    현재 플레이어 색상에 따라 적절한 rule-based 행동을 반환
    """
    if player_color == "black":
        return rule_based_black_action(stones)
    else:  # white
        return rule_based_white_action(stones)

def get_opponent_fn():
    return rule_based_black_action

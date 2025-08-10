# physics.py
import pymunk
from pymunk import Vec2d
import random
import math 

# 설정
WIDTH, HEIGHT = 800, 800
MARGIN = 50
STONE_RADIUS = 25
STONE_MASS = 1

# 드래그 관련 상수
MAX_DRAG_LENGTH = 100    # (픽셀) 최대 허용 드래그 거리
FORCE_MULTIPLIER = 20     # (임펄스/픽셀) 드래그 길이당 곱해줄 배율
MIN_FORCE = 20          # (임펄스) 최소 기본 힘

def create_stone(space, position, color):
    moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
    body = pymunk.Body(STONE_MASS, moment)
    body.position = position
    shape = pymunk.Circle(body, STONE_RADIUS)
    shape.collision_type = 1
    shape.elasticity = 1.0
    shape.friction = 0.9
    shape.color = (*color, 255)
    space.add(body, shape)
    return shape

def move_random_white(stones):
    white_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]
    if not white_stones:
        return
    stone = random.choice(white_stones)

    angle = random.uniform(0, 2 * 3.14159)
    direction = Vec2d(1, 0).rotated(angle)
    impulse = direction * random.uniform(200, 500)
    stone.body.apply_impulse_at_world_point(impulse, stone.body.position)

# 첫 번째 reset_stones 함수 (주석 처리됨)
# def reset_stones(space, stones):
#     from physics import create_stone  # 재귀 방지용 (또는 top-level에서 import 하세요)
#     for shape in stones:
#         space.remove(shape, shape.body)
#     stones.clear()
# 
#     cell = (WIDTH - 2 * MARGIN) / 18
#     x_positions = [MARGIN + cell * i for i in range(3, 16, 4)]
#     y_top = MARGIN + STONE_RADIUS * 2 + 66.5
#     y_bottom = HEIGHT - MARGIN - STONE_RADIUS * 2 - 66.5
# 
#     for x in x_positions:
#         stones.append(create_stone(space, (x, y_top), color=(255, 255, 255)))
#     for x in x_positions:
#         stones.append(create_stone(space, (x, y_bottom), color=(0, 0, 0)))
# 
#     return 4, 4

def reset_stones(space, stones):
    """모드 1, 2, 3용 고정 배치 함수"""
    from physics import create_stone  # 재귀 방지용 (또는 top-level import)

    for shape in stones:
        space.remove(shape, shape.body)
    stones.clear()

    cell = (WIDTH - 2 * MARGIN) / 18
    x_positions = [MARGIN + cell * i for i in range(3, 16, 4)]
    y_top = MARGIN + STONE_RADIUS * 2 + 66.5
    y_bottom = HEIGHT - MARGIN - STONE_RADIUS * 2 - 66.5

    # --- 백돌 (고정 위치) ---
    for x in x_positions:
        stones.append(create_stone(space, (x, y_top), color=(255, 255, 255)))

    # --- 흑돌 (일직선 배치) ---
    # 하단에 4개 흑돌을 일직선으로 배치
    for x in x_positions:
        stones.append(create_stone(space, (x, y_bottom), color=(0, 0, 0)))

    return 4, 4  # 백돌, 흑돌 개수 반환


def reset_stones_random(space, stones):
    """모드 4용 랜덤 배치 함수"""
    from physics import create_stone  # 재귀 방지용 (또는 top-level import)
    import numpy as np

    for shape in stones:
        space.remove(shape, shape.body)
    stones.clear()

    cell = (WIDTH - 2 * MARGIN) / 18
    x_positions = [MARGIN + cell * i for i in range(3, 16, 4)]
    y_top = MARGIN + STONE_RADIUS * 2 + 66.5
    y_bottom = HEIGHT - MARGIN - STONE_RADIUS * 2 - 66.5

    # --- 백돌 (고정 위치) ---
    for x in x_positions:
        stones.append(create_stone(space, (x, y_top), color=(255, 255, 255)))

    # --- 흑돌 (충돌 없이 랜덤 배치) ---
    placed_positions = []
    max_attempts = 100

    def is_far_enough(x, y, placed, min_dist):
        for px, py in placed:
            if np.linalg.norm([x - px, y - py]) < min_dist:
                return False
        return True

    while len(placed_positions) < 4 and max_attempts > 0:
        x = np.random.uniform(MARGIN + STONE_RADIUS, WIDTH - MARGIN - STONE_RADIUS)
        y = y_bottom + np.random.uniform(-40, 40)

        if is_far_enough(x, y, placed_positions, STONE_RADIUS * 3.5):
            placed_positions.append((x, y))
            stones.append(create_stone(space, (x, y), color=(0, 0, 0)))
        else:
            max_attempts -= 1

    return 4, 4  # 백돌, 흑돌 개수 반환


def reset_stones_custom(space, stones, custom_black_positions):
    """커스텀 흑돌 위치로 돌을 배치하는 함수"""
    from physics import create_stone  # 재귀 방지용 (또는 top-level import)
    
    for shape in stones:
        space.remove(shape, shape.body)
    stones.clear()

    cell = (WIDTH - 2 * MARGIN) / 18
    x_positions = [MARGIN + cell * i for i in range(3, 16, 4)]
    y_top = MARGIN + STONE_RADIUS * 2 + 66.5

    # --- 백돌 (고정 위치) ---
    for x in x_positions:
        stones.append(create_stone(space, (x, y_top), color=(255, 255, 255)))

    # --- 흑돌 (커스텀 위치) ---
    for x, y in custom_black_positions:
        stones.append(create_stone(space, (x, y), color=(0, 0, 0)))

    return 4, 4  # 백돌, 흑돌 개수 반환


def all_stones_stopped(stones, threshold=5):
    """모든 돌의 속도가 충분히 느릴 때 True 반환"""
    for shape in stones:
        if shape.body.velocity.length > threshold:
            return False
    return True

def scale_force(force_normalized):
    """
    AI가 예측한 [0.0 ~ 1.0] force 값을 실제 물리 임펄스 값으로 변환
    """
    MAX_FORCE = MAX_DRAG_LENGTH * FORCE_MULTIPLIER
    return MIN_FORCE + force_normalized * (MAX_FORCE - MIN_FORCE)

# [추가] physics.py 파일에 아래 함수를 추가하세요.

def reset_stones_beginner(space, stones):
    """모드 6 (초보자용) 배치 함수 (백돌 4개 vs 흑돌 6개)"""
    from physics import create_stone

    for shape in stones:
        if shape in space.shapes:
            space.remove(shape, shape.body)
    stones.clear()

    cell = (WIDTH - 2 * MARGIN) / 18
    
    # --- 백돌 4개 (고정 위치) ---
    x_positions_white = [MARGIN + cell * i for i in range(3, 16, 4)]
    y_top = MARGIN + STONE_RADIUS * 2 + 66.5
    for x in x_positions_white:
        stones.append(create_stone(space, (x, y_top), color=(255, 255, 255)))

    # --- 흑돌 6개 (0.5칸 오른쪽 이동) ---
    # 18개 격자 중에서 중앙에 맞춰 넓게 분산하여 6개 위치 선택 (1.5, 4.5, 7.5, 10.5, 13.5, 16.5)
    x_positions_black = [MARGIN + cell * (i + 0.5) for i in [1, 4, 7, 10, 13, 16]]
    y_bottom = HEIGHT - MARGIN - STONE_RADIUS * 2 - 66.5
    for x in x_positions_black:
        stones.append(create_stone(space, (x, y_bottom), color=(0, 0, 0)))

    return 4, 6 # 백돌, 흑돌 개수 반환
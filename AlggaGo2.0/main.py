import pygame
import pymunk
import random
from pymunk import Vec2d
import time
import numpy as np
import csv
import os
import math
from datetime import datetime

from physics import (
    create_stone,
    reset_stones,
    reset_stones_random,
    reset_stones_custom,
    all_stones_stopped,
    reset_stones_beginner,scale_force,
    WIDTH, HEIGHT, MARGIN,
    STONE_RADIUS, STONE_MASS,
    MAX_DRAG_LENGTH, FORCE_MULTIPLIER, MIN_FORCE,
)
# agent.py에서 MainRLAgent, apply_action_to_stone, choose_ai를 임포트
from agent import MainRLAgent, apply_action_to_stone
# env.py는 _get_obs() 메서드를 활용하기 위해 임포트
from opponent_c import model_c_action
from env import AlggaGoEnv

# 프로젝트 루트/파일 기준 경로 유틸
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(BASE_DIR, *parts)

print("Pymunk version:", pymunk.version)

class ModelCAgent:
    """Model C의 규칙 기반 로직을 실행하는 에이전트 클래스"""
    def select_action(self, stones, player_color):
        """
        현재 돌 상태를 받아 Model C의 행동을 직접 반환합니다.
        반환값: (돌 인덱스, 각도, 힘) 튜플 또는 None
        """
        return model_c_action(stones, player_color)
    
def get_font(size):
    """한글을 지원하는 폰트를 반환하는 함수"""
    # 한글 폰트 우선순위
    korean_fonts = [
        "Malgun Gothic",  # Windows 기본 한글 폰트
        "NanumGothic",    # 나눔고딕
        "NanumBarunGothic",  # 나눔바른고딕
        "Dotum",          # 돋움
        "Batang",         # 바탕
        "Gulim",          # 굴림
        "Arial Unicode MS",  # Arial Unicode
        "Arial"           # Arial (fallback)
    ]
    
    for font_name in korean_fonts:
        try:
            return pygame.font.SysFont(font_name, size)
        except:
            continue
    
    # 모든 폰트가 실패하면 기본 폰트 사용
    return pygame.font.SysFont("arial", size)


def create_obs_for_player(stones, current_player, game_mode):
    """
    main.py에서 AI가 사용할 관측을 생성합니다.
    AI 모델은 25차원 입력을 기대하므로, 그에 맞춰 데이터를 생성합니다.
    """
    obs = []
    my_stones = []
    opponent_stones = []

    for shape in stones:
        stone_is_white = shape.color[:3] == (255, 255, 255)
        is_mine = (current_player == "white" and stone_is_white) or \
                  (current_player == "black" and not stone_is_white)
        
        if is_mine:
            my_stones.append(shape)
        else:
            opponent_stones.append(shape)
    
    sorted_stones = my_stones + opponent_stones

    for shape in sorted_stones:
        x, y = shape.body.position
        
        stone_is_white = shape.color[:3] == (255, 255, 255)
        if current_player == "white":
            is_mine = 1.0 if stone_is_white else 0.0
        else:
            is_mine = 1.0 if not stone_is_white else 0.0
        
        if current_player == "black":
            y = HEIGHT - y
        
        obs.extend([float(x), float(y), float(is_mine)])
    
    # AI가 인식할 수 있도록 돌 정보를 8개(24차원)로 제한
    if len(obs) > 24:
        obs = obs[:24]

    # 부족한 부분은 빈 값으로 채움
    while len(obs) < 24:
        obs.extend([-1.0, -1.0, -1.0])

    obs.append(1.0 if current_player == "white" else 0.0)
    
    return np.array(obs, dtype=np.float32)

def predict_action_4d(model, obs, default_strategy=-1.0):
    """
    어떤 모델이든 예측 액션을 항상 (4,)로 맞춰 반환.
    - 4D 모델: 그대로 반환
    - 3D 모델: [strategy, index, angle, force]에서 strategy만 기본값으로 채워 선두에 붙임
    - 그 외: 규칙기반/무작위 대체 등 안전장치
    """
    try:
        action, _ = model.predict(obs, deterministic=True)
    except Exception:
        # 예측 실패 시 규칙 기반 등으로 대체하는 로직을 넣고 싶으면 여기 처리
        # 임시로 0 벡터 반환
        return np.array([default_strategy, 0.0, 0.0, 0.0], dtype=np.float32)

    a = np.ravel(action).astype(np.float32)
    if a.size == 4:
        return a
    elif a.size == 3:
        # 기존(3D): [index, angle, force]라고 가정 → 맨 앞에 전략선택을 기본값으로 붙임
        return np.array([default_strategy, a[0], a[1], a[2]], dtype=np.float32)
    elif a.size > 4:
        return a[:4]
    else:
        # 크기가 맞지 않으면 안전값
        return np.array([default_strategy, 0.0, 0.0, 0.0], dtype=np.float32)

def get_default_ai_agent():
    """기본 AI 에이전트를 반환하는 함수"""
    model_path = "rl_models_competitive/AlggaGo1.5_324.zip"
    if os.path.exists(model_path):
        print(f"[AI 선택] 기본 모델 '{model_path}'을(를) 사용합니다.")
        return MainRLAgent(model_path=model_path)
    else:
        print(f"[AI 선택] 기본 모델을 찾을 수 없습니다. 순수 Rule-based를 사용합니다.")
        return MainRLAgent(model_path=None)

def get_ai_agent(game_mode, win_streak=0):
    """
    게임 모드와 연승 횟수에 따라 적절한 AI 에이전트를 반환합니다.
    """
    # 1번 모드 (알까기 챔피언십)의 경우, 연승에 따라 상대가 변경됨
    if game_mode == 1:
        if 0 <= win_streak <= 2:  # 1~3라운드
            model_path = rel_path("rl_models_competitive", "AlggaGo1.0_180.zip")
            agent_name = "AlggaGo1.0"
        elif 3 <= win_streak <= 5:  # 4~6라운드
            print(f"[AI 변경] {win_streak + 1}라운드 상대는 Model C 입니다.")
            return ModelCAgent()
        else:  # 7라운드 이상
            model_path = "rl_models_competitive/AlggaGo2.0.zip"
            agent_name = "AlggaGo2.0"
    # 2번 모드는 항상 AlggaGo2.0
    elif game_mode == 2:
        model_path = "rl_models_competitive/AlggaGo1.72_537.zip"
        agent_name = "AlggaGo2.0"
    # 그 외 모드는 AlggaGo1.0
    else:
        model_path = "rl_models_competitive/AlggaGo1.0_180.zip"
        agent_name = "AlggaGo1.0"

    if os.path.exists(model_path):
        # game_mode 1일 때만 라운드 표시, 그 외엔 일반 메시지
        if game_mode == 1:
            print(f"[AI 변경] {win_streak + 1}라운드 상대는 {agent_name} 입니다.")
        else:
            print(f"[AI 선택] 모델 '{model_path}'을(를) 사용합니다.")
        return MainRLAgent(model_path=model_path)
    else:
        print(f"[AI 경고] 모델 파일({model_path})을 찾을 수 없습니다. Rule-based로 대체합니다.")
        return MainRLAgent(model_path=None)

def get_top_players():
    """CSV 파일에서 최다연승 순위를 가져오는 함수"""
    csv_filename = rel_path("game_records.csv")
    if not os.path.exists(csv_filename):
        return []
    
    # 닉네임별 최고 연승 기록 수집
    player_records = {}
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nickname = row['nickname']
            win_streak = int(row['win_streak'])
            
            if nickname not in player_records:
                player_records[nickname] = win_streak
            else:
                player_records[nickname] = max(player_records[nickname], win_streak)
    
    # 연승 순으로 정렬 (내림차순)
    sorted_players = sorted(player_records.items(), key=lambda x: x[1], reverse=True)
    return sorted_players[:10]  # 상위 10명만 반환


def show_ranking(screen, clock):
    """최다연승 순위표를 보여주는 함수"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    top_players = get_top_players()
    
    start_time = time.time()

    while True:
        screen.fill((50, 50, 50))
        
        # 제목
        title_surface = font_large.render("최다 연승 순위", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(WIDTH // 2, 80))
        screen.blit(title_surface, title_rect)
        
        if not top_players:
            no_data_surface = font_medium.render("아직 기록이 없습니다.", True, (200, 200, 200))
            no_data_rect = no_data_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(no_data_surface, no_data_rect)
        else:
            # 순위표 표시
            for i, (nickname, win_streak) in enumerate(top_players):
                y_pos = 180 + i * 40
                
                # 순위
                rank_text = f"{i+1}."
                rank_surface = font_medium.render(rank_text, True, (144, 238, 144))
                rank_rect = rank_surface.get_rect(center=(WIDTH // 2 - 150, y_pos))
                screen.blit(rank_surface, rank_rect)
                
                # 닉네임
                name_surface = font_medium.render(nickname, True, (255, 255, 255))
                name_rect = name_surface.get_rect(center=(WIDTH // 2, y_pos))
                screen.blit(name_surface, name_rect)
                
                # 연승
                streak_text = f"{win_streak}연승"
                streak_surface = font_medium.render(streak_text, True, (255, 255, 255))
                streak_rect = streak_surface.get_rect(center=(WIDTH // 2 + 150, y_pos))
                screen.blit(streak_surface, streak_rect)
        
        # 안내 메시지
        hint_surface = font_small.render("클릭하여 돌아가기", True, (200, 200, 200))
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if time.time() - start_time < 1.0:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return None
                else:
                    return "MODE_SELECT"

            elif event.type == pygame.MOUSEBUTTONDOWN: # 이 부분 추가
                if time.time() - start_time < 1.0:
                    continue
                return "MODE_SELECT"
        
        clock.tick(60)
    
    return False


def get_nickname_input(screen, clock):
    """사용자로부터 닉네임을 입력받는 함수"""
    pygame.init()
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    nickname = ""
    input_active = True
    
    while input_active:
        screen.fill((50, 50, 50))
        
        # 제목
        title_surface = font_large.render("AlggaGo 2.0", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(WIDTH // 2, HEIGHT // 3 + 10))
        screen.blit(title_surface, title_rect)
        
        # 닉네임 입력 안내
        instruction_text = "닉네임을 입력하세요:"
        instruction_surface = font_medium.render(instruction_text, True, (255, 255, 255))
        instruction_rect = instruction_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 -20))
        screen.blit(instruction_surface, instruction_rect)
        
        # 입력창
        input_box = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 10, 300, 40)
        pygame.draw.rect(screen, (255, 255, 255), input_box, 2)
        
        # 입력된 텍스트
        if nickname:
            text_surface = font_medium.render(nickname, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=input_box.center)
            screen.blit(text_surface, text_rect)
        
        # 안내 메시지
        hint_surface = font_small.render("Enter 키를 눌러 다음", True, (200, 200, 200))
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4 -50))
        screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and nickname.strip():
                    return nickname.strip()
                elif event.key == pygame.K_BACKSPACE:
                    nickname = nickname[:-1]
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif len(nickname) < 10:  # 최대 10자로 제한
                    # 한글 입력 지원 - 모든 유니코드 문자 허용
                    if event.unicode:
                        nickname += event.unicode
        
        clock.tick(60)
    
    return None

def show_controls_screen(screen, clock):
    """조작 안내를 보여주는 별도의 화면 (0.5초 대기 후 초기화)"""
    font_title = get_font(24)
    font_large = get_font(36)
    font_medium = get_font(22)
    font_small = get_font(18)
    running = True

    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_duration = 30
    fade_step = 255 // fade_duration

    tutorial_space = pymunk.Space()
    tutorial_space.gravity = (0, 0)
    tutorial_space.damping = 0.1
    tutorial_stones = []
    tutorial_width = 350
    tutorial_height = 500
    tutorial_x = (WIDTH - tutorial_width) // 2
    tutorial_y = 90
    tutorial_area = pygame.Rect(tutorial_x, tutorial_y, tutorial_width, tutorial_height)
    
    show_failure_screen = False
    tutorial_dragging = False
    arrow_visible = True
    tutorial_path_cache = {}
    
    # 0.5초 대기용 타이머 변수
    reset_timer = None

    def reset_tutorial():
        nonlocal show_failure_screen, tutorial_dragging, arrow_visible, tutorial_path_cache, reset_timer
        show_failure_screen, tutorial_dragging, arrow_visible = False, False, True
        reset_timer = None # 타이머 초기화
        
        for body in list(tutorial_space.bodies): tutorial_space.remove(body)
        for shape in list(tutorial_space.shapes): tutorial_space.remove(shape)
        tutorial_stones.clear()
        
        player_body.position = (tutorial_x + tutorial_width // 2, tutorial_y + tutorial_height - 90)
        player_body.velocity = (0, 0)
        tutorial_space.add(player_body, player_shape)
        tutorial_stones.append(player_shape)
        
        opponent_body.position = (tutorial_x + tutorial_width // 2, tutorial_y + 90)
        opponent_body.velocity = (0, 0)
        tutorial_space.add(opponent_body, opponent_shape)
        tutorial_stones.append(opponent_shape)
        tutorial_path_cache.clear()
    
    player_moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
    player_body = pymunk.Body(STONE_MASS, player_moment)
    player_shape = pymunk.Circle(player_body, STONE_RADIUS)
    player_shape.elasticity = 1.0; player_shape.friction = 0.9
    player_shape.color = (0, 0, 0, 255)
    
    opponent_moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
    opponent_body = pymunk.Body(STONE_MASS, opponent_moment)
    opponent_shape = pymunk.Circle(opponent_body, STONE_RADIUS)
    opponent_shape.elasticity = 1.0; opponent_shape.friction = 0.9
    opponent_shape.color = (255, 255, 255, 255)
    
    reset_tutorial()
    
    tutorial_drag_shape = None
    tutorial_drag_start = Vec2d(0, 0)
    arrow_time = 0
    arrow_animation_speed = 0.1

    while running:
        screen.fill((50, 50, 50))
        title_surf = font_title.render("조작 안내", True, (255, 255, 255))
        screen.blit(title_surf, title_surf.get_rect(topleft=(20, 20)))
        pygame.draw.rect(screen, (210, 180, 140), tutorial_area)
        pygame.draw.rect(screen, (0, 0, 0), tutorial_area, 3)
        grid_spacing = 50
        for x in range(tutorial_x, tutorial_x + tutorial_width + 1, grid_spacing):
            line_color = (139, 69, 19) if x in (tutorial_x, tutorial_x + tutorial_width) else (160, 82, 45)
            pygame.draw.line(screen, line_color, (x, tutorial_y), (x, tutorial_y + tutorial_height), 2 if x in (tutorial_x, tutorial_x + tutorial_width) else 1)
        for y in range(tutorial_y, tutorial_y + tutorial_height + 1, grid_spacing):
            line_color = (139, 69, 19) if y in (tutorial_y, tutorial_y + tutorial_height) else (160, 82, 45)
            pygame.draw.line(screen, line_color, (tutorial_x, y), (tutorial_x + tutorial_width, y), 2 if y in (tutorial_y, tutorial_y + tutorial_height) else 1)
        for stone in tutorial_stones:
            pygame.draw.circle(screen, stone.color[:3], (int(stone.body.position.x), int(stone.body.position.y)), STONE_RADIUS)

        if arrow_visible and not tutorial_dragging:
            arrow_time += arrow_animation_speed
            arrow_offset = int(10 * abs(math.sin(arrow_time)))
            player_pos = player_body.position
            arrow_y = int(player_pos.y) - STONE_RADIUS - 30 + arrow_offset
            arrow_points = [(int(player_pos.x) - 15, arrow_y + 20), (int(player_pos.x) + 15, arrow_y + 20), (int(player_pos.x), arrow_y)]
            pygame.draw.polygon(screen, (255, 255, 0), arrow_points)
            pygame.draw.polygon(screen, (255, 165, 0), arrow_points, 2)
            instruction_surface = font_medium.render("검은 돌을 아래로 당겨서 흰 돌을 맞추세요!", True, (255, 255, 255))
            screen.blit(instruction_surface, instruction_surface.get_rect(center=(WIDTH // 2, tutorial_y + tutorial_height + 30)))

        if tutorial_dragging and tutorial_drag_shape:
            mouse_pos = Vec2d(*pygame.mouse.get_pos())
            raw_vec = mouse_pos - tutorial_drag_start
            length = raw_vec.length
            unit = raw_vec.normalized() if length > 0 else Vec2d(0, 0)
            center = tutorial_drag_shape.body.position
            end_pos = center + unit * min(length, MAX_DRAG_LENGTH)
            pygame.draw.line(screen, (255, 0, 0), (int(center.x), int(center.y)), (int(end_pos.x), int(end_pos.y)), 2)
            
            if length > 0:
                actual_raw_vec = tutorial_drag_start - mouse_pos
                if actual_raw_vec.length > MAX_DRAG_LENGTH:
                    actual_raw_vec = actual_raw_vec.normalized() * MAX_DRAG_LENGTH
                cache_key = (int(actual_raw_vec.x), int(actual_raw_vec.y))
                if cache_key not in tutorial_path_cache:
                    temp_space = pymunk.Space(); temp_space.damping = tutorial_space.damping
                    temp_opponent_body = pymunk.Body(STONE_MASS, opponent_moment); temp_opponent_body.position = opponent_body.position
                    temp_opponent_shape = pymunk.Circle(temp_opponent_body, STONE_RADIUS); temp_opponent_shape.elasticity = 1.0; temp_opponent_shape.friction = 0.9
                    temp_space.add(temp_opponent_body, temp_opponent_shape)
                    temp_player_body = pymunk.Body(STONE_MASS, player_moment); temp_player_body.position = center
                    temp_player_shape = pymunk.Circle(temp_player_body, STONE_RADIUS); temp_player_shape.elasticity = 1.0; temp_player_shape.friction = 0.9
                    temp_space.add(temp_player_body, temp_player_shape)
                    impulse = actual_raw_vec.normalized() * (actual_raw_vec.length * FORCE_MULTIPLIER + MIN_FORCE)
                    temp_player_body.apply_impulse_at_world_point(impulse, temp_player_body.position)
                    path_points = [temp_player_body.position]
                    for _ in range(120):
                        temp_space.step(1/60.0)
                        path_points.append(temp_player_body.position)
                        if temp_player_body.velocity.length < 5: break
                    tutorial_path_cache[cache_key] = path_points
                path_points = tutorial_path_cache[cache_key]
                if len(path_points) > 1:
                    pygame.draw.lines(screen, (0, 200, 255), False, path_points, 2)
                if path_points:
                    pygame.draw.circle(screen, (0, 200, 255), (int(path_points[-1].x), int(path_points[-1].y)), 4)

        if show_failure_screen:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 150)); screen.blit(overlay, (0, 0))
            message_box = pygame.Rect(WIDTH//2-150, HEIGHT//2-80, 300, 160); pygame.draw.rect(screen, (139, 34, 34), message_box); pygame.draw.rect(screen, (255, 255, 255), message_box, 2)
            failure_text = font_large.render("실패!", True, (255, 255, 255)); screen.blit(failure_text, failure_text.get_rect(center=(WIDTH//2, HEIGHT//2-40)))
            subtitle_text = font_small.render("다시 한번 시도해보세요", True, (255, 220, 220)); screen.blit(subtitle_text, subtitle_text.get_rect(center=(WIDTH//2, HEIGHT//2)))
            retry_button_rect = pygame.Rect(WIDTH//2-80, HEIGHT//2+30, 160, 40); pygame.draw.rect(screen, (70, 130, 180), retry_button_rect); pygame.draw.rect(screen, (255, 255, 255), retry_button_rect, 2)
            retry_text = font_medium.render("다시하기", True, (255, 255, 255)); screen.blit(retry_text, retry_text.get_rect(center=retry_button_rect.center))
            
        if not show_failure_screen:
            hint_surf = font_small.render("클릭하여 돌아가기", True, (200, 200, 200))
            screen.blit(hint_surf, hint_surf.get_rect(center=(WIDTH // 2, HEIGHT - 80)))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = Vec2d(*event.pos)
                if show_failure_screen:
                    retry_button_rect = pygame.Rect(WIDTH//2-80, HEIGHT//2+30, 160, 40)
                    if retry_button_rect.collidepoint(event.pos): reset_tutorial()
                else:
                    if tutorial_area.collidepoint(mouse_pos):
                        if reset_timer is None: # 리셋 대기 중에는 돌을 움직일 수 없음
                            for stone in tutorial_stones:
                                if stone.color[:3] == (0, 0, 0) and (stone.body.position - mouse_pos).length <= STONE_RADIUS:
                                    tutorial_dragging, tutorial_drag_shape, tutorial_drag_start, arrow_visible = True, stone, stone.body.position, False
                                    break
                    else:
                        running = False
            
            elif event.type == pygame.MOUSEBUTTONUP and tutorial_dragging:
                drag_end = Vec2d(*event.pos)
                raw_vec = tutorial_drag_start - drag_end
                if raw_vec.length > MAX_DRAG_LENGTH:
                    raw_vec = raw_vec.normalized() * MAX_DRAG_LENGTH
                if raw_vec.length > 0:
                    impulse = raw_vec.normalized() * (raw_vec.length * FORCE_MULTIPLIER + MIN_FORCE)
                    tutorial_drag_shape.body.apply_impulse_at_world_point(impulse, tutorial_drag_shape.body.position)
                tutorial_dragging, tutorial_drag_shape, arrow_visible = False, None, False
                if len(tutorial_path_cache) > 5: tutorial_path_cache.clear()

        # --- 리셋 로직 수정 (0.5초 대기) ---
        # 1. 리셋 타이머가 설정되었는지 확인
        if reset_timer is not None:
            # 2. 현재 시간과 타이머 시작 시간을 비교하여 0.5초(500ms)가 지났는지 확인
            if pygame.time.get_ticks() - reset_timer >= 750:
                reset_tutorial() # 0.5초가 지나면 초기화하고 타이머를 None으로 되돌림

        # 3. 돌이 판 밖으로 나갔는지 실시간으로 체크 (타이머가 돌고 있지 않을 때만)
        if reset_timer is None and not tutorial_dragging and not show_failure_screen:
            is_opponent_out = not tutorial_area.inflate(STONE_RADIUS, STONE_RADIUS).collidepoint(opponent_body.position)
            is_player_out = not tutorial_area.inflate(STONE_RADIUS, STONE_RADIUS).collidepoint(player_body.position)
            
            if is_opponent_out or is_player_out:
                # 4. 돌이 나가면 현재 시간을 타이머에 기록하여 타이머 시작
                reset_timer = pygame.time.get_ticks()

        tutorial_space.step(1/60.0)
        clock.tick(60)

def show_model_details_screen(screen, clock):
    """모델 상세 정보를 보여주는 별도의 화면 (설명 왼쪽 정렬)"""
    # 모드 선택 화면의 타이틀 폰트 크기(24)와 동일하게 설정
    font_title = get_font(24)
    font_large = get_font(33)
    font_medium = get_font(27)
    font_small = get_font(18)
    font_desc = get_font(16)
    font_arrow = get_font(48)
    running = True

    models_info = [
        {"name": "AlggaGo1.0", "desc": [
            "180만번 학습한 AlggaGo 초기 모델입니다.",
            "",
            "특징:",
            " - 상대 돌을 무난하게 쳐냅니다.",
            " - 기회가 될 때 1타 2피 이상을 치도록 학습되었습니다.",
            "",
            "학습 방식:",
            " 본 모델은 강화학습으로 구축 되었습니다.",
            " (1) 상대를 정확하게 치는 rule-based로 가중치를 초기화합니다.",
            " (2) 엔트로피를 점차 향상시켜가며 GAN 방식에 영감을 받아 ",
            "경쟁 방식으로 학습을 진행하였습니다."

        ]},
        {"name": "Model C", "desc": [
            "AlggaGo2.0을 학습시키기 위해 제작된 규칙 기반 모델입니다.",
            "",
            "특징:",
            " - 일반적인 상황에서 상대를 정교하게 칩니다.",
            " - 특정한 상황에서 새로운 전략을 사용합니다.",
            " - 경로 계산 로직을 가지고 있어 아군을 보호합니다."
        ]},
        {"name": "AlggaGo2.0", "desc": [
            "1300만번 학습한 AlggaGo2.0입니다.",
            "인간을 이기기 위해 설계되었습니다.",
            "",
            "특징:",
            " - 1타 2피 이상을 더 잘 쳐냅니다.",
            " - 두 개의 전략을 자유자재로 똑똑하게 활용합니다.",
            "",
            "학습 방식:",
            " 본 모델은 강화학습으로 구축 되었습니다.",
            " (1) 두 개의 전략을 정확하게 구사하도록 가중치를 초기화합니다.",
            " (2) Model C를 상대로 이길 때까지 학습을 진행합니다.",
            " (3) 이후 AlggaGo1.0 학습 방식과 거의 동일하게 학습합니다."
        ]}
    ]
    current_model_index = 0

    while running:
        screen.fill((50, 50, 50))
        mouse_pos = pygame.mouse.get_pos()

        left_arrow_rect = None
        right_arrow_rect = None

        title_surf = font_title.render("모델 상세", True, (255, 255, 255))
        title_rect = title_surf.get_rect(topleft=(20, 20))
        screen.blit(title_surf, title_rect)

        current_model = models_info[current_model_index]

        model_surf = font_large.render(current_model["name"], True, (255, 255, 255))
        model_rect = model_surf.get_rect(center=(WIDTH // 2, 170))
        screen.blit(model_surf, model_rect)
        
        # --- 모델 설명 (왼쪽 정렬로 수정) ---
        desc_lines = current_model["desc"]
        
        # 1. 가장 긴 줄의 너비 찾기
        max_width = 0
        for line in desc_lines:
            line_width, _ = font_desc.size(line)
            if line_width > max_width:
                max_width = line_width

        # 2. 텍스트 블록의 시작 x 좌표 계산 (화면 중앙 기준)
        start_x = (WIDTH // 2) - (max_width // 2)
        
        # 3. 렌더링 시작 y 좌표 및 줄 간격 설정
        line_height = font_desc.get_height() + 5
        start_y = model_rect.bottom + 60

        # 4. 계산된 시작점에 맞춰 각 줄을 왼쪽 정렬로 그리기
        for i, line in enumerate(desc_lines):
            desc_surf = font_desc.render(line, True, (220, 220, 220))
            desc_rect = desc_surf.get_rect(topleft=(start_x, start_y + (i * line_height)))
            screen.blit(desc_surf, desc_rect)
        # --- 수정 끝 ---

        if current_model_index > 0:
            left_arrow_surf = font_arrow.render("<", True, (255, 255, 255))
            left_arrow_rect = left_arrow_surf.get_rect(center=(60, HEIGHT // 2))
            screen.blit(left_arrow_surf, left_arrow_rect)

        if current_model_index < len(models_info) - 1:
            right_arrow_surf = font_arrow.render(">", True, (255, 255, 255))
            right_arrow_rect = right_arrow_surf.get_rect(center=(WIDTH - 60, HEIGHT // 2))
            screen.blit(right_arrow_surf, right_arrow_rect)
        
        hint_surf = font_small.render("클릭하여 돌아가기", True, (200, 200, 200))
        hint_rect = hint_surf.get_rect(center=(WIDTH // 2, HEIGHT - 100))
        screen.blit(hint_surf, hint_rect)

        is_over_left = left_arrow_rect and left_arrow_rect.collidepoint(mouse_pos)
        is_over_right = right_arrow_rect and right_arrow_rect.collidepoint(mouse_pos)
        if is_over_left or is_over_right:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if left_arrow_rect and left_arrow_rect.collidepoint(event.pos):
                    current_model_index -= 1
                elif right_arrow_rect and right_arrow_rect.collidepoint(event.pos):
                    current_model_index += 1
                else:
                    running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and current_model_index > 0:
                     current_model_index -= 1
                elif event.key == pygame.K_RIGHT and current_model_index < len(models_info) - 1:
                     current_model_index += 1
                else:
                    running = False

        clock.tick(60)
    
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

def select_game_mode(screen, clock, nickname):
    """게임 모드를 선택하는 함수 (커서 오류 수정)"""
    pygame.init()
    font_small_large = get_font(26)
    font_medium = get_font(24)
    font_small = get_font(18)

    controls_button_rect = pygame.Rect(WIDTH - 130, 20, 110, 40)
    model_details_button_rect = pygame.Rect(WIDTH - 130, 70, 110, 40)
    
    selected_mode = None
    
    mode_buttons = []

    modes = [
        "AlggaGo1.0",
        "AlggaGo2.0",
        "",
        "1. 알까기 챔피언십",
        "2. 나는야 알까기의 이세돌",
        "3. 내 마음대로 배치하기",
        "4. Easy 모드"
    ]
    
    key_to_mode = {
        pygame.K_q: 5, pygame.K_w: 2, pygame.K_1: 1,
        pygame.K_2: 3, pygame.K_3: 4, pygame.K_4: 6
    }
    
    while selected_mode is None:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill((50, 50, 50))

        is_hovering_controls = controls_button_rect.collidepoint(mouse_pos)
        button_color = (100, 100, 100) if is_hovering_controls else (70, 70, 70)
        pygame.draw.rect(screen, (255, 255, 255), controls_button_rect, 2, border_radius=5) # 테두리
        button_text_surf = font_small.render("조작 안내", True, (255, 255, 255))
        button_text_rect = button_text_surf.get_rect(center=controls_button_rect.center)
        screen.blit(button_text_surf, button_text_rect)

        is_hovering_details = model_details_button_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (255, 255, 255), model_details_button_rect, 2, border_radius=5)
        details_text_surf = font_small.render("모델 상세", True, (255, 255, 255))
        details_text_rect = details_text_surf.get_rect(center=model_details_button_rect.center)
        screen.blit(details_text_surf, details_text_rect)
        
        title_surface = font_medium.render("AlggaGo", True, (255, 255, 255))
        title_rect = title_surface.get_rect(topleft=(20, 20))
        screen.blit(title_surface, title_rect)
        
        mode_buttons.clear()
        is_over_any_button = False  # 버튼 위에 마우스가 있는지 확인하는 플래그

        y_pos_start = 220
        for i, mode_text in enumerate(modes):
            y_pos = y_pos_start + i * 50
            if not mode_text:
                continue

            # --- 여기서 1.0, 2.0만 다른 폰트 적용 ---
            if "1.0" in mode_text or "2.0" in mode_text:
                use_font = font_small_large
            else:
                use_font = font_medium

            temp_surface = use_font.render(mode_text, True, (0,0,0))
            mode_rect = temp_surface.get_rect(topleft=(290, y_pos))

            # 마우스 hover 체크
            if mode_rect.collidepoint(mouse_pos):
                color = (255, 255, 255)
                is_over_any_button = True 
            else:
                color = (255, 255, 255)

            final_surface = use_font.render(mode_text, True, color)
            screen.blit(final_surface, mode_rect)
            
            mode_num = None
            if "1.0" in mode_text: mode_num = 5
            elif "2.0" in mode_text: mode_num = 2
            elif "1." in mode_text: mode_num = 1
            elif "2." in mode_text: mode_num = 3
            elif "3." in mode_text: mode_num = 4
            elif "4." in mode_text: mode_num = 6
            
            if mode_num is not None:
                mode_buttons.append((mode_rect, mode_num))

        if controls_button_rect.collidepoint(mouse_pos) or \
            model_details_button_rect.collidepoint(mouse_pos):
                is_over_any_button = True

        # 플래그 상태에 따라 마우스 커서 변경
        if is_over_any_button:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        hint_surface = font_small.render("게임 모드를 선택하세요", True, (200, 200, 200))
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT - 80))
        screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                return None
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # '조작 안내' 버튼을 눌렀는지 먼저 확인
                if controls_button_rect.collidepoint(event.pos):
                    show_controls_screen(screen, clock) # 안내 화면 호출
                elif model_details_button_rect.collidepoint(event.pos):
                    show_model_details_screen(screen, clock)
                else:
                    # 기존의 모드 선택 버튼 확인
                    for rect, mode_num in mode_buttons:
                        if rect.collidepoint(event.pos):
                            selected_mode = mode_num
                            break
            
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_mode:
                    selected_mode = key_to_mode[event.key]
                elif event.key == pygame.K_ESCAPE:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    return None
        
        clock.tick(60)
    
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
    return selected_mode


def save_game_record(nickname, win_streak, game_result, human_score, robot_score, game_mode=1):
    """게임 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "game_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'win_streak', 'game_result', 'human_score', 'robot_score', 'game_mode']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'win_streak': win_streak,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score,
            'game_mode': game_mode
        })


def save_vs_record(nickname, game_result, human_score, robot_score):
    """2번 모드용 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "vs_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })


def get_vs_stats():
    """2번 모드용 전적 통계를 가져오는 함수"""
    csv_filename = "vs_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins

def save_alggago2_record(nickname, game_result, human_score, robot_score):
    """AlggaGo 2.0 모드용 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "alggago2_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })

def get_alggago2_stats():
    """AlggaGo 2.0 모드용 전적 통계를 가져오는 함수"""
    csv_filename = "alggago2_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins

def get_leesedol_hall_of_fame():
    """이세돌 모드에서 HUMAN_WIN을 기록한 닉네임들을 집합으로 수집해 정렬된 리스트로 반환"""
    csv_filename = "leesedol_records.csv"
    hall = set()
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('game_result') == 'HUMAN_WIN':
                    hall.add(row.get('nickname', ''))
    return sorted(hall)

def save_leesedol_record(nickname, game_result, human_score, robot_score):
    """3번 모드용 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "leesedol_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })


def get_leesedol_win_order(nickname):
    """3번 모드에서 특정 닉네임의 승리 순서를 가져오는 함수"""
    csv_filename = "leesedol_records.csv"
    if not os.path.exists(csv_filename):
        return 0
    
    win_count = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['nickname'] == nickname and row['game_result'] == 'HUMAN_WIN':
                win_count += 1
    
    return win_count


def get_leesedol_stats():
    """3번 모드용 전적 통계를 가져오는 함수"""
    csv_filename = "leesedol_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins


def get_leesedol_attempt_count(nickname):
    """3번 모드에서 전체 총 도전 횟수를 가져오는 함수 (닉네임 상관없이)"""
    csv_filename = "leesedol_records.csv"
    if not os.path.exists(csv_filename):
        return 0
    
    attempt_count = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            attempt_count += 1  # 모든 기록을 카운트 (닉네임 상관없이)
    
    return attempt_count

def save_custom_placement_record(nickname, game_result, human_score, robot_score):
    """4번 모드용 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "custom_placement_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })


def get_custom_placement_stats():
    """4번 모드용 전적 통계를 가져오는 함수"""
    csv_filename = "custom_placement_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins


def save_basic_ai_record(nickname, game_result, human_score, robot_score):
    """5번 모드용 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "basic_ai_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })


def get_basic_ai_stats():
    """5번 모드용 전적 통계를 가져오는 함수"""
    csv_filename = "basic_ai_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins

def save_beginner_mode_record(nickname, game_result, human_score, robot_score):
    """6번 모드(초보자용) 전적 기록을 CSV 파일에 저장하는 함수"""
    csv_filename = "beginner_mode_records.csv"
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'nickname', 'game_result', 'human_score', 'robot_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 없으면 헤더 작성
        if not file_exists:
            writer.writeheader()
        
        # 게임 기록 작성
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nickname': nickname,
            'game_result': game_result,
            'human_score': human_score,
            'robot_score': robot_score
        })


def get_beginner_mode_stats():
    """6번 모드(초보자용) 전적 통계를 가져오는 함수"""
    csv_filename = "beginner_mode_records.csv"
    if not os.path.exists(csv_filename):
        return 0, 0  # human_wins, ai_wins
    
    human_wins = 0
    ai_wins = 0
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['game_result'] == 'HUMAN_WIN':
                human_wins += 1
            elif row['game_result'] == 'AI_WIN':
                ai_wins += 1
    
    return human_wins, ai_wins


def show_win_streak(screen, clock, nickname, win_streak):
    """승리 시 연승 정보를 보여주는 함수 (자동 진행)"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    start_time = time.time()
    display_duration = 3.0  # 3초간 표시 후 자동 진행
    
    while True:
        screen.fill((50, 50, 50))
        
        # 승리 메시지
        win_text = f"{nickname}님 승리!"
        win_surface = font_large.render(win_text, True, (255, 255, 255))
        win_rect = win_surface.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        screen.blit(win_surface, win_rect)
        
        # 1위 기록 가져오기
        top_players = get_top_players()
        if top_players:
            top_name, top_streak = top_players[0]
            top_msg = f"현재 1위: {top_name}님 ({top_streak}연승)"
        else:
            top_msg = "현재 1위: 기록 없음"

        # 연승 달성 메시지
        streak_msg = f"연승 달성 ({win_streak}연승)"
        streak_surface = font_medium.render(streak_msg, True, (255, 255, 255))
        streak_rect = streak_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(streak_surface, streak_rect)

        # 1위 정보 메시지
        top_surface = font_medium.render(top_msg, True, (255, 255, 255))
        top_rect = top_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        screen.blit(top_surface, top_rect)
        
        # 남은 시간 표시
        remaining_time = max(0, display_duration - (time.time() - start_time))
        if remaining_time > 0:
            time_text = f"다음 경기까지 {remaining_time:.1f}초"
            time_surface = font_small.render(time_text, True, (200, 200, 200))
            time_rect = time_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(time_surface, time_rect)
        
        pygame.display.flip()
        
        # 3초 후 자동 진행
        if time.time() - start_time >= display_duration:
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                # 키 또는 마우스 클릭 시 즉시 진행
                return True
        
        clock.tick(60)
    
    return False


def show_game_result(screen, clock, nickname, human_score, robot_score, winner, win_streak):
    """게임 결과를 보여주는 함수"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    result_shown = False
    start_time = time.time()
    
    while not result_shown:
        screen.fill((50, 50, 50))
        
        # 결과 메시지
        if winner == "human":
            result_text = f"{nickname}님 승리!"
            result_color = (255, 255, 255)
        else:
            result_text = "AI 승리!"
            result_color = (255, 255, 255)
        
        result_surface = font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        screen.blit(result_surface, result_rect)
        
        # 연승 정보 + 순위
        streak_text = f"최종 연승 기록: {win_streak}연승"
        streak_surface = font_medium.render(streak_text, True, (255, 255, 255))
        streak_rect = streak_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(streak_surface, streak_rect)

        # 순위 계산
        top_players = get_top_players()

        # --- 전체 플레이어 기록 불러오기 (랭킹 산정용) ---
        csv_filename = "game_records.csv"
        all_records = []
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                player_records = {}
                for row in reader:
                    nname = row['nickname']
                    streak = int(row['win_streak'])
                    if nname not in player_records:
                        player_records[nname] = streak
                    else:
                        player_records[nname] = max(player_records[nname], streak)
                all_records = sorted(player_records.items(), key=lambda x: x[1], reverse=True)

        # 현재 플레이어 순위 계산
        rank = 0
        for idx, (nname, streak) in enumerate(all_records):
            if nname == nickname:
                rank = idx + 1
                break

        if rank == 0:
            rank_text = "당신의 순위: 기록 없음"
        else:
            rank_text = f"당신의 순위: {rank}위"


        rank_surface = font_medium.render(rank_text, True, (255, 255, 255))
        rank_rect = rank_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
        screen.blit(rank_surface, rank_rect)
        
        # 안내 메시지
        if time.time() - start_time > 3:  # 3초 후 안내 표시
            hint_surface = font_small.render("아무 키나 눌러 순위 보기", True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                if time.time() - start_time > 2:  # 2초 후에만 키 입력 받음
                    return "MODE_SELECT"
        
        clock.tick(60)
    
    return False


def show_vs_result(screen, clock, nickname, human_score, robot_score, winner):
    """2번 모드용 게임 결과를 보여주는 함수 (전적 현황 포함)"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    result_shown = False
    start_time = time.time()
    
    while not result_shown:
        screen.fill((50, 50, 50))
        
        # 결과 메시지
        if winner == "human":
            result_text = f"{nickname}님 승리!"
            result_color = (0, 255, 0)
        else:
            result_text = "AI 승리!"
            result_color = (255, 0, 0)
        
        result_surface = font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        screen.blit(result_surface, result_rect)
        
        # 스코어
        score_text = f"최종 스코어 - {nickname}: {human_score}  AI: {robot_score}"
        score_surface = font_medium.render(score_text, True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(score_surface, score_rect)
        
        # 전적 현황
        human_wins, ai_wins = get_vs_stats()
        total_games = human_wins + ai_wins
        
        if total_games > 0:
            stats_text = f"전적 현황 - 지구인: {human_wins}승  AI: {ai_wins}승  (총 {total_games}경기)"
            stats_surface = font_medium.render(stats_text, True, (255, 255, 0))
            stats_rect = stats_surface.get_rect(center=(WIDTH // 2, HEIGHT * 2 // 3))
            screen.blit(stats_surface, stats_rect)
        
        # 안내 메시지
        if time.time() - start_time > 3:  # 3초 후 안내 표시
            hint_surface = font_small.render("아무 키나 눌러 다시 시작", True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if time.time() - start_time > 3:  # 3초 후에만 키 입력 받음
                    return True
        
        clock.tick(60)
    
    return False


def show_leesedol_result(screen, clock, nickname, human_score, robot_score, winner):
    """3번 모드용 게임 결과를 보여주는 함수 (전적 현황 포함)"""


    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    result_shown = False
    start_time = time.time()

    hall = get_leesedol_hall_of_fame()
    
    while not result_shown:
        screen.fill((50, 50, 50))
        
        # 결과 메시지
        if winner == "human":
            # 이세돌 승리 순서 계산
            if nickname in hall:
                idx = hall.index(nickname)
                win_order = len(hall)
            else:
                win_order = 0
            if win_order > 0:
                # 승리 메시지 (번호 강조)
                order_text = f"축하합니다 {win_order}번째 이세돌!"
                
                # 번호 부분만 강조하기 위해 텍스트를 분할
                before_number = "축하합니다 "
                number_part = f"{win_order}"
                after_number = "번째 이세돌!"
                
                # 전체 텍스트의 위치 계산
                full_text = before_number + number_part + after_number
                full_surface = font_large.render(full_text, True, (255, 255, 255))
                full_rect = full_surface.get_rect(center=(WIDTH // 2, HEIGHT // 3))
                
                # 각 부분의 위치 계산
                before_surface = font_large.render(before_number, True, (255, 255, 255))
                number_surface = font_large.render(number_part, True, (255, 255, 0))  # 노란색으로 강조
                after_surface = font_large.render(after_number, True, (255, 255, 255))
                
                # 각 부분의 너비 계산
                before_width = before_surface.get_width()
                number_width = number_surface.get_width()
                after_width = after_surface.get_width()
                
                # 시작 위치 계산
                start_x = full_rect.x
                before_x = start_x
                number_x = start_x + before_width
                after_x = start_x + before_width + number_width
                
                # 각 부분을 개별적으로 렌더링
                screen.blit(before_surface, (before_x, full_rect.y))
                screen.blit(number_surface, (number_x, full_rect.y))
                screen.blit(after_surface, (after_x, full_rect.y))
            
            else:
                # 일반 승리 메시지
                result_text = f"{nickname}님 승리!"
                result_surface = font_large.render(result_text, True, (255, 255, 255))
                result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4))
                screen.blit(result_surface, result_rect)
                
        else:
            result_text = "이세돌 되기 실패!"
            result_surface = font_large.render(result_text, True, (255, 255, 255))
            result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4))
            screen.blit(result_surface, result_rect)
        
        title_hall = "[ 명예의 전당 ]"
        surf_title = font_medium.render(title_hall, True, (255, 255, 255))
        rect_title = surf_title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 70))
        screen.blit(surf_title, rect_title)
        hall = get_leesedol_hall_of_fame()

        if hall:
            chunk_size = 5
            for idx, start in enumerate(range(0, len(hall), chunk_size)):
                names_chunk = hall[start:start + chunk_size]
                line_text = ", ".join(names_chunk)
                surf = font_medium.render(line_text, True, (255, 255, 255))
                y = HEIGHT // 2 + idx * 30
                rect = surf.get_rect(center=(WIDTH // 2, y - 20))
                screen.blit(surf, rect)
        else:
            no_hall = "아직 아무도 없어요"
            surf = font_medium.render(no_hall, True, (255, 255, 255))
            rect = surf.get_rect(center=(WIDTH // 2, y -20))
            screen.blit(surf, rect)
    
        
        # 도전 횟수 표시
        attempt_count = get_leesedol_attempt_count(nickname)
        attempt_text = f"이세돌이 되기 위해 도전한 총 횟수: {attempt_count}회"
        attempt_surface = font_medium.render(attempt_text, True, (255, 255, 255))
        attempt_rect = attempt_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4 - 40))
        screen.blit(attempt_surface, attempt_rect)
        
        # 안내 메시지
        if time.time() - start_time > 3:  # 3초 후 안내 표시
            hint_surface = font_small.render("클릭하여 돌아가기", True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4 + 20))
            screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                if time.time() - start_time > 2:  # 3초 후에만 키 입력 받음
                    return True
        
        clock.tick(60)
    
    return False


def show_mode3_intro(screen, clock, nickname):
    """3번 모드 시작 시 안내 메시지를 보여주는 함수"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    start_time = time.time()
    display_duration = 5.0  # 5초간 표시
    
    while True:
        screen.fill((50, 50, 50))
        
        # 제목
        title_surface = font_large.render("나는야 알까기의 이세돌", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4 + 20))
        screen.blit(title_surface, title_rect)
        
        # 안내 메시지
        instruction1 = f"{nickname}님은 백돌을 조작합니다"
        instruction1_surface = font_medium.render(instruction1, True, (255, 255, 255))
        instruction1_rect = instruction1_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
        screen.blit(instruction1_surface, instruction1_rect)
        
        instruction2 = "AI가 흑돌을 먼저 시작합니다"
        instruction2_surface = font_medium.render(instruction2, True, (255, 255, 255))
        instruction2_rect = instruction2_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 0))
        screen.blit(instruction2_surface, instruction2_rect)
        
        instruction3 = "이세돌의 신의 한 수에 도전하세요!"
        instruction3_surface = font_medium.render(instruction3, True, (255, 255, 0))
        instruction3_rect = instruction3_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
        screen.blit(instruction3_surface, instruction3_rect)
        
        # 남은 시간 표시
        remaining_time = max(0, display_duration - (time.time() - start_time))
        if remaining_time > 0:
            time_text = f"게임 시작까지 {remaining_time:.1f}초"
            time_surface = font_small.render(time_text, True, (200, 200, 200))
            time_rect = time_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(time_surface, time_rect)
        
        pygame.display.flip()
        
        # 5초 후 자동 진행
        if time.time() - start_time >= display_duration:
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                # 키 입력 시 즉시 진행
                return True
        
        clock.tick(60)
    
    return False


def show_custom_placement_result(screen, clock, nickname, human_score, robot_score, winner):
    """4번 모드용 게임 결과를 보여주는 함수 (전적 현황 포함)"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    result_shown = False
    start_time = time.time()
    
    while not result_shown:
        screen.fill((50, 50, 50))
        
        # 결과 메시지
        if winner == "human":
            result_text = f"{nickname}님 승리!"
            result_color = (255, 255, 255)
        else:
            result_text = "AI 승리!"
            result_color = (255, 255, 255)
        
        result_surface = font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4 +80))
        screen.blit(result_surface, result_rect)
        
        
        # 전적 현황
        human_wins, ai_wins = get_custom_placement_stats()
        total_games = human_wins + ai_wins
        
        if total_games > 0:
            stats_text = f"전적 현황 - 연구자: {human_wins}승  AI: {ai_wins}승  (총 {total_games}경기)"
            stats_surface = font_medium.render(stats_text, True, (255, 255, 255))
            stats_rect = stats_surface.get_rect(center=(WIDTH // 2, HEIGHT * 2 // 3 - 85))
            screen.blit(stats_surface, stats_rect)
        
        # 안내 메시지
        if time.time() - start_time > 3:  # 3초 후 안내 표시
            hint_surface = font_small.render("클릭하여 돌아가기", True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                if time.time() - start_time > 2:  # 3초 후에만 키 입력 받음
                    return True
        
        clock.tick(60)
    
    return False


def show_basic_ai_result(screen, clock, nickname, human_score, robot_score, winner):
    """5번 모드용 게임 결과를 보여주는 함수 (전적 현황 포함)"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    result_shown = False
    start_time = time.time()
    
    while not result_shown:
        screen.fill((50, 50, 50))
        
        # 결과 메시지
        if winner == "human":
            result_text = f"{nickname}님 승리!"
            result_color = (255, 255, 255)
        else:
            result_text = "AI 승리!"
            result_color = (255, 255, 255)
        
        result_surface = font_large.render(result_text, True, result_color)
        result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4 + 50))
        screen.blit(result_surface, result_rect)
        
        
        # 전적 현황
        human_wins, ai_wins = get_basic_ai_stats()
        total_games = human_wins + ai_wins
        
        if total_games > 0:
            stats_text = f"전적 현황 - 지구인: {human_wins}승  AI: {ai_wins}승  (총 {total_games}경기)"
            stats_surface = font_medium.render(stats_text, True, (255, 255, 0))
            stats_rect = stats_surface.get_rect(center=(WIDTH // 2, HEIGHT * 2 // 3))
            screen.blit(stats_surface, stats_rect)
        
        # 안내 메시지
        if time.time() - start_time > 3:  # 3초 후 안내 표시
            hint_surface = font_small.render("아무 키나 눌러 다시 시작", True, (200, 200, 200))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT * 3 // 4))
            screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if time.time() - start_time > 3:  # 3초 후에만 키 입력 받음
                    return True
        
        clock.tick(60)
    
    return False


def setup_custom_black_stones(screen, clock, nickname):
    """4번 모드용 흑돌 배치 설정 화면"""
    font_large = get_font(36)
    font_medium = get_font(24)
    font_small = get_font(18)
    
    # 흑돌 위치 리스트 (None이면 랜덤 배치 사용)
    black_stone_positions = None
    
    # 하단 40% 영역 계산
    bottom_area_start = HEIGHT * 0.6  # 하단 40% 시작점
    min_distance = STONE_RADIUS * 2.5  # 돌 간 최소 거리
    
    # 기본 4개 흑돌 배치 (하단 40% 영역 내에 랜덤하게 배치)
    def generate_random_positions():
        positions = []
        max_attempts = 1000
        
        for _ in range(4):
            attempts = 0
            while attempts < max_attempts:
                x = np.random.uniform(MARGIN + STONE_RADIUS, WIDTH - MARGIN - STONE_RADIUS)
                y = np.random.uniform(bottom_area_start, HEIGHT - MARGIN - STONE_RADIUS)
                
                # 기존 돌들과의 거리 확인
                too_close = False
                for px, py in positions:
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append((x, y))
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                # 최대 시도 횟수 초과 시 강제로 배치
                x = MARGIN + STONE_RADIUS + len(positions) * 100
                y = bottom_area_start + 50
                positions.append((x, y))
        
        return positions
    
    # 마우스 위치 추적
    mouse_pos = None
    
    while True:
        screen.fill((50, 50, 50))
        pygame.draw.rect(screen, (210, 180, 140), pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN))
        
        # 바둑판 그리기
        cell = (WIDTH - 2 * MARGIN) / 18
        for i in range(19):
            x = MARGIN + i * cell
            lw = 3 if i in (0, 18) else 1
            pygame.draw.line(screen, (0, 0, 0), (int(x), MARGIN), (int(x), HEIGHT - MARGIN), lw)
        for j in range(19):
            y = MARGIN + j * cell
            lw = 3 if j in (0, 18) else 1
            pygame.draw.line(screen, (0, 0, 0), (MARGIN, int(y)), (WIDTH - MARGIN, int(y)), lw)
        for si in (3, 9, 15):
            for sj in (3, 9, 15):
                sx = MARGIN + si * cell
                sy = MARGIN + sj * cell
                pygame.draw.circle(screen, (0, 0, 0), (int(sx), int(sy)), 5)
        
        font_bold_large = pygame.font.SysFont("Malgun Gothic", 36, bold=True)
        font_bold_medium = pygame.font.SysFont("Malgun Gothic", 24, bold=True)

        # 제목
        title_surface = font_bold_large.render("흑돌 배치 설정", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(WIDTH // 2, 85))
        screen.blit(title_surface, title_rect)
        
        
        # 안내 메시지
        instruction_surface = font_bold_medium.render(
            "하단 40% 영역에서 클릭하여 흑돌을 배치하세요 (최대 4개)",
            True, (255, 255, 255)
        )
        instruction_rect = instruction_surface.get_rect(center=(WIDTH // 2, 130))
        screen.blit(instruction_surface, instruction_rect)
        
        # 하단 40% 영역 표시
        pygame.draw.rect(screen, (255, 255, 0, 50), 
                        pygame.Rect(MARGIN, int(bottom_area_start), 
                                  WIDTH - 2 * MARGIN, HEIGHT - MARGIN - int(bottom_area_start)), 2)
        
        # 흑돌 그리기 (직접 배치 모드일 때만)
        if black_stone_positions is not None:
            for i, (x, y) in enumerate(black_stone_positions):
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), STONE_RADIUS)
                
                # 돌 번호 표시
                number_surface = font_small.render(str(i+1), True, (255, 255, 255))
                number_rect = number_surface.get_rect(center=(int(x), int(y)))
                screen.blit(number_surface, number_rect)
        
        # 마우스 위치에 실루엣 돌 표시 (직접 배치 모드이고 하단 40% 영역에 있을 때)
        if black_stone_positions is not None and mouse_pos is not None:
            if (mouse_pos.y >= bottom_area_start and 
                mouse_pos.x >= MARGIN + STONE_RADIUS and 
                mouse_pos.x <= WIDTH - MARGIN - STONE_RADIUS and
                mouse_pos.y <= HEIGHT - MARGIN - STONE_RADIUS):
                
                # 실루엣 돌 그리기 (반투명)
                silhouette_surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
                pygame.draw.circle(silhouette_surface, (0, 0, 0, 128), (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
                screen.blit(silhouette_surface, (int(mouse_pos.x) - STONE_RADIUS, int(mouse_pos.y) - STONE_RADIUS))
        
        # 안내 메시지
        hint_surface = font_small.render("Enter: 게임 시작  D: 배치하기  ESC: 취소", True, (200, 200, 200))
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT - 30))
        screen.blit(hint_surface, hint_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = Vec2d(*event.pos)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return black_stone_positions
                elif event.key == pygame.K_r:
                    # 랜덤 배치 모드로 전환
                    black_stone_positions = None
                elif event.key == pygame.K_d:
                    # 직접 배치 모드로 전환 (빈 리스트로 시작)
                    black_stone_positions = []
                elif event.key == pygame.K_ESCAPE:
                    return None
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if black_stone_positions is not None and mouse_pos is not None:
                    # 하단 40% 영역 내에 있는지 확인
                    if (mouse_pos.y >= bottom_area_start and 
                        mouse_pos.x >= MARGIN + STONE_RADIUS and 
                        mouse_pos.x <= WIDTH - MARGIN - STONE_RADIUS and
                        mouse_pos.y <= HEIGHT - MARGIN - STONE_RADIUS):
                        
                        # 최대 4개까지만 배치 가능
                        if len(black_stone_positions) < 4:
                            # 다른 돌들과의 충돌 확인
                            collision_detected = False
                            for x, y in black_stone_positions:
                                distance = math.sqrt((mouse_pos.x - x)**2 + (mouse_pos.y - y)**2)
                                if distance < min_distance:
                                    collision_detected = True
                                    break
                            
                            if not collision_detected:
                                black_stone_positions.append((mouse_pos.x, mouse_pos.y))
        
        clock.tick(60)
    
    return None


def play_game(screen, clock, nickname, game_mode):
    """게임을 실행하는 함수"""
    # 4번 모드인 경우 흑돌 배치 설정
    custom_black_positions = None
    if game_mode == 4:
        custom_black_positions = setup_custom_black_stones(screen, clock, nickname)
        if custom_black_positions is None:
            return True  # 취소된 경우 메인 메뉴로 돌아가기
    
    # 기본 AI 에이전트 선택
    white_rl_agent = get_ai_agent(game_mode)
    
    pygame.mixer.init()
    collision_sound = pygame.mixer.Sound(rel_path("collision.mp3"))
    collision_sound.set_volume(1.0)
    
    # 게임 초기화
    human_score = 0
    robot_score = 0
    turn_text = ""
    last_ai_time = None

    # 경로 시뮬레이션 캐시 (성능 최적화용)
    path_cache = {}
    
        # 모드별 시작 턴 설정
    if game_mode == 3:
        turn = "waiting_b"  # 3번 모드: AI(흑돌) 먼저 시작, 사람은 백돌 조작
    else:
        turn = "black"  # 1, 2, 4, 5번 모드: 인간(흑돌) 먼저 시작
    
    win_streak = 0  # 연승 카운터

    # 물리 공간 설정
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.1
    
    # 충돌 효과음용 콜백 등록
    def on_collision(arbiter, space, data):
        collision_sound.play()
    space.on_collision(1, 1, begin=on_collision)
    
    # 경계 생성 (안쪽 네모 박스)
    static_body = space.static_body
    corners = [
        (MARGIN, MARGIN), (WIDTH - MARGIN, MARGIN),
        (WIDTH - MARGIN, HEIGHT - MARGIN), (MARGIN, HEIGHT - MARGIN)
    ]
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        seg = pymunk.Segment(static_body, a, b, 1)
        seg.sensor = True 
        space.add(seg)

    stones = []
    
    # 모드별 돌 배치
    if game_mode == 4:
        if custom_black_positions:
            # 사용자가 직접 배치한 경우
            black_count, white_count = reset_stones_custom(space, stones, custom_black_positions)
        else:
            # 랜덤 배치 (4번 모드에서만 랜덤 사용)
            black_count, white_count = reset_stones_random(space, stones)
    
    elif game_mode == 6:
        black_count, white_count = reset_stones_beginner(space, stones)
    else:
        # 모드 1, 2, 3, 5: 고정 배치
        black_count, white_count = reset_stones(space, stones)

    dragging = False
    drag_shape = None
    drag_start = Vec2d(0, 0)
    
    # 3번 모드 시작 시 안내 메시지
    if game_mode == 3:
        show_mode3_intro(screen, clock, nickname)
    
    if game_mode == 3:
        turn = "waiting_b"
        turn_text = "AI Turn (Black)"    # AI가 먼저 치는 모드
    else:
        turn = "black"
        turn_text = f"Your Turn (Black)"  # 인간(흑돌) 먼저 치는 모드
    
    # ────── 메인 게임 루프 ──────
    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                return False
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_ESCAPE:
                    return True  # 첫 화면으로 돌아가기
                if evt.key == pygame.K_BACKSPACE:
                    return None

            elif evt.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = Vec2d(*evt.pos)
                if turn == "black" or turn == "white": # 인간 턴
                    # 모드별 인간이 조작하는 돌 색상 결정
                    if game_mode == 3:
                        target_color = (255, 255, 255)  # 3번 모드: 인간이 백돌(흰돌) 조작
                    else:
                        target_color = (0, 0, 0)  # 1, 2, 4, 5번 모드: 인간이 흑돌 조작
                    
                    for shape in stones:
                        if shape.color[:3] == target_color:
                            if (shape.body.position - mouse_pos).length <= STONE_RADIUS:
                                dragging = True
                                drag_shape = shape
                                drag_start = shape.body.position
                                break

            elif evt.type == pygame.MOUSEBUTTONUP and dragging:
                drag_end = Vec2d(*evt.pos)
                raw_vec = drag_start - drag_end
                dist = raw_vec.length                  

                if dist > MAX_DRAG_LENGTH:
                    raw_vec = raw_vec.normalized() * MAX_DRAG_LENGTH
                    dist = MAX_DRAG_LENGTH

                if dist > 0:
                    impulse = raw_vec.normalized() * (dist * FORCE_MULTIPLIER + MIN_FORCE)
                    drag_shape.body.apply_impulse_at_world_point(impulse, drag_shape.body.position)

                dragging = False
                drag_shape = None
                if game_mode == 3:
                    turn = "waiting_b" # 백돌이 움직이고 흑돌 AI 턴이 되기 전
                else:
                    turn = "waiting_w" # 흑돌이 움직이고 백돌 AI 턴이 되기 전
                
        # 물리 업데이트
        space.step(1 / 60.0)

        # 백돌 AI 턴 (2번, 5번 모드)
        if turn == "waiting_w" and all_stones_stopped(stones) and game_mode != 3:
            turn_text = "AI Turn (White)"
            if last_ai_time is None:
                last_ai_time = time.time()
            elif time.time() - last_ai_time >= 1.0:
                
                # ===== AI 로직 분기 처리 시작 =====
                if isinstance(white_rl_agent, ModelCAgent):
                    # Model C 로직 실행
                    action_tuple = white_rl_agent.select_action(stones, "white")
                    if action_tuple:
                        idx, angle, force_normalized = action_tuple
                        white_stones = [s for s in stones if s.color[:3] == (255, 255, 255)]
                        if 0 <= idx < len(white_stones):
                            stone_to_shoot = white_stones[idx]
                            direction = Vec2d(1, 0).rotated(angle)
                            impulse = direction * scale_force(force_normalized)
                            stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)
                
                else: 
                    current_obs = create_obs_for_player(stones, "white", game_mode)
                    num_white_stones_alive = sum(1 for s in stones if s.color[:3] == (255, 255, 255))

                    # AI 에이전트로부터 (index_weight, angle_offset, force_offset) 튜플을 직접 받음
                    action_offsets = white_rl_agent.select_action(current_obs)

                    # 수정된 apply_action_to_stone 함수 호출 (모델 경로 정보 전달)
                    apply_action_to_stone(
                        action_offsets, 
                        stones, 
                        (255, 255, 255)
                    )
                
                turn = "waiting_b" # 백돌이 움직이고 흑돌(인간) 턴이 되기 전
                last_ai_time = None

        # 흑돌 AI 턴 (3번 모드)
        elif turn == "waiting_b" and all_stones_stopped(stones) and game_mode == 3:
            turn_text = "AI Turn (Black/흑돌)"
            if last_ai_time is None:
                last_ai_time = time.time()
            elif time.time() - last_ai_time >= 1.0:
                current_obs = create_obs_for_player(stones, "black", game_mode)
                num_black_stones_alive = sum(1 for s in stones if s.color[:3] == (0, 0, 0))

                # AI 에이전트로부터 오프셋 튜플을 직접 받음
                action_offsets = white_rl_agent.select_action(current_obs)

                # 수정된 apply_action_to_stone 함수 호출 (모델 경로 정보 전달)
                apply_action_to_stone(
                    action_offsets,
                    stones,
                    (0, 0, 0)
                )
                
                turn = "waiting_w" # 흑돌이 움직이고 백돌(인간) 턴이 되기 전
                last_ai_time = None

        # 흑돌 인간 턴 (1, 2, 4, 5번 모드)
        elif turn == "waiting_b" and all_stones_stopped(stones) and game_mode != 3:
            turn_text = f"Your Turn (Black)"
            turn = "black"
        
        # 백돌 인간 턴 (3번 모드)
        elif turn == "waiting_w" and all_stones_stopped(stones) and game_mode == 3:
            turn_text = f"Your Turn (White/백돌)"
            turn = "white"

        # 경기장 밖으로 나간 돌 처리
        current_black_count = sum(1 for s in stones if s.color[:3] == (0, 0, 0))
        current_white_count = sum(1 for s in stones if s.color[:3] == (255, 255, 255))
        
        for shape in stones[:]:
            x, y = shape.body.position
            if x < MARGIN or x > WIDTH - MARGIN or y < MARGIN or y > HEIGHT - MARGIN:
                space.remove(shape, shape.body)
                stones.remove(shape)

        # 게임 종료 및 초기화
        if current_black_count == 0 or current_white_count == 0:
            if current_black_count == 0:
                # 3번 모드에서는 플레이어가 백돌을 조작하므로, 흑돌이 0개면 플레이어 승리
                if game_mode == 3:
                    human_score += 1 # 플레이어(백돌) 승리
                    print(f"{nickname} 승리!")
                    winner = "human"
                else:
                    robot_score += 1 # 백돌 승리 (다른 모드)
                    print(f"AI 승리! {nickname} 패배!")
                    winner = "ai"
                
                if game_mode == 1:
                    # 1번 모드: 연승 모드
                    final_win_streak = win_streak  # 패배 전 연승 기록 저장
                    win_streak = 0
                    
                    # AI 승리 시에는 CSV에 저장하지 않음
                    
                    # 게임 결과 표시
                    if not show_game_result(screen, clock, nickname, human_score, robot_score, winner, final_win_streak):
                        return False
                    
                    # 모드 종료, 닉네임 설정 화면으로 돌아가기
                    return True
                    
                elif game_mode == 2:
                    # AlggaGo 2.0 모드에서는 모든 결과를 alggago2_records.csv에 저장
                    save_alggago2_record(nickname, "AI_WIN", human_score, robot_score)
                    
                    # 돌들 재배치
                    black_count, white_count = reset_stones(space, stones)

                    # 새 게임 시작 안내
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                    continue
                    
                elif game_mode == 3:
                    # 3번 모드: 이세돌 모드
                    # 이세돌 모드에서는 모든 결과를 저장
                    save_leesedol_record(nickname, "HUMAN_WIN", human_score, robot_score)
                    
                    # 게임 결과 표시 (전적 현황 포함)
                    if not show_leesedol_result(screen, clock, nickname, human_score, robot_score, winner):
                        return False
                    
                    # 모드 종료, 닉네임 설정 화면으로 돌아가기
                    return True
                    
                elif game_mode == 4:
                    # 4번 모드: 커스텀 배치 모드
                    # 커스텀 배치 모드에서는 모든 결과를 저장
                    save_custom_placement_record(nickname, "AI_WIN", human_score, robot_score)
                    
                    # 게임 결과 표시 (전적 현황 포함)
                    if not show_custom_placement_result(screen, clock, nickname, human_score, robot_score, winner):
                        return False
                    
                    # 모드 종료, 닉네임 설정 화면으로 돌아가기
                    return True
                    
                elif game_mode == 5:
                     # 5번 모드: 기본 AI 대전 모드
                    # 결과를 vs_records.csv에 저장
                    save_vs_record(nickname,
                               "AI_WIN" if current_black_count==0 else "HUMAN_WIN",
                               human_score,
                               robot_score)
                    # 별도 결과 화면 없이 즉시 새 게임 시작
                    black_count, white_count = reset_stones(space, stones)
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                
                    continue

                elif game_mode == 6:
                    # 6번 모드: 초보자용 모드
                    save_beginner_mode_record(nickname, "AI_WIN", human_score, robot_score)
                    black_count, white_count = reset_stones_beginner(space, stones)
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                    continue
                
            elif current_white_count == 0:
                # 3번 모드에서는 플레이어가 백돌을 조작하므로, 백돌이 0개면 AI 승리
                if game_mode == 3:
                    robot_score += 1 # AI(흑돌) 승리
                    print(f"AI 승리! {nickname} 패배!")
                    winner = "ai"
                else:
                    human_score += 1 # 흑돌 승리 (다른 모드)
                    print(f"{nickname} 승리!")
                    winner = "human"
                
                if game_mode == 1:
                    # 1번 모드: 연승 모드
                    # 승리했으므로 연승 증가
                    win_streak += 1
                    
                    # CSV에 기록 저장
                    save_game_record(nickname, win_streak, "HUMAN_WIN", human_score, robot_score, game_mode)
                    
                    # 승리 시 연승 정보 표시 (자동 진행)
                    if not show_win_streak(screen, clock, nickname, win_streak):
                        return False
                        
                elif game_mode == 2:
                    # AlggaGo 2.0 모드에서는 모든 결과를 alggago2_records.csv에 저장
                    save_alggago2_record(nickname, "HUMAN_WIN", human_score, robot_score)
                    
                    # 돌들 재배치
                    black_count, white_count = reset_stones(space, stones)

                    # 새 게임 시작 안내
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                    continue
                    
                    
                elif game_mode == 3:
                    # 3번 모드: 이세돌 모드
                    # 이세돌 모드에서는 모든 결과를 저장
                    save_leesedol_record(nickname, "AI_WIN", human_score, robot_score)
                    
                    # 게임 결과 표시 (전적 현황 포함)
                    if not show_leesedol_result(screen, clock, nickname, human_score, robot_score, winner):
                        return False
                    
                    # 모드 종료, 닉네임 설정 화면으로 돌아가기
                    return True
                    
                elif game_mode == 4:
                    # 4번 모드: 커스텀 배치 모드
                    # 커스텀 배치 모드에서는 모든 결과를 저장
                    save_custom_placement_record(nickname, "HUMAN_WIN", human_score, robot_score)
                    
                    # 게임 결과 표시 (전적 현황 포함)
                    if not show_custom_placement_result(screen, clock, nickname, human_score, robot_score, winner):
                        return False
                    
                    # 모드 종료, 닉네임 설정 화면으로 돌아가기
                    return True
                    
                elif game_mode == 5 and current_white_count == 0:
                    # 인간 승리 → 기록만 하고 새 게임
                    save_vs_record(nickname, "HUMAN_WIN", human_score, robot_score)
                    black_count, white_count = reset_stones(space, stones)
                    # 새 판 시작하자마자 플레이어 턴 안내
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                    continue

                elif game_mode == 6 and current_white_count == 0:
                    save_beginner_mode_record(nickname, "HUMAN_WIN", human_score, robot_score)
                    black_count, white_count = reset_stones_beginner(space, stones)
                    turn = "black"
                    turn_text = "Your Turn (Black)"
                    continue

            # 새로운 게임 시작 (1번 모드에서만)
            if game_mode == 1:
                black_count, white_count = reset_stones(space, stones)
                white_rl_agent = get_ai_agent(game_mode, win_streak)
                turn = "black"  # 다시 인간(흑돌) 턴부터 시작
                turn_text = f"Your Turn (Black)"
            
        # ────────── 렌더링 ──────────
        screen.fill((150, 150, 150))
        pygame.draw.rect(screen, (210, 180, 140), pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN))
        
        if game_mode == 1:
            pygame.draw.rect(screen, (144, 238, 144), pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN), width=5)      
        elif game_mode == 3:
            pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN), width=5)        
        elif game_mode == 4:
            pygame.draw.rect(screen, (173, 216, 230), pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN), width=5)
                    
    
        cell = (WIDTH - 2 * MARGIN) / 18
        for i in range(19):
            x = MARGIN + i * cell
            lw = 3 if i in (0, 18) else 1
            pygame.draw.line(screen, (0, 0, 0), (int(x), MARGIN), (int(x), HEIGHT - MARGIN), lw)
        for j in range(19):
            y = MARGIN + j * cell
            lw = 3 if j in (0, 18) else 1
            pygame.draw.line(screen, (0, 0, 0), (MARGIN, int(y)), (WIDTH - MARGIN, int(y)), lw)
        for si in (3, 9, 15):
            for sj in (3, 9, 15):
                sx = MARGIN + si * cell
                sy = MARGIN + sj * cell
                pygame.draw.circle(screen, (0, 0, 0), (int(sx), int(sy)), 5)

        for shape in stones:
            pos = shape.body.position
            color = shape.color[:3]
            pygame.draw.circle(screen, color, (int(pos.x), int(pos.y)), STONE_RADIUS)

        if dragging and drag_shape:
            mouse_pos = Vec2d(*pygame.mouse.get_pos())
            raw_vec = mouse_pos - drag_start
            length = raw_vec.length
            unit = raw_vec.normalized() if length > 0 else Vec2d(0, 0)
            display_len = min(length, MAX_DRAG_LENGTH)
            center = drag_shape.body.position
            end_pos = center + unit * display_len

            pygame.draw.line(
                screen, (255, 0, 0),
                (int(center.x), int(center.y)),
                (int(end_pos.x), int(end_pos.y)), 2
            )

            # 게임 모드 6일 때, 실제 이동 경로 시뮬레이션 및 표시
            if game_mode == 6:
                # 실제 이동 경로 시뮬레이션 및 표시
                if length > 0:
                    # 실제 발사와 동일한 벡터 계산 (drag_start - drag_end)
                    actual_raw_vec = drag_start - mouse_pos
                    actual_length = actual_raw_vec.length
                    
                    if actual_length > MAX_DRAG_LENGTH:
                        actual_raw_vec = actual_raw_vec.normalized() * MAX_DRAG_LENGTH
                        actual_length = MAX_DRAG_LENGTH
                    
                    # 캐시 키 생성 (실제 발사 벡터 기반)
                    cache_key = (int(actual_raw_vec.x), int(actual_raw_vec.y), int(actual_length))
                    
                    # 캐시된 경로가 없거나 마우스 위치가 변경된 경우에만 시뮬레이션 실행
                    if cache_key not in path_cache:
                        # 임시 물리 공간 생성
                        temp_space = pymunk.Space()
                        temp_space.gravity = (0, 0)
                        # 실제 공간과 동일한 감쇠 적용
                        temp_space.damping = space.damping
                        
                        # 실제 게임의 모든 돌들을 임시 공간에 복사 (더 정확한 시뮬레이션)
                        temp_stones = []
                        for stone in stones:
                            if stone != drag_shape:  # 현재 드래그 중인 돌은 제외
                                temp_moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
                                temp_body = pymunk.Body(STONE_MASS, temp_moment)
                                temp_body.position = stone.body.position
                                temp_body.velocity = stone.body.velocity
                                temp_shape = pymunk.Circle(temp_body, STONE_RADIUS)
                                # physics.create_stone과 동일하게 맞춤
                                temp_shape.elasticity = 1.0
                                temp_shape.friction = 0.9
                                temp_space.add(temp_body, temp_shape)
                                temp_stones.append((temp_body, temp_shape))
                        
                        # 임시 돌 생성 (현재 돌과 동일한 물리 속성)
                        temp_moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
                        temp_body = pymunk.Body(STONE_MASS, temp_moment)
                        temp_body.position = center
                        temp_shape = pymunk.Circle(temp_body, STONE_RADIUS)
                        # physics.create_stone과 동일하게 맞춤
                        temp_shape.elasticity = 1.0
                        temp_shape.friction = 0.9
                        temp_space.add(temp_body, temp_shape)
                        
                        # 임펄스 적용 (실제 발사와 동일한 힘 계산)
                        impulse = actual_raw_vec.normalized() * (actual_length * FORCE_MULTIPLIER + MIN_FORCE)
                        temp_body.apply_impulse_at_world_point(impulse, temp_body.position)
                        
                        # 경로 포인트들을 저장할 리스트
                        path_points = []
                        
                        # 시뮬레이션 실행 (약 2초간, 더 정확한 시뮬레이션)
                        for _ in range(120):  # 60fps * 2초
                            temp_space.step(1/60.0)
                            path_points.append(temp_body.position)
                            
                            # 속도가 충분히 느려지면 중단
                            if temp_body.velocity.length < 5:
                                break
                        
                        # 캐시에 저장
                        path_cache[cache_key] = path_points
                        
                        # 임시 공간 정리
                        temp_space.remove(temp_body, temp_shape)
                        for temp_body, temp_shape in temp_stones:
                            temp_space.remove(temp_body, temp_shape)
                    
                    # 캐시된 경로 사용
                    path_points = path_cache[cache_key]
                    
                    # 경로 그리기 (점선으로, 더 조밀하게)
                    if len(path_points) > 1:
                        for i in range(len(path_points) - 1):
                            if i % 1 == 0:  # 모든 포인트를 연결 (더 조밀하게)
                                start_pt = path_points[i]
                                end_pt = path_points[i + 1]
                                pygame.draw.line(
                                    screen, (0, 200, 255),  # 밝은 하늘색
                                    (int(start_pt.x), int(start_pt.y)),
                                    (int(end_pt.x), int(end_pt.y)), 2
                                )
                    
                    # 최종 위치에 작은 원 표시
                    if path_points:
                        final_pos = path_points[-1]
                        pygame.draw.circle(screen, (0, 200, 255), (int(final_pos.x), int(final_pos.y)), 4)


        font = get_font(24)
        hint_font = pygame.font.SysFont("DotumChe", 21)

        if game_mode == 1:
            if win_streak > 0:
                win_text = f"현재 {win_streak}연승 중"
            else:
                win_text = "현재 0연승 중"
            score_surface = font.render(win_text, True, (0, 0, 0))
            font_turn = get_font(24)
            turn_surface = font_turn.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 연승 텍스트 생성
            win_text = f"현재 {win_streak}연승 중" if win_streak > 0 else "현재 0연승 중"

            # 진한 초록 + 굵은 글꼴
            font_bold_green = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font_bold_green.render(win_text, True, (144, 238, 144))  # 초록색

            # 오른쪽 상단에 정렬
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)

            opponent_name = ""
            if isinstance(white_rl_agent, ModelCAgent):
                opponent_name = "Model C (Level 2)"
            # white_rl_agent.model_path를 확인하여 AI 이름 결정
            elif hasattr(white_rl_agent, 'model_path') and white_rl_agent.model_path:
                if "AlggaGo2.0" in white_rl_agent.model_path:
                    opponent_name = "AlggaGo2.0 (Level 3)"
                else: # 기본값
                    opponent_name = "AlggaGo1.0 (Level 1)"
            else: # Rule-based 또는 기타
                opponent_name = "AlggaGo1.0"

            # 중앙 상단에 "VS [상대이름]" 표시
            vs_text = f"{opponent_name}"
            vs_surface = font.render(vs_text, True, (0, 0, 0))
            vs_rect = vs_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(vs_surface, vs_rect)
    
        
        elif game_mode == 2:
            # 왼쪽 상단: 턴 정보
            turn_surface = font.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 오른쪽 상단: AlggaGo 2.0 누적 전적
            human_wins, ai_wins = get_alggago2_stats() 
            score_text = f"인간: {human_wins} VS AI: {ai_wins}"
            
            font_bold_purple = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font.render(score_text, True, (0, 0, 0))
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)

            # 중앙 상단: 현재 모드 표시
            mode_text = "AlggaGo2.0"
            mode_surface = font.render(mode_text, True, (0, 0, 0))
            mode_rect = mode_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(mode_surface, mode_rect)

            hint_surf = hint_font.render("Backspace를 눌러 돌아가기", True, (255, 255, 255))
            screen.blit(hint_surf, (MARGIN, HEIGHT - 40))
        
        elif game_mode == 3:
            font_bold_white = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            # 왼쪽 상단: 턴 정보
            turn_surface = font.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 오른쪽 상단: 누적 전적 (leesedol_records.csv 기반)
            leesedol_human_wins, leesedol_ai_wins = get_leesedol_stats()
            score_text = f"도전자: {leesedol_human_wins}   AI: {leesedol_ai_wins}"

            # 중앙 상단: 현재 모드 표시
            mode_text = "AlggaGo1.0"
            mode_surface = font.render(mode_text, True, (0, 0, 0))
            mode_rect = mode_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(mode_surface, mode_rect)

            font_bold_purple = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font_bold_white.render(score_text, True, (255, 255, 0))  # Medium Orchid
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)
        
        elif game_mode == 4:
            font_bold_blue = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            
            # 왼쪽 상단: 턴 정보
            turn_surface = font.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 오른쪽 상단: 누적 전적 (custom_placement_records.csv 기반)
            custom_human_wins, custom_ai_wins = get_custom_placement_stats()
            score_text = f"연구자: {custom_human_wins}   AI: {custom_ai_wins}"

            # 중앙 상단: 현재 모드 표시
            mode_text = "AlggaGo1.0"
            mode_surface = font.render(mode_text, True, (0, 0, 0))
            mode_rect = mode_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(mode_surface, mode_rect)

            font_bold_purple = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font_bold_blue.render(score_text, True, (173, 216, 230))  # Medium Orchid
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)
        
        elif game_mode == 5 :
            # 왼쪽 상단: 턴 정보
            turn_surface = font.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 누적 전적 (오른쪽 상단)
            vs_human_wins, vs_ai_wins = get_vs_stats()
            score_text = f"인간: {vs_human_wins} VS AI: {vs_ai_wins}"

            font_bold_purple = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font.render(score_text, True, (0, 0, 0))
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)

            # 중앙 상단: 현재 모드 표시
            mode_text = "AlggaGo1.0"
            mode_surface = font.render(mode_text, True, (0, 0, 0))
            mode_rect = mode_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(mode_surface, mode_rect)

            hint_surf = hint_font.render("Backspace를 눌러 돌아가기", True, (255, 255, 255))
            screen.blit(hint_surf, (MARGIN, HEIGHT - 40))
        
        elif game_mode == 6:
            # 왼쪽 상단: 턴 정보
            turn_surface = font.render(turn_text, True, (0, 0, 0))
            turn_rect = turn_surface.get_rect(topleft=(MARGIN, 10))
            screen.blit(turn_surface, turn_rect)

            # 오른쪽 상단: 6번 모드 누적 전적
            beginner_human_wins, beginner_ai_wins = get_beginner_mode_stats()
            score_text = f"나: {beginner_human_wins}승  AI: {beginner_ai_wins}승"
            
            font_bold_green = pygame.font.SysFont("Malgun Gothic", 24, bold=True)
            score_surface = font.render(score_text, True, (0, 0, 0))
            score_rect = score_surface.get_rect(topright=(WIDTH - MARGIN, 10))
            screen.blit(score_surface, score_rect)

            mode_text = "AlggaGo1.0"
            mode_surface = font.render(mode_text, True, (0, 0, 0))
            mode_rect = mode_surface.get_rect(midtop=(WIDTH // 2, 10))
            screen.blit(mode_surface, mode_rect)

            hint_surf = hint_font.render("Backspace를 눌러 모드 선택으로 돌아가기", True, (255, 255, 255))
            screen.blit(hint_surf, (MARGIN, HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)
        
    return True


def main():
    print("Let's Play AlggaGo!")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AlggaGo")
    clock = pygame.time.Clock()

    while True:
        # 1. 닉네임 입력
        nickname = get_nickname_input(screen, clock)
        if nickname is None:
            break

        # 2~5. 모드 선택, 실행, 결과, 랭킹 반복 루프
        while True:
            # 2. 게임 모드 선택
            game_mode = select_game_mode(screen, clock, nickname)
            if game_mode is None:
                # ESC/BACKSPACE on mode select -> back to nickname
                break

            # 3. 게임 실행
            result = play_game(screen, clock, nickname, game_mode)
            if result == "MODE_SELECT":
                # BACKSPACE during play -> return to mode select
                continue
            if result is False:
                # Quit -> exit program
                pygame.quit()
                return
            # result == True -> play ended normally

            # 4. 모드 1일 때 순위표
            if game_mode == 1:
                rank_res = show_ranking(screen, clock)
                if rank_res == "MODE_SELECT":
                    # BACKSPACE on ranking -> back to mode select
                    continue
                if rank_res is False:
                    # Quit on ranking -> exit
                    pygame.quit()
                    return
                # rank_res == True: finished ranking, fall through

            # 5. 모든 모드 종료 시 -> 다시 모드 선택
            continue  # back to top of mode loop

        # inner loop exited -> back to nickname entry
    pygame.quit()

if __name__ == "__main__":
    main()
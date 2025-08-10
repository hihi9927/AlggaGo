# test_env.py
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
from pymunk import Vec2d

from physics import (
    create_stone,
    WIDTH, HEIGHT, MARGIN,
    STONE_RADIUS,
    scale_force,
    all_stones_stopped
)

MAX_PHYSICS_STEPS_PER_ACTION = 600

class AlggaGoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi, -0.5], dtype=np.float32),
            high=np.array([1.0, np.pi, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 25, dtype=np.float32),
            high=np.array([float(max(WIDTH, HEIGHT))] * 25, dtype=np.float32),
            shape=(25,),
            dtype=np.float32
        )
        self.stones = []
        self.space = None
        self.initial_black_count = 0
        self.initial_white_count = 0
        self.current_player = "black"
        self.reset()

    def set_current_player(self, color: str):
        if color not in ["white", "black"]:
            raise ValueError("Player color must be 'white' or 'black'.")
        self.current_player = color

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.space:
            for body in list(self.space.bodies): self.space.remove(body)
            for shape in list(self.space.shapes): self.space.remove(shape)
            for constraint in list(self.space.constraints): self.space.remove(constraint)
        self.space = pymunk.Space(); self.space.gravity = (0, 0); self.space.damping = 0.1
        self.stones.clear()
        static_body = self.space.static_body
        corners = [(MARGIN, MARGIN), (WIDTH-MARGIN, MARGIN), (WIDTH-MARGIN, HEIGHT-MARGIN), (MARGIN, HEIGHT-MARGIN)]
        for i in range(4):
            a, b = corners[i], corners[(i+1)%4]
            seg = pymunk.Segment(static_body, a, b, 1); seg.sensor = True; self.space.add(seg)

        y_top_center = MARGIN + STONE_RADIUS*2 + 66.5
        y_bottom_center = HEIGHT - MARGIN - STONE_RADIUS*2 - 66.5
        MIN_DIST = STONE_RADIUS * 3.0

        def is_far_enough(x, y, placed_positions):
            for px, py in placed_positions:
                if np.linalg.norm([x - px, y - py]) < MIN_DIST:
                    return False
            return True

        white_placed_positions = []
        while len(white_placed_positions) < 4:
            x = np.random.uniform(MARGIN + STONE_RADIUS, WIDTH - MARGIN - STONE_RADIUS)
            y = y_top_center + np.random.uniform(-40, 40)
            if is_far_enough(x, y, white_placed_positions):
                white_placed_positions.append((x, y))
                self.stones.append(create_stone(self.space, (x, y), (255, 255, 255)))

        black_placed_positions = []
        while len(black_placed_positions) < 4:
            x = np.random.uniform(MARGIN + STONE_RADIUS, WIDTH - MARGIN - STONE_RADIUS)
            y = y_bottom_center + np.random.uniform(-40, 40)
            if is_far_enough(x, y, black_placed_positions):
                black_placed_positions.append((x, y))
                self.stones.append(create_stone(self.space, (x, y), (0, 0, 0)))

        self.initial_black_count = 4
        self.initial_white_count = 4
        self.current_player = options.get("initial_player", "black") if options else "black"
        return self._get_obs(), {"initial_player": self.current_player}

    def _get_obs(self):
        obs = []
        sorted_stones = sorted(self.stones, key=lambda s: (s.color[:1], s.body.position.x))
        for shape in sorted_stones:
            if shape.body in self.space.bodies:
                x, y = shape.body.position
                stone_is_white = shape.color[:3] == (255, 255, 255)
                is_mine = 1.0 if (self.current_player == "white" and stone_is_white) or \
                                 (self.current_player == "black" and not stone_is_white) else 0.0
                if self.current_player == "black": y = HEIGHT - y
                obs.extend([float(x), float(y), float(is_mine)])
            else: obs.extend([-1.0, -1.0, -1.0])
        while len(obs) < 8 * 3: obs.extend([-1.0, -1.0, -1.0])
        obs.append(1.0 if self.current_player == "white" else 0.0)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        from opponent import rule_based_action
        terminated = False; truncated = False; reward = 0.0; info = {}
        
        rule_action = rule_based_action(self.stones, self.current_player)
        if rule_action is None:
            reward = 0.0; terminated = True # reward -= 5.0
            return self._get_obs(), reward, terminated, truncated, info

        rule_idx, rule_angle, rule_force = rule_action
        if self.current_player == "black":
            index_weight, angle_offset, force_offset = np.squeeze(action)
        else:
            index_weight, angle_offset, force_offset = 0.0, 0.0, 0.0
        
        player_color_tuple = (255, 255, 255) if self.current_player == "white" else (0, 0, 0)
        player_stones = [s for s in self.stones if s.color[:3] == player_color_tuple]
        if not player_stones:
            reward = 0.0; terminated = True # reward -= 5.0
            return self._get_obs(), reward, terminated, truncated, info
        
        if len(player_stones) > 1:
            idx_offset = int(np.round(index_weight * (len(player_stones) - 1) / 2))
            final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
        else: final_idx = 0
        
        final_angle = rule_angle + angle_offset
        final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
        
        selected_stone_to_shoot = player_stones[final_idx]
        direction = Vec2d(1, 0).rotated(final_angle)
        impulse = direction * scale_force(final_force)
        selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

        physics_step_count = 0
        while not all_stones_stopped(self.stones) and physics_step_count < MAX_PHYSICS_STEPS_PER_ACTION:
            self.space.step(1/60.0); physics_step_count += 1
            for shape in self.stones[:]:
                if not (MARGIN<shape.body.position.x<WIDTH-MARGIN and MARGIN<shape.body.position.y<HEIGHT-MARGIN):
                    self.space.remove(shape, shape.body); self.stones.remove(shape)
        
        current_black = sum(1 for s in self.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in self.stones if s.color[:3] == (255,255,255))
        
        prev_black = self.initial_black_count
        prev_white = self.initial_white_count
        black_removed = prev_black - current_black
        white_removed = prev_white - current_white

        reward = 0.0

        # 자살성 움직임 패널티 (상대 돌 하나도 제거 못함)
        if self.current_player == "black" and white_removed == 0:
            reward -= 5.0
        elif self.current_player == "white" and black_removed == 0:
            reward -= 5.0

        # 상대 돌 제거 보상
        if self.current_player == "black":
            reward += white_removed * 15.0
            if white_removed >= 2:
                reward += 30.0
        elif self.current_player == "white":
            reward += black_removed * 15.0
            if black_removed >= 2:
                reward += 30.0

        # 게임 종료 시 승리 보상
        if current_black == 0 and current_white > 0:
            terminated = True
            info['winner'] = 'white'
            if self.current_player == 'white':
                reward += 30.0
            else:
                reward -= 30.0
        elif current_white == 0 and current_black > 0:
            terminated = True
            info['winner'] = 'black'
            if self.current_player == 'black':
                reward += 30.0
            else:
                reward -= 30.0
        
        self.initial_black_count = current_black; self.initial_white_count = current_white
        if not terminated: self.current_player = "white" if self.current_player == "black" else "black"
        info["current_black_stones"] = current_black; info["current_white_stones"] = current_white
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, screen):
        screen.fill((150, 150, 150))
        pygame.draw.rect(screen, (210, 180, 140), pygame.Rect(MARGIN, MARGIN, WIDTH - 2*MARGIN, HEIGHT - 2*MARGIN))
        for shape in self.stones:
            pygame.draw.circle(screen, shape.color[:3], (int(shape.body.position.x), int(shape.body.position.y)), STONE_RADIUS)

    def close(self):
        pass
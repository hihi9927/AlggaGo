import math
import numpy as np
import pygame
import pymunk
from pymunk import Vec2d

# ── Board (5:1) ───────────────────────────────────────────────────────────────
WIDTH  = 1000
HEIGHT = 200
MARGIN = 15

# ── Stone ── exact same values as AlggaGo/physics.py ─────────────────────────
STONE_RADIUS = 25
STONE_MASS   = 1

# ── Force ── exact same values as AlggaGo/physics.py ─────────────────────────
MAX_DRAG_LENGTH  = 100
FORCE_MULTIPLIER = 20
MIN_FORCE        = 20


def scale_force(force_normalized: float) -> float:
    """Identical to AlggaGo scale_force()."""
    MAX_FORCE = MAX_DRAG_LENGTH * FORCE_MULTIPLIER
    return MIN_FORCE + force_normalized * (MAX_FORCE - MIN_FORCE)


def _can_1ta2pi(bx: float, by: float,
                w1x: float, w1y: float,
                w2x: float, w2y: float,
                n_angles: int = 36,
                n_forces: int = 5) -> bool:
    """
    각도 × 힘 grid를 순수 numpy로 시뮬레이션 — 검정돌이 살아남으면서
    두 흰돌을 모두 경계 밖으로 내보내는 수가 존재하는지 검증.
    """
    R    = STONE_RADIUS
    dmp  = 0.1 ** (1 / 60)
    dt   = 1 / 60.0
    force_norms = np.linspace(0.3, 1.0, n_forces)

    for fi in range(n_forces):
        sp0 = scale_force(force_norms[fi])
        for i in range(n_angles):
            angle = -math.pi / 2 + math.pi * i / max(n_angles - 1, 1)

            pos = [np.array([bx,  by],  dtype=float),
                   np.array([w1x, w1y], dtype=float),
                   np.array([w2x, w2y], dtype=float)]
            vel = [np.array([math.cos(angle), math.sin(angle)]) * sp0,
                   np.zeros(2), np.zeros(2)]
            alive = [True, True, True]
            w1_out = w2_out = False

            for _ in range(400):
                for k in range(3):
                    if alive[k]:
                        vel[k] *= dmp
                        pos[k] += vel[k] * dt

                for a, b in ((0, 1), (0, 2), (1, 2)):
                    if not alive[a] or not alive[b]:
                        continue
                    diff = pos[b] - pos[a]
                    dist = np.linalg.norm(diff)
                    if 0 < dist < 2 * R:
                        n = diff / dist
                        ov = (2 * R - dist) * 0.5
                        pos[a] -= n * ov
                        pos[b] += n * ov
                        rv = float(np.dot(vel[a] - vel[b], n))
                        if rv > 0:
                            vel[a] -= rv * n
                            vel[b] += rv * n

                for k in range(3):
                    if alive[k]:
                        x, y = pos[k]
                        if not (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
                            alive[k] = False
                            if k == 1: w1_out = True
                            if k == 2: w2_out = True

                if w1_out and w2_out and alive[0]:  # 검정돌 생존 + 흰돌 둘 다 탈락
                    break
                if all(not alive[k] or np.linalg.norm(vel[k]) <= 5 for k in range(3)):
                    break

            if w1_out and w2_out and alive[0]:
                return True

    return False


class BilliardEnv:
    def __init__(self, render: bool = False):
        self.do_render = render
        # declare all instance attrs upfront so type checkers can infer types
        self.space:             pymunk.Space = pymunk.Space()
        self.black:             pymunk.Shape
        self.whites:            list = []
        self.hit:               list = [False, False]
        self._white_contacted:  bool = False
        self.done:              bool = False
        self.screen: pygame.Surface | None = None
        self.clock:  pygame.time.Clock | None = None
        self.font = None
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Playground - Billiard RL")
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont("malgun gothic", 20)
            except Exception:
                self.font = pygame.font.Font(None, 24)
        self.reset()

    @property
    def black_pos(self) -> Vec2d:
        """Black ball position as Vec2d (safe accessor for play.py)."""
        body = self.black.body
        assert body is not None
        return Vec2d(body.position.x, body.position.y)

    # ── Pymunk helpers ─────────────────────────────────────────────────────────
    def _new_space(self) -> pymunk.Space:
        space = pymunk.Space()
        space.gravity = (0, 0)
        space.damping = 0.1              # same as AlggaGo
        borders = [
            ((MARGIN, MARGIN),               (WIDTH - MARGIN, MARGIN)),
            ((WIDTH - MARGIN, MARGIN),        (WIDTH - MARGIN, HEIGHT - MARGIN)),
            ((WIDTH - MARGIN, HEIGHT - MARGIN),(MARGIN,         HEIGHT - MARGIN)),
            ((MARGIN,         HEIGHT - MARGIN),(MARGIN,         MARGIN)),
        ]
        for a, b in borders:
            seg = pymunk.Segment(space.static_body, a, b, 1)
            seg.sensor = True            # same as AlggaGo: no bounce, out-of-bounds removal
            space.add(seg)
        return space

    def _add_stone(self, x: float, y: float, color: tuple) -> pymunk.Shape:
        moment = pymunk.moment_for_circle(STONE_MASS, 0, STONE_RADIUS)
        body   = pymunk.Body(STONE_MASS, moment)
        body.position = (x, y)
        shape = pymunk.Circle(body, STONE_RADIUS)
        shape.elasticity       = 1.0          # same as AlggaGo
        shape.friction         = 0.9          # same as AlggaGo
        shape.stone_color      = color
        shape.collision_type   = 1 if color == (0, 0, 0) else 2  # black=1, white=2
        self.space.add(body, shape)
        return shape

    # ── Episode lifecycle ──────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.space  = self._new_space()
        bx = MARGIN + STONE_RADIUS * 2
        by = HEIGHT / 2
        self.black  = self._add_stone(bx, by, (0, 0, 0))
        self.whites = []
        self.hit    = [False, False]
        self._white_contacted = False
        self._place_whites()
        self.done = False
        return self._obs()

    def _place_whites(self):
        """
        _can_1ta2pi()로 실제 시뮬레이션 검증 후 배치.
        1타2피가 가능한 수가 존재하는 배치만 사용.
        """
        R  = STONE_RADIUS
        bx, by = self.black.body.position
        x1 = x2 = y1 = y2 = 0.0

        while True:
            x1 = np.random.uniform(WIDTH * 0.45, WIDTH * 0.72)
            y1 = np.random.uniform(MARGIN + R + 5, HEIGHT - MARGIN - R - 5)

            dx, dy = x1 - bx, y1 - by
            ln     = math.hypot(dx, dy)
            ux, uy = dx / ln, dy / ln

            ang    = np.random.uniform(-0.6, 0.6)
            ca, sa = math.cos(ang), math.sin(ang)
            px, py = ux * ca - uy * sa, ux * sa + uy * ca

            d  = np.random.uniform(R * 3, R * 8)
            x2 = float(np.clip(x1 + px * d, MARGIN + R, WIDTH - MARGIN - R))
            y2 = float(np.clip(y1 + py * d, MARGIN + R, HEIGHT - MARGIN - R))

            # 두 흰돌 겹침 방지
            if math.hypot(x2 - x1, y2 - y1) < R * 2.2:
                continue

            # 실제로 1타2피가 가능한 각도가 존재하는지 검증
            if _can_1ta2pi(bx, by, x1, y1, x2, y2):
                break

        self.whites = [
            self._add_stone(x1, y1, (255, 255, 255)),
            self._add_stone(x2, y2, (255, 255, 255)),
        ]

    # ── Shooting ───────────────────────────────────────────────────────────────
    def shoot_raw(self, impulse: Vec2d) -> float:
        """Manual play: apply raw impulse Vec2d (from drag mechanic)."""
        self.black.body.apply_impulse_at_world_point(impulse, self.black.body.position)
        return self._simulate()

    def step(self, angle: float, force_norm: float = 0.5):
        """RL training: angle (rad) + normalised force [0,1]."""
        direction = Vec2d(math.cos(angle), math.sin(angle))
        impulse   = direction * scale_force(force_norm)
        self.black.body.apply_impulse_at_world_point(impulse, self.black.body.position)
        reward = self._simulate()
        return self._obs(), reward, True, {"hits": sum(self.hit)}

    def _simulate(self) -> float:
        """Run physics until stopped (max 600 steps = ~10 s). Returns reward."""
        # 흰돌 초기 위치 기록 — 시뮬레이션 후 이동 여부로 접촉 판정
        white_init = [(w.body.position.x, w.body.position.y)
                      for w in self.whites if w in self.space.shapes]

        for _ in range(600):           # same cap as AlggaGo MAX_PHYSICS_STEPS_PER_ACTION
            self.space.step(1 / 60.0)

            # 경계 이탈한 돌 제거 (절반 이상 나가면 탈락)
            if self.black in self.space.shapes:
                bx, by = self.black.body.position
                if not (MARGIN < bx < WIDTH - MARGIN and MARGIN < by < HEIGHT - MARGIN):
                    self.space.remove(self.black, self.black.body)

            for i, w in enumerate(self.whites):
                if w in self.space.shapes:
                    x, y = w.body.position
                    if not (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
                        self.hit[i] = True
                        self.space.remove(w, w.body)

            if self.do_render:
                self.draw()
                self.clock.tick(120)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        return 0.0
            if self._stopped():
                break
        # 접촉 판정: 판 위에 남아있는 흰돌이 초기 위치에서 1px 이상 이동했으면 맞은 것
        for i, (ix, iy) in enumerate(white_init):
            w = self.whites[i]
            if w in self.space.shapes:
                if math.hypot(w.body.position.x - ix, w.body.position.y - iy) > 1.0:
                    self._white_contacted = True
                    break
            else:
                self._white_contacted = True  # 탈락한 돌도 맞은 것
                break

        self.done = True
        return self._reward()

    def _stopped(self, threshold: float = 5.0) -> bool:
        """Same threshold as AlggaGo all_stones_stopped(). Skips removed stones."""
        active = [s for s in [self.black] + self.whites if s in self.space.shapes]
        return all(s.body.velocity.length <= threshold for s in active)

    # ── Reward ─────────────────────────────────────────────────────────────────
    def _reward(self) -> float:
        n = sum(self.hit)
        black_alive = self.black in self.space.shapes
        if n == 2 and black_alive: return 2.0   # 1타2피
        if n >= 1:                 return 1.0   # 1타1피
        if self._white_contacted:  return -0.1   # 맞혔지만 탈락 못 시킴
        return -0.5                              # 완전 miss

    # ── Observation ────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        bx, by = self.black.body.position
        return np.array([
            bx / WIDTH, by / HEIGHT,
            self.whites[0].body.position.x / WIDTH,
            self.whites[0].body.position.y / HEIGHT,
            self.whites[1].body.position.x / WIDTH,
            self.whites[1].body.position.y / HEIGHT,
        ], dtype=np.float32)

    # ── Rendering ── matches AlggaGo render() exactly ─────────────────────────
    def draw(self, aim_line=None, message=None):
        if not self.screen:
            return

        # Background — (150, 150, 150)
        self.screen.fill((150, 150, 150))

        # Board surface — (210, 180, 140), no border (same as AlggaGo)
        pygame.draw.rect(
            self.screen, (210, 180, 140),
            pygame.Rect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN),
        )

        # White stones — skip removed ones (same as AlggaGo)
        for w in self.whites:
            if w in self.space.shapes:
                x, y = int(w.body.position.x), int(w.body.position.y)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), STONE_RADIUS)

        # Black stone
        if self.black in self.space.shapes:
            bx, by = int(self.black.body.position.x), int(self.black.body.position.y)
            pygame.draw.circle(self.screen, (0, 0, 0), (bx, by), STONE_RADIUS)

        # Drag guide line (red, same as AlggaGo tutorial)
        if aim_line is not None:
            pygame.draw.line(self.screen, (255, 0, 0), aim_line[0], aim_line[1], 2)

        if message and self.font:
            surf = self.font.render(message, True, (255, 255, 0))
            self.screen.blit(surf, (10, 8))

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
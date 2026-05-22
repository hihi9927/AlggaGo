import pygame
from pymunk import Vec2d
from env import BilliardEnv, STONE_RADIUS, MAX_DRAG_LENGTH, FORCE_MULTIPLIER, MIN_FORCE


def play():
    pygame.init()
    env = BilliardEnv(render=True)  # __init__에서 reset() + _place_whites() 이미 호출됨
    pygame.event.clear()  # placement blocking 동안 쌓인 이벤트 제거

    assert env.clock is not None

    dragging   = False
    drag_start = Vec2d(0, 0)
    message    = "돌을 클릭해서 당긴 뒤 놓으세요  |  R: 리셋  |  ESC: 종료"

    while True:
        # ── Drag guide line (stone → mouse direction, red) ────────────────────
        aim_line = None
        if dragging:
            mx, my = pygame.mouse.get_pos()
            bp     = env.black_pos
            raw    = Vec2d(mx - bp.x, my - bp.y)
            dist   = raw.length
            if dist > 0:
                end      = bp + raw.normalized() * min(dist, MAX_DRAG_LENGTH)
                aim_line = ((int(bp.x), int(bp.y)), (int(end.x), int(end.y)))

        env.draw(aim_line=aim_line, message=message)

        # ── Events ────────────────────────────────────────────────────────────
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                env.close()
                return

            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_ESCAPE:
                    env.close()
                    return
                if evt.key == pygame.K_r:
                    env.reset()
                    pygame.event.clear()  # placement 동안 쌓인 이벤트 제거
                    dragging = False
                    message  = "돌을 클릭해서 당긴 뒤 놓으세요  |  R: 리셋  |  ESC: 종료"

            # ── Click on black stone to start drag ────────────────────────────
            elif evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1 and not env.done:
                mouse_pos = Vec2d(float(evt.pos[0]), float(evt.pos[1]))
                if (env.black_pos - mouse_pos).length <= STONE_RADIUS:
                    dragging   = True
                    drag_start = env.black_pos

            # ── Release: shoot in opposite direction of drag ───────────────────
            elif evt.type == pygame.MOUSEBUTTONUP and evt.button == 1 and dragging:
                drag_end = Vec2d(float(evt.pos[0]), float(evt.pos[1]))
                raw_vec  = drag_start - drag_end    # away from mouse = shoot direction
                dist     = raw_vec.length

                if dist > MAX_DRAG_LENGTH:
                    raw_vec = raw_vec.normalized() * MAX_DRAG_LENGTH
                    dist    = MAX_DRAG_LENGTH

                if dist > 0:
                    # Same formula as AlggaGo main.py line 1789
                    impulse = raw_vec.normalized() * (dist * FORCE_MULTIPLIER + MIN_FORCE)
                    env.shoot_raw(impulse)
                    hits    = sum(env.hit)
                    message = ["miss (-1)  |  R: 리셋",
                               "1타1피! (+1)  |  R: 리셋",
                               "1타2피!! (+3)  |  R: 리셋"][hits]

                dragging = False

        env.clock.tick(60)


if __name__ == "__main__":
    play()

import csv
import os
import time
import datetime
import pygame
from env import BilliardEnv
from agent import Agent

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
_LOGS_DIR = os.path.join(_HERE, "logs")
_DEFAULT_MODEL = os.path.join(_MODELS_DIR, "model.pth")


def train(episodes=10000, render_every=1000, model_path=_DEFAULT_MODEL, log_path=None):
    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.makedirs(_LOGS_DIR, exist_ok=True)
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(_LOGS_DIR, f"log_{ts}.csv")
    env   = BilliardEnv(render=False)
    agent = Agent(lr=3e-3)
    counts       = {0: 0, 1: 0, 2: 0}
    total_reward = 0.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "miss", "1피", "2피", "총보상"])

        for ep in range(1, episodes + 1):
            obs          = env.reset()
            angle, force = agent.act(obs)
            _, reward, _, info = env.step(angle, force)
            agent.learn(reward)
            counts[info["hits"]] += 1
            total_reward += reward

            if ep % 100 == 0:
                total = sum(counts.values())
                miss_r  = counts[0] / total
                one_r   = counts[1] / total
                two_r   = counts[2] / total
                print(f"[{ep:6d}/{episodes}]  "
                      f"miss:{miss_r:.0%}  "
                      f"1피:{one_r:.0%}  "
                      f"2피:{two_r:.0%}  "
                      f"총보상:{total_reward:.1f}")
                writer.writerow([ep, f"{miss_r:.4f}", f"{one_r:.4f}", f"{two_r:.4f}", f"{total_reward:.2f}"])
                f.flush()
                counts       = {0: 0, 1: 0, 2: 0}
                total_reward = 0.0

            if ep % render_every == 0:
                _demo(agent)

    agent.save(model_path)
    print(f"저장 완료: {model_path}")


def _demo(agent):
    env = BilliardEnv(render=True)
    obs          = env.reset()
    angle, force = agent.act(obs, greedy=True)
    _, reward, _, info = env.step(angle, force)
    labels = ["miss", "1타1피", "1타2피!"]
    env.draw(message=f"{labels[info['hits']]}  reward={reward:.0f}")
    time.sleep(1.5)
    env.close()


def watch(model_path="models/model.pth", rounds=15):
    agent = Agent()
    agent.load(model_path)
    env = BilliardEnv(render=True)
    assert env.clock is not None
    clock = env.clock
    score = 0
    print(f"AI 점수 게임 시작 (총 {rounds}판) — 1타2피+생존 시 +1점")

    def wait_sec(seconds: float) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            clock.tick(60)
        return True

    for rnd in range(1, rounds + 1):
        obs = env.reset()
        pygame.event.clear()

        env.draw(message=f"{rnd}/{rounds}판  누적: {score}점  준비 중...")
        if not wait_sec(1.0):
            break

        angle, force = agent.act(obs, greedy=True)
        _, _, _, info = env.step(angle, force)

        black_alive = env.black in env.space.shapes
        hits = info["hits"]
        if hits == 2 and black_alive:
            score += 100
            result = f"1타2피!! +100  →  누적: {score}점"
        elif hits == 2:
            score += 1
            result = f"1타2피 (검정돌 탈락) +1  →  누적: {score}점"
        elif hits == 1:
            score += 1
            result = f"1타1피 +1  →  누적: {score}점"
        else:
            result = f"miss  누적: {score}점"

        print(f"[{rnd:2d}/{rounds}] {result}")
        env.draw(message=f"{rnd}/{rounds}판  {result}")
        if not wait_sec(1.5):
            break
    else:
        env.draw(message=f"게임 종료!  최종 점수: {score}/{rounds}점")
        print(f"\n최종 점수: {score}/{rounds}점")
        wait_sec(3.0)

    env.close()


if __name__ == "__main__":
    train()
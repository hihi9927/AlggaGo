import time
import pygame
from env import BilliardEnv
from agent import Agent

_MODEL = "models/model.pth"


def watch(model_path=_MODEL):
    agent = Agent()
    agent.load(model_path)
    env = BilliardEnv(render=True)
    assert env.clock is not None
    clock = env.clock
    labels = ["miss (-1)", "1타1피 (+1)", "1타2피!! (+3)"]
    print("AI 관전 중... (창 닫으면 종료)")

    def wait_sec(seconds: float) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            clock.tick(60)
        return True

    while True:
        obs = env.reset()
        pygame.event.clear()

        env.draw(message="AI 발사 준비 중...")
        if not wait_sec(1.0):
            break

        angle, force = agent.act(obs, greedy=True)
        print(f"angle: {angle}, force: {force}")
        _, reward, _, info = env.step(angle, force)

        env.draw(message=labels[info["hits"]])
        if not wait_sec(1.0):
            break

    env.close()


if __name__ == "__main__":
    model_no = int(input("Enter the model number: "))
    model_path = ""
    if model_no == 0: model_path = _MODEL
    else: model_path = f"models/model_{model_no}.pth"
    watch(model_path=model_path)

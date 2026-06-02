import csv
import os
import json
import time
import datetime
import pygame
from env import BilliardEnv
from agent import Agent

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
_LOGS_DIR = os.path.join(_HERE, "logs")
_DEFAULT_MODEL = os.path.join(_MODELS_DIR, "model.pth")

COUNTER_FILE = "model_counter.json"

def load_counter():
    path = os.path.join(_HERE, COUNTER_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["last_model_no"]
    return 0

def save_counter(no=0):
    path = os.path.join(_HERE, COUNTER_FILE)
    with open(path, "w") as f:
        json.dump({"last_model_no": no}, f)


def train(episodes=5000, train_section=500, n_test=50, save_every=1000, model_no=None, model_path=_DEFAULT_MODEL, log_path=None):
    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.makedirs(_LOGS_DIR, exist_ok=True)
    if model_no is None or log_path is None:
        model_no = load_counter() + 1
        model_path = os.path.join(_MODELS_DIR, f"model_{model_no}.pth")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(_LOGS_DIR, f"log_model_{model_no}_[{ts}].csv")
        save_counter(model_no)

    env   = BilliardEnv(render=False)
    agent = Agent(lr=5e-4)
    # counts = ["miss", "one", "two", "black alive", "white touch"]
    counts       = [0, 0, 0, 0, 0]
    episode_elapsed = 0
    total_reward = 0.0
    save_timing = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "miss", "1피", "2피", "총보상"])

        while episode_elapsed <= episodes:
            # Training for a certain number of episodes
            for ep in range(train_section):
                obs = env.reset()
                angle, force = agent.act(obs)
                _, reward, _, info = env.step(angle, force)
                agent.learn(reward)
            
            episode_elapsed += train_section

            # Test the trained model and write the log
            for tc in range(1, n_test + 1):
                obs = env.reset()
                angle, force = agent.act(obs, greedy=True)
                _, reward, _, info = env.step(angle, force)
                total_reward += reward
                counts[info["hits"]] += 1; counts[3] += info["alive"]; counts[4] += info["touched"]

            total_trial = counts[0] + counts[1] + counts[2]
            miss_r = counts[0] / total_trial
            one_r = counts[1] / total_trial
            two_r = counts[2] / total_trial
            alive_r = counts[3] / total_trial
            touch_r = counts[4] / total_trial
            print(f"[{episode_elapsed:6d}/{episodes}]  "
                      f"miss:{miss_r:.0%}  "
                      f"1피:{one_r:.0%}  "
                      f"2피:{two_r:.0%}  "
                      f"총보상:{total_reward:.1f}  "
                      f"생존:{alive_r}  "
                      f"터치:{touch_r}")
            writer.writerow([episode_elapsed, f"{miss_r:.4f}", f"{one_r:.4f}", f"{two_r:.4f}", f"{total_reward:.2f}", f"{alive_r}", f"{touch_r}"])
            f.flush()
            counts = [0, 0, 0, 0, 0]
            total_reward = 0.0

            if episode_elapsed % save_every >= save_timing:
                agent.save(model_path)
                print(f"Save complete: {model_path}")
                save_timing = episode_elapsed % save_every
            

        


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

# 필요 패키지: pip install pygame torch numpy
#
# 실행 방법:
#   python main.py play          -- 직접 플레이
#   python main.py train         -- AI 학습
#   python main.py watch         -- 학습된 AI 관전
#   python main.py train --episodes 20000 --model my.pth

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playground - Billiard RL")
    parser.add_argument("mode", nargs="?", default="play",
                        choices=["play", "train", "watch"],
                        help="play: 직접 플레이 | train: AI 학습 | watch: AI 관전")
    parser.add_argument("--episodes", type=int, default=10000, help="학습 에피소드 수")
    parser.add_argument("--model",    default="models/model.pth", help="모델 파일 경로")
    parser.add_argument("--log",      default="log.csv",       help="로그 CSV 파일 경로")
    args = parser.parse_args()

    if args.mode == "play":
        from play import play
        play()
    elif args.mode == "train":
        from train import train
        train(episodes=args.episodes, model_path=args.model, log_path=args.log)
    elif args.mode == "watch":
        from train import watch
        watch(model_path=args.model)
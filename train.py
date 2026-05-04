import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alggago.training.strategies import run_competitive_training


def main():
    run_competitive_training()


if __name__ == "__main__":
    main()

# AlggaGo - 강화학습 알까기 AI 프로젝트

## 프로젝트 개요

물리 기반 보드게임 알까기를 플레이하는 AI를 강화학습(PPO)으로 직접 설계하는 프로젝트입니다.

게임 환경, 물리 엔진, 학습 루프, 시각화, 파일 저장은 **모두 고정**되어 있습니다.  
여러분이 할 일은 딱 두 가지입니다.

---

## 여러분이 수정하는 파일 (2개)

| 파일 | 역할 |
|------|------|
| `src/alggago/reward.py` | **보상 함수 설계** — AI가 어떤 행동에 보상/패널티를 받을지 정의 |
| `src/alggago/training/config.py` | **학습 방법 설계** — 하이퍼파라미터, 학습 스케줄, 평가 빈도 등 조정 |

> 이 두 파일 외의 코드는 수정하지 않아도 됩니다.  
> 학습 중 시각화 화면, 모델 저장 위치, 로그 파일 등은 자동으로 고정된 방식으로 처리됩니다.

---

## 디렉토리 구조

```
AlggaGo1.9/
├── train.py                          # 학습 실행 진입점 (수정 금지)
├── main.py                           # 게임 실행 (플레이 / AI 대전 / 관람)
└── src/alggago/
    ├── reward.py                     # 보상 함수 정의          ← 수정 가능
    ├── training/
    │   └── config.py                 # 학습 하이퍼파라미터      ← 수정 가능
    │
    │   (이하 수정 금지)
    ├── env.py                        # 게임 환경 (Gymnasium)
    ├── physics.py                    # 물리 엔진
    ├── visualize.py                  # 학습 중 시각화
    ├── training/
    │   ├── strategies.py             # 학습 루프 (예선전 + 경쟁 학습)
    │   ├── evaluation.py             # 평가 함수
    │   ├── callbacks.py              # 학습 콜백
    │   ├── env_wrappers.py           # 환경 래퍼
    │   └── model_utils.py            # 모델 저장/불러오기 유틸
    └── agents/
        ├── rl_agent.py               # RL 에이전트 로딩/추론
        ├── rule_base.py              # 규칙 기반 행동 로직
        └── model_c.py                # 상대 AI (Model C)
```

---

## 학습 실행

```bash
python train.py
```

실행하면 두 단계가 자동으로 진행됩니다.

**1단계 — 예선전 (Model A vs Model C)**  
새 AI(Model A)가 고정 규칙 기반 상대(Model C)를 이길 때까지 반복 학습합니다.

**2단계 — 경쟁 학습 (Model A vs Model B)**  
Model B는 Model A의 복제본으로 시작하여, 둘이 서로를 상대로 경쟁하며 성장합니다.

학습 중에는 pygame 시각화 창이 열려 실시간 대전을 볼 수 있습니다. *(고정, 수정 불가)*  
학습을 중단했다가 재실행하면 자동으로 이전 상태부터 이어서 학습합니다.

---

## 게임 실행 (학습된 모델 테스트)

```bash
python main.py
```

`rl_models_competitive/` 폴더에 저장된 모델을 불러와 AI 대전을 관람하거나 직접 플레이할 수 있습니다.

---

## 게임 규칙

- 흑돌 4개 vs 백돌 4개로 시작
- 상대 돌을 모두 제거하면 승리
- 흑돌 선공

---

## 환경 설명 (참고용)

### 관측 공간 (Observation Space) — 25차원 벡터

```
obs[0..23] : 돌 8개의 상태 (돌 1개당 3개 값)
    - x 좌표
    - y 좌표  (항상 현재 플레이어 기준: 내 진영 = 아래)
    - is_mine : 내 돌이면 1.0, 상대 돌이면 0.0

obs[24]    : 현재 플레이어 (white=1.0, black=0.0)
```

### 행동 공간 (Action Space) — 5차원 연속 벡터 `[-1, 1]`

| 인덱스 | 의미 | 설명 |
|--------|------|------|
| `action[0]` | 일반 공격 선호도 | 높을수록 상대 돌 직접 조준 |
| `action[1]` | 틈새 공격 선호도 | 높을수록 두 돌 사이 틈새 조준 |
| `action[2]` | 돌 선택 오프셋 | 규칙 기반 추천 돌에서 몇 번째를 쏠지 조정 |
| `action[3]` | 각도 오프셋 | 조준 각도를 ±45° 범위에서 조정 |
| `action[4]` | 힘 오프셋 | 힘을 ±50% 범위에서 조정 |

AI는 규칙 기반(Rule-based)의 행동을 기본값으로 받고, 거기에 오프셋을 더해 최종 행동을 결정합니다.

---

## 보상 함수 설계 (`src/alggago/reward.py`)

> **여러분이 수정하는 첫 번째 파일입니다.**

### 함수 시그니처

```python
def reward_fn(player, prev_black, prev_white, curr_black, curr_white, info) -> float:
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `player` | `str` | 현재 행동한 플레이어 (`"black"` 또는 `"white"`) |
| `prev_black` | `int` | 이번 수 이전의 흑돌 수 |
| `prev_white` | `int` | 이번 수 이전의 백돌 수 |
| `curr_black` | `int` | 이번 수 이후의 흑돌 수 |
| `curr_white` | `int` | 이번 수 이후의 백돌 수 |
| `info` | `dict` | 추가 정보 (아래 표 참조) |

### `info` 딕셔너리

| 키 | 타입 | 설명 |
|----|------|------|
| `info['black_removed']` | `int` | 이번 턴에 제거된 흑돌 수 |
| `info['white_removed']` | `int` | 이번 턴에 제거된 백돌 수 |
| `info['is_regular_success']` | `bool` | 일반 공격으로 상대 돌 제거 성공 여부 |
| `info['is_split_success']` | `bool` | 틈새 공격 성공 여부 |
| `info['wedge_reward']` | `float` | 틈새 공격 품질 점수 (0.0~0.5, 실패 시 -0.5) |
| `info['winner']` | `str` / `None` | 승자 (`"black"`, `"white"`, 또는 `None`) |
| `info['strategy_choice']` | `int` | 선택된 전략 (0=일반 공격, 1=틈새 공격) |
| `info['current_black_stones']` | `int` | 현재 남은 흑돌 수 |
| `info['current_white_stones']` | `int` | 현재 남은 백돌 수 |

### 내 돌 vs 상대 돌 구분

```python
if player == "black":
    enemy_removed = info['white_removed']  # 내가 없앤 상대(백) 돌
    my_removed    = info['black_removed']  # 내가 잃은 내(흑) 돌
else:
    enemy_removed = info['black_removed']  # 내가 없앤 상대(흑) 돌
    my_removed    = info['white_removed']  # 내가 잃은 내(백) 돌
```

### 기본 제공 보상 구조 (참고)

```
상대 돌 제거:   +1 / +3 / +5 / +7  (1개/2개/3개/4개 제거 시)
내 돌 제거:    -2 / -4 / -6         (1개/2개/3개 손실 시)
틈새 공격 성공:  0.0 ~ +0.5
틈새 공격 실패: -0.5
승리:          +1.5 ~ +10.5  (남은 돌 수에 비례)
패배:          -1.5 ~ -10.5
```

### 설계 시 고려사항

- **희소 보상(Sparse) vs 밀집 보상(Dense):** 승패만으로 보상을 주면 학습이 느립니다. 돌 제거 등 중간 과정에도 보상을 주세요.
- **보상 스케일:** 일반적으로 -10 ~ +10 범위를 권장합니다. 너무 크거나 작으면 학습이 불안정해집니다.
- **자살 패널티:** 내 돌이 나가는 것에 패널티가 없으면 AI가 자기 돌을 희생하는 전략을 학습할 수 있습니다.
- **틈새 공격 유도:** `wedge_reward`는 틈새 공격 위치 품질을 나타냅니다. 이 값을 활용하면 두 돌을 동시에 공격하는 전략을 학습시킬 수 있습니다.

---

## 학습 방법 설계 (`src/alggago/training/config.py`)

> **여러분이 수정하는 두 번째 파일입니다.**

`TrainingConfig` 클래스의 필드를 수정하여 학습 방식을 조정합니다.

```python
@dataclass
class TrainingConfig:
    max_stages: int = 300              # 최대 학습 스테이지 수
    timesteps_per_stage: int = 50000   # 스테이지당 학습 타임스텝
    initial_ent_coef_a: float = 0.05   # Model A 초기 엔트로피
    initial_ent_coef_b: float = 0.1    # Model B 초기 엔트로피
    ent_coef_increment: float = 0.1    # 엔트로피 증가 폭
    max_ent_coef: float = 0.5          # 최대 엔트로피
    eval_episodes_for_competition: int = 200   # 경쟁 평가 에피소드 수
    gauntlet_eval_episodes_per_color: int = 100  # 예선 평가 에피소드 수
    gauntlet_timesteps: int = 50000    # 예선 라운드당 타임스텝
    n_envs_vs_c: int = 2               # 예선 병렬 환경 수
    n_envs_self_play: int = 2          # 경쟁 학습 병렬 환경 수
```

### 주요 개념

**엔트로피 (Entropy)**  
AI가 얼마나 다양한 행동을 탐색할지를 결정합니다.
- 높을수록 더 다양한 전략 탐색 (초기 탐험에 유리)
- 낮을수록 학습된 전략에 집중 (후기 수렴에 유리)

현재 구조에서는 **A가 B를 이길수록 B의 엔트로피가 단계적으로 증가**합니다. 강해질수록 더 다양한 전략을 탐색하도록 유도됩니다.

### 커스터마이징 아이디어

- `timesteps_per_stage` 증가 → 각 스테이지에서 더 오래 학습
- `initial_ent_coef_a` 조정 → 초기 탐험 성향 변경
- `gauntlet_timesteps` 증가 → 예선 라운드를 더 길게
- `max_ent_coef` 낮춤 → 최대 탐험 범위 제한
- `eval_episodes_for_competition` 증가 → 더 정확한 경쟁 평가 (대신 학습 시간 증가)

---

## 모델 저장 위치 (자동, 수정 불가)

학습이 진행되면 모델이 자동으로 저장됩니다.

```
rl_models_competitive/
├── model_a_50000_0.050.zip      # Model A (스텝 수, 엔트로피)
├── model_b_100000_0.100.zip     # Model B
├── best_model.zip               # 전체 학습 중 가장 높은 승률의 모델
└── training_state.npy           # 학습 재개를 위한 상태
```

학습 로그는 `rl_logs_competitive/` 폴더에 CSV 파일로 저장됩니다.

---

## 의존성 설치

```bash
pip install pygame pymunk gymnasium stable-baselines3 torch numpy tqdm
```

---

## 자주 발생하는 오류

| 오류 | 원인 | 해결 방법 |
|------|------|-----------|
| `ModuleNotFoundError` | 패키지 미설치 | `pip install` 명령어로 설치 |
| 학습이 전혀 개선 안 됨 | 보상 함수 문제 | 보상 값의 부호(+/-)와 크기를 점검 |
| AI가 항상 같은 행동만 함 | 엔트로피 너무 낮음 | `initial_ent_coef_a` 값을 높여보세요 |
| AI가 너무 무작위로 행동 | 엔트로피 너무 높음 | `max_ent_coef` 값을 낮춰보세요 |
| `FileNotFoundError` (모델 없음) | 학습 전 main.py 실행 | `train.py`를 먼저 실행하세요 |

---

## 수정 가능 파일 요약

| 파일 | 수정 가능 여부 | 설명 |
|------|--------------|------|
| `src/alggago/reward.py` | **수정 가능** | 보상 함수 — 학습 신호 설계 |
| `src/alggago/training/config.py` | **수정 가능** | 학습 하이퍼파라미터 및 방법 설계 |
| 그 외 모든 파일 | **수정 금지** | 환경, 물리, 학습 루프, 시각화, 저장 |

"""
보상 함수 (학생 편집 영역)

이 파일에서 보상 함수를 정의하세요.

함수 시그니처:
    reward_fn(player, prev_black, prev_white, curr_black, curr_white, info) -> float

파라미터:
    player      : 현재 플레이어 ("black" 또는 "white")
    prev_black  : 이전 스텝의 흑돌 수
    prev_white  : 이전 스텝의 백돌 수
    curr_black  : 현재 스텝의 흑돌 수
    curr_white  : 현재 스텝의 백돌 수
    info        : 추가 정보 딕셔너리
        - info['black_removed']       : 이번 스텝에 제거된 흑돌 수
        - info['white_removed']       : 이번 스텝에 제거된 백돌 수
        - info['is_regular_success']  : 일반 공격 성공 여부 (bool)
        - info['is_split_success']    : 틈새 공격 성공 여부 (bool)
        - info['wedge_reward']        : 틈새 공격 품질 점수 (0.0 ~ 0.5, 실패 시 -0.5)
        - info['winner']              : 승자 ("black", "white", None)
        - info['strategy_choice']     : 선택된 전략 (0=일반, 1=틈새)
        - info['current_black_stones']: 현재 흑돌 수
        - info['current_white_stones']: 현재 백돌 수

반환값:
    float : 보상 값
"""


def reward_fn(player, prev_black, prev_white, curr_black, curr_white, info):
    reward = 0.0

    black_removed   = info['black_removed']
    white_removed   = info['white_removed']
    strategy_choice = info['strategy_choice']
    winner          = info.get('winner')

    opp_removed = white_removed if player == "black" else black_removed
    own_removed = black_removed if player == "black" else white_removed

    # 틈새 공격 선택 억제 (일반 공격으로 유도)
    if strategy_choice == 1:
        reward -= 1.0

    # 상대 돌 제거 보상 / 미스 패널티
    if opp_removed == 0:    reward -= 0.5
    elif opp_removed == 1:  reward += 2.0
    elif opp_removed == 2:  reward += 5.0
    elif opp_removed == 3:  reward += 9.0
    elif opp_removed >= 4:  reward += 14.0

    # 내 돌 손실 패널티
    if own_removed == 1:    reward -= 1.5
    elif own_removed == 2:  reward -= 3.5
    elif own_removed == 3:  reward -= 6.0

    # 승리/패배 보너스 — 중간 보상의 합보다 훨씬 크게 설정
    if winner is not None:
        my_curr  = curr_black if player == "black" else curr_white
        opp_curr = curr_white if player == "black" else curr_black
        if winner == player:
            reward += 20.0 + 4.0 * my_curr   # +24 ~ +36
        else:
            reward -= 20.0 + 4.0 * opp_curr  # -24 ~ -36

    return reward

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
    wedge_reward    = info['wedge_reward']
    winner          = info.get('winner')

    # 틈새 공격: 위치 품질이 곧 보상
    if strategy_choice == 1:
        reward = wedge_reward
    else:
        # 일반 공격: 제거한 상대 돌 수에 따라 누진 보상
        removed_count = white_removed if player == "black" else black_removed
        if removed_count == 1:   reward += 1
        elif removed_count == 2: reward += 3
        elif removed_count == 3: reward += 5
        elif removed_count >= 4: reward += 7
        # 좋은 위치에 착지했다면 추가 보상
        if wedge_reward > 0:
            reward += wedge_reward

    # 내 돌 제거 패널티
    if player == 'black':
        if black_removed == 1:   reward -= 2
        elif black_removed == 2: reward -= 4
        elif black_removed == 3: reward -= 6
    else:
        if white_removed == 1:   reward -= 2
        elif white_removed == 2: reward -= 4
        elif white_removed == 3: reward -= 6

    # 승리/패배 보너스 (남은 돌 수 비례)
    if winner == 'white':
        margin = curr_white
        W = 1.5 + 3.0 * (margin - 1) if margin > 0 else 0
        reward += W if player == 'white' else -W
    elif winner == 'black':
        margin = curr_black
        W = 1.5 + 3.0 * (margin - 1) if margin > 0 else 0
        reward += W if player == 'black' else -W

    return reward

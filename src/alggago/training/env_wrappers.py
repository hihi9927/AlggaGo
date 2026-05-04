import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from alggago.agents.model_c import model_c_action
from alggago.env import AlggaGoEnv
from alggago.reward import reward_fn



class VsModelCEnv(gym.Env):
    """
    단일 PPO 에이전트가 고정 상대(Model C)와 번갈아 싸우며 학습하도록 래핑한 환경.
    매 번의 step() 호출에서:
      - PPO(에이전트)의 수를 env.step(action)으로 반영
      - 게임 미종료면, 곧바로 C의 수를 내부에서 실행
      - 다시 PPO 차례가 된 시점의 관측 obs, reward(에이전트 관점), done 등을 반환
    """
    metadata = {"render_modes": []}

    def __init__(self, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv(reward_fn=reward_fn)  # 기존 환경 재사용
        self.agent_side = agent_side  # 'black' or 'white'
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

        # 내부 상태 추적용
        self._last_obs = None

    def set_bonus_modes(self, regular_active: bool, split_active: bool):
        """훈련 스크립트의 env_method 호출을 실제 게임 환경으로 전달합니다."""
        self.base_env.set_bonus_modes(regular_active=regular_active, split_active=split_active)

    def reset(self, *, seed=None, options=None):
        # PPO가 항상 먼저 두도록 시작 턴을 강제
        initial_player = self.agent_side
        self._last_obs, info = self.base_env.reset(options={"initial_player": initial_player})
        # 만약 시작 플레이어가 PPO가 아닌 경우, C가 먼저 한 수를 두고 PPO 차례로 맞춰줌
        if self.base_env.current_player != self.agent_side:
            self._play_model_c_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info

    def step(self, action):
        # 1) 에이전트 턴
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated:
            return obs, reward_agent, terminated, truncated, info

        # 2) 상대 턴(모델 C)
        self._play_model_c_turn()

        # 3) 종료/패널티 보정 및 다음 관측 반환
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_c_turn()
        total_reward = reward_agent + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info

    # ===== 내부 세팅 =====
    def _play_model_c_turn(self):
        # 현재 차례가 C인지 확인
        current_player_color = self.base_env.current_player
        if current_player_color == self.agent_side:
            return  # 이미 PPO 차례면 아무것도 안 함

        action_tuple = model_c_action(self.base_env.stones, current_player_color)
        if action_tuple:
            from pymunk import Vec2d
            from alggago.physics import scale_force, all_stones_stopped, WIDTH, HEIGHT, MARGIN
            idx, angle, force = action_tuple

            player_color_tuple = (0, 0, 0) if current_player_color == "black" else (255, 255, 255)
            player_stones = [s for s in self.base_env.stones if s.color[:3] == player_color_tuple]
            if 0 <= idx < len(player_stones):
                stone_to_shoot = player_stones[idx]
                direction = Vec2d(1, 0).rotated(angle)
                impulse = direction * scale_force(force)
                stone_to_shoot.body.apply_impulse_at_world_point(impulse, stone_to_shoot.body.position)

            # 물리 진행 (평가 코드와 동일 처리):
            from alggago.physics import all_stones_stopped, WIDTH, HEIGHT, MARGIN
            physics_steps = 0
            while not all_stones_stopped(self.base_env.stones) and physics_steps < 600:
                self.base_env.space.step(1/60.0)
                physics_steps += 1

            # 바깥으로 나간 돌 제거
            for shape in self.base_env.stones[:]:
                x, y = shape.body.position
                if not (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
                    if shape in self.base_env.space.shapes:
                        self.base_env.space.remove(shape, shape.body)
                    if shape in self.base_env.stones:
                        self.base_env.stones.remove(shape)

            # 턴 전환 및 관측 업데이트
            self.base_env.current_player = "white" if current_player_color == "black" else "black"

    def _check_terminal_and_penalty_after_c_turn(self):
        """
        C 턴 진행 직후 종료 여부와 에이전트 관점 패널티를 계산.
        env.step 내부의 보상은 '방금 둔 쪽' 기준으로 산출되므로,
        C가 이겨서 끝난 경우 에이전트에게 강한 패널티를 더해줌(-5.0).
        """
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0 and current_white > 0:
            # 백만 살아남음
            winner = "white"
            terminated = True
        elif current_white == 0 and current_black > 0:
            winner = "black"
            terminated = True
        else:
            return False, 0.0

        # 에이전트 패배 시에만 추가 패널티
        agent_color = self.agent_side
        if (winner == "white" and agent_color == "black") or (winner == "black" and agent_color == "white"):
            return True, -5.0
        return True, 0.0

class VsFixedOpponentEnv(gym.Env):
    """
    단일 PPO 에이전트가 '고정된 PPO 상대(opponent_model)'와 번갈아 싸우며 학습하는 환경.
    """
    metadata = {"render_modes": []}

    def __init__(self, opponent_model: PPO, agent_side="black"):
        super().__init__()
        self.base_env = AlggaGoEnv(reward_fn=reward_fn)
        self.opponent = opponent_model
        self.agent_side = agent_side  # 'black' or 'white'
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        initial_player = self.agent_side
        self._last_obs, info = self.base_env.reset(options={"initial_player": initial_player})
        # 시작 차례가 에이전트가 아니면, 상대가 먼저 한 수를 두고 에이전트 차례로 맞춤
        if self.base_env.current_player != self.agent_side:
            self._play_opponent_turn()
            self._last_obs = self.base_env._get_obs()
        return self._last_obs, info

    def step(self, action):
        # 1) 에이전트 턴
        obs, reward_agent, terminated, truncated, info = self.base_env.step(action)
        if terminated or truncated:
            return obs, reward_agent, terminated, truncated, info

        # 2) 상대 턴(PPO)
        # 함수의 반환값을 opp_reward 변수에 저장합니다.
        opp_reward = self._play_opponent_turn()

        # 3) 종료/패널티 보정 및 다음 관측 반환 (다시 에이전트 차례)
        terminated_now, loser_penalty = self._check_terminal_and_penalty_after_opponent()
        total_reward = (reward_agent - opp_reward) + loser_penalty
        self._last_obs = self.base_env._get_obs()
        return self._last_obs, total_reward, terminated_now, False, info

    # ===== 내부 세팅 =====
    def _play_opponent_turn(self):
        if self.base_env.current_player == self.agent_side:
            return 0.0  # 상대 턴이 아니면 보상 0 반환
        opp_obs = self.base_env._get_obs()
        obs_shape = getattr(self.opponent.observation_space, 'shape', (25,))
        expected_dim = obs_shape[0] if obs_shape else 25
        obs_to_use = opp_obs[:24] if expected_dim == 24 and len(opp_obs) == 25 else opp_obs
        opp_action, _ = self.opponent.predict(obs_to_use, deterministic=True)
        # 상대방의 step 결과에서 reward 값을 받아옴
        _obs, opp_reward, _terminated, _truncated, _info = self.base_env.step(opp_action)
        return opp_reward
    
    def set_opponent(self, new_opponent_model: PPO):
        self.opponent = new_opponent_model

    def _check_terminal_and_penalty_after_opponent(self):
        current_black = sum(1 for s in self.base_env.stones if s.color[:3] == (0, 0, 0))
        current_white = sum(1 for s in self.base_env.stones if s.color[:3] == (255, 255, 255))
        if current_black == 0 and current_white > 0:
            winner = "white"; terminated = True
        elif current_white == 0 and current_black > 0:
            winner = "black"; terminated = True
        else:
            return False, 0.0

        # 에이전트가 진 경우에만 추가 패널티
        if (winner == "white" and self.agent_side == "black") or \
           (winner == "black" and self.agent_side == "white"):
            return True, -5.0
        return True, 0.0
    

def make_vs_c_env_vec(n_envs: int = 2):
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsModelCEnv(agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])


def make_vs_opponent_env_vec(opponent_model: PPO, n_envs: int = 2):
    def _maker(i):
        side = "black" if i % 2 == 0 else "white"
        return lambda: VsFixedOpponentEnv(opponent_model=opponent_model, agent_side=side)
    return DummyVecEnv([_maker(i) for i in range(n_envs)])

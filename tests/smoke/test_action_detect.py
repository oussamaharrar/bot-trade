from bot_trade.env.action_space import detect_action_space
import gymnasium as gym


def test_box_detection():
    env = gym.make('Pendulum-v1')
    info = detect_action_space(env)
    assert not info.is_discrete and info.shape[0] == env.action_space.shape[0]


import gymnasium as gym

from bot_trade.env.action_space import detect_action_space as legacy_detect
from bot_trade.env.space_detect import detect_action_space


def test_box_detection_and_shim():
    env = gym.make("Pendulum-v1")
    info = detect_action_space(env)
    legacy_env = legacy_detect(env)
    legacy_space = legacy_detect(env.action_space)
    assert info == legacy_env == legacy_space
    assert not info.is_discrete
    assert info.shape == env.action_space.shape
    assert info.low == float(env.action_space.low.min())
    assert info.high == float(env.action_space.high.max())
    env.close()


def test_discrete_detection():
    env = gym.make("CartPole-v1")
    info = detect_action_space(env)
    assert info.is_discrete
    assert info.shape == (1,)
    assert info.low == 0.0
    assert info.high == env.action_space.n - 1
    assert legacy_detect(env) == info
    env.close()

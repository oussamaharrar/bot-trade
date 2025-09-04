import pytest
from gymnasium import spaces
from bot_trade import rl_builders


def test_builders_box():
    algos = ["PPO", "SAC", "TD3"]
    try:
        from stable_baselines3 import TQC  # type: ignore
        algos.append("TQC")
    except Exception:
        pass
    for algo in algos:
        model, env, cbs = rl_builders.build_agent(algo, {})
        assert isinstance(env.action_space, spaces.Box)


def test_require_box_guard():
    with pytest.raises(TypeError):
        rl_builders._require_box(spaces.Discrete(2))

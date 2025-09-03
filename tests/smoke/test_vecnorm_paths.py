from bot_trade.config.rl_paths import vecnorm_path


def test_vecnorm_paths_scoped(tmp_path):
    ppo = vecnorm_path('BTCUSDT', '1m', 'PPO', 'run1')
    sac = vecnorm_path('BTCUSDT', '1m', 'SAC', 'run2')
    assert 'PPO' in str(ppo)
    assert 'SAC' in str(sac)
    assert ppo != sac


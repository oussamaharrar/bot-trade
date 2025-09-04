from bot_trade.tools.results_watcher import parse_log_line


def test_results_events_parse():
    lines = [
        "[DEBUG_EXPORT] reward_rows=1 step_rows=2 train_rows=3 risk_rows=4 callbacks_rows=5 signals_rows=6",
        "[CHARTS] dir=/tmp images=5",
        "[POSTRUN] run_id=abcd symbol=BTC frame=1m algorithm=PPO",
    ]
    events = [parse_log_line(l) for l in lines]
    assert events[0]["event"] == "debug_export" and events[0]["reward_rows"] == 1
    assert events[1]["event"] == "charts" and events[1]["images"] == 5
    assert events[2]["event"] == "postrun" and events[2]["run_id"] == "abcd"

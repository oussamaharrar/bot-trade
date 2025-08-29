# Bot Architecture Map

## High-level Flow

```
[CLI] -> [Config] -> [Data Loading] -> [Env] -> [Agent] -> [Writers] -> [Memory] -> [Reports/Results] -> [Monitors]
```

## Module Inventory

- ai_core.__init__
- ai_core.ai_dashboard
- ai_core.convert
- ai_core.dashboard_io
- ai_core.entry_verifier
- ai_core.portfolio
- ai_core.self_improver
- ai_core.simulation_engine
- ai_core.strategy_learner
- config.__init__
- config.env_config
- config.env_trading
- config.evaluate
- config.evaluate_all_agents
- config.evaluate_enhanced
- config.loader
- config.log_setup
- config.market_data
- config.results_logger
- config.risk_manager
- config.rl_args
- config.rl_builders
- config.rl_callbacks
- config.rl_paths
- config.rl_writers
- config.signals.danger_signals
- config.signals.entry_signals
- config.signals.freeze_signals
- config.signals.recovery_signals
- config.signals.reward_signals
- config.signals.signals_bridge
- config.strategy_features
- config.update_manager
- tools.__init__
- tools.analytics_common
- tools.analyze_risk_and_training
- tools.analyze_sessions
- tools.anomaly_watch
- tools.apply_config_proposal
- tools.binance_klines_feather
- tools.bootstrap
- tools.cli_console
- tools.data_aggregator
- tools.export_charts
- tools.fast
- tools.generate_dev_map
- tools.generate_markdown_report
- tools.knowledge_hub
- tools.knowledge_sync
- tools.live_ticker
- tools.make_config_override
- tools.memory_cli
- tools.memory_index
- tools.memory_manager
- tools.memory_store
- tools.merge_feather_files
- tools.monitor_launch
- tools.monitor_manager
- tools.paths
- tools.resource_monitor
- tools.run_state
- tools.runctx
- tools.self_improver
- tools.session_learning_engine
- tools.tuning_engine

## Runtime Flow

1. Parse CLI args
2. Load configuration
3. Build data loaders and env
4. Instantiate agent
5. Train/evaluate with writers
6. Persist memory and knowledge
7. Generate reports and launch monitors

## Inputs/Outputs

- configs: config/*.py
- data: results/{symbol}/{frame}/
- reports: report/{symbol}/{frame}/{run_id}/
- logs: logs/{symbol}/{frame}/{run_id}/

## Environment Variables

- MONITOR_SHELL, MONITOR_DEBUG, MONITOR_USE_CONDA_RUN

## Adding new tools

Place new scripts under tools/ and ensure absolute imports.

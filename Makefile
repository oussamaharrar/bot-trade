.PHONY: train evaluate evaluate-full rl-train rl-run run dashboard install smoke sweep longrun docs clean

install:
	pip install -r requirements.txt
IMAGE_NAME ?= bot-trade

train:
	python autolearn.py

evaluate:
	python evaluate_model.py

evaluate-full:
	python evaluate_model_metrics.py

rl-train:
	python train_rl.py

rl-run:
	python run_rl_agent.py

train-rl:
	python train_rl.py

run:
	python run_bot.py

dashboard:
	streamlit run dashboard.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -it $(IMAGE_NAME)

devmap:
	python -m tools.generate_dev_map > docs/bot_map.md


smoke:
	python -m py_compile $$(git ls-files 'bot_trade/**/*.py' 'bot_trade/*.py'); \
	python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready; \
	python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --total-steps 128 --headless --allow-synth --data-dir data_ready --no-monitor; \
	python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet; \
	python -m bot_trade.tools.mk_artifacts_index --symbol BTCUSDT --frame 1m --run-id latest --algorithm PPO

sweep:
	python -m bot_trade.tools.sweep --mode random --n-trials 4 \
	  --symbol BTCUSDT --frame 1m --algorithm SAC --continuous-env \
	  --headless --allow-synth --data-dir data_ready

longrun:
	python -m bot_trade.tools.train_long_run --algorithm SAC --continuous-env \
	  --symbol BTCUSDT --frame 1m --headless --allow-synth --data-dir data_ready \
	  --preset training=sac_cont_cpu_smoke --preset net=tiny \
	  --save-every 128 --eval-every 128 --max-steps 256

docs:
	@ls docs/REGISTRIES.md docs/CLI_HELP.md docs/KB_SCHEMA.md docs/RELEASE.md docs/QUICKSTART.md

clean:
	find reports -name '*.tmp' -delete || true

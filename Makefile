.PHONY: train evaluate evaluate-full rl-train rl-run run dashboard install

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


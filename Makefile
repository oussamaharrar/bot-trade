IMAGE_NAME ?= bot-trade

train:
	python autolearn.py

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


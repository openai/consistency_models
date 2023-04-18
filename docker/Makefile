NAME=consistency_models
TAG=0.1
PROJECT_DIRECTORY = $(shell pwd)/..

build:
	docker build -t ${NAME}:${TAG} -f Dockerfile .

run:
	docker container run --gpus all\
		--restart=always\
		-it -d \
		-v $(PROJECT_DIRECTORY):/home/${NAME}\
		--name ${NAME} ${NAME}:${TAG} /bin/bash

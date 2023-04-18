NAME=consistency_models
TAG=0.1
PROJECT_DIRECTORY = $(shell pwd)/..

build:
	docker build -t ${NAME}:${TAG} -f Dockerfile .

run:
	@$(MAKE) container-stop-rm;\
	docker container run --gpus all\
		--restart=always\
		-it -d \
		-v $(PROJECT_DIRECTORY):/home/${NAME}\
		--name ${NAME} ${NAME}:${TAG} /bin/bash

container-stop-rm:
	@if [ `docker container ls -a --filter "name=${NAME}" | wc -l | sed "s/ //g"` -eq 2 ]; then\
		docker container stop ${NAME};\
		docker container rm ${NAME};\
		echo -e "\x1b[32mcontainer successfully stopped and removed.\x1b[m";\
	fi

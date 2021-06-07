# Dockerfile for SAG (simple adversarial generator) for VNNCOMP

FROM python:3.6

# set environment variable
ENV PYTHONPATH=$PYTHONPATH:/sag/src

# copy current directory to docker
COPY . /sag

WORKDIR /sag

# install other (required) dependencies
RUN pip3 install -r requirements.txt 

### As default command: run the tests ###
CMD cd /sag/tests && ./run_tests.sh

# USAGE:
# Build container and name it 'sag':
# docker build . -t sag

# # run tests (default command)
# docker run sag

# # get a shell:
# docker run -it hylaa bash
# hylaa is available in /hylaa
# to delete docker container use: docker rm hylaa

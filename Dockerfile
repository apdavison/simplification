#
# A Docker image for model simplification
#
# To build:
#     docker build -t simplify .
#
# To run, follow the instructions at
#     https://github.com/NeuralEnsemble/neuralensemble-docker/blob/master/simulationx/README.md


FROM neuralensemble/simulationx
MAINTAINER andrew.davison@unic.cnrs-gif.fr

USER docker

RUN $VENV/bin/pip install jupyter efel
RUN $VENV/bin/pip install git+http://github.com/BlueBrain/deap/
RUN $VENV/bin/pip install bluepyopt

RUN sed 's/#force_color_prompt/force_color_prompt/' .bashrc > tmp; mv tmp .bashrc
RUN echo "source /home/docker/env/neurosci/bin/activate" >> .bashrc

RUN mkdir /home/docker/projects

USER root

RUN apt-get update
RUN apt-get install -y iceweasel

EXPOSE 8888

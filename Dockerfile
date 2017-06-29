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

USER root

RUN apt-get update
RUN apt-get install -y iceweasel python-pandas

USER docker

RUN $VENV/bin/pip install jupyter efel
RUN $VENV/bin/pip install git+http://github.com/BlueBrain/deap/

WORKDIR /home/docker/packages
RUN git clone https://github.com/apdavison/BluePyOpt.git
RUN cd BluePyOpt; git checkout pynn-models

RUN $VENV/bin/pip install -e /home/docker/packages/BluePyOpt

WORKDIR $HOME
RUN sed 's/#force_color_prompt/force_color_prompt/' .bashrc > tmp; mv tmp .bashrc
RUN echo "source /home/docker/env/neurosci/bin/activate" >> .bashrc

RUN mkdir /home/docker/projects

USER root
EXPOSE 8888

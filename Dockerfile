ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer='UC San Diego ITS/ETS <ets-consult@ucsd.edu>'

USER root 

RUN apt-get -y install htop tmux

# USER jovyan

# RUN pip install --no-cache-dir statsmodels networkx xgboost lightgbm cvxopt gensim planarity autopep8 stellargraph


# The build-stage image:
# FROM continuumio/miniconda3 AS build

# # # Install the package as normal:
COPY environment-test.yml .
RUN conda env create -f environment-test.yml 

# # ENTRYPOINT ["conda", "run", "-n", "finance-base", \
# #     "python", "-c", \
#     # "import numpy; print('success!')"]
# # RUN conda activate finance-base
# SHELL ["conda","run","-n","finance-base","/bin/bash","-c"]

# USER root 
# RUN apt-get install -y htop tmux
# # Install conda-pack:
# RUN conda install -c conda-forge conda-pack


# # in /venv:
# RUN conda-pack -n finance-base -o /tmp/env.tar && \
#     mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
#     rm /tmp/env.tar

# # We've put venv in same path it'll be in final image,
# # so now fix up paths:
# RUN /venv/bin/conda-unpack


# # The runtime-stage image; we can use Debian as the
# # base image since the Conda env also includes Python
# # for us.
# FROM debian:buster AS runtime

# # Copy /venv from the previous stage:
# COPY --from=build /venv /venv

# # When image is run, run the code with the environment
# # activated:
# SHELL ["/bin/bash", "-c"]
# ENTRYPOINT source /venv/bin/activate && \
#     python -c "import numpy; print('success!')"


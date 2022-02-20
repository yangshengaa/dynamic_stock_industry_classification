ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer='UC San Diego ITS/ETS <ets-consult@ucsd.edu>'

USER root 

RUN apt-get -y install htop

COPY environment.yml .
RUN conda env create -f environment.yml 

FROM jupyter/base-notebook:45b8529a6bfc

COPY . $HOME
USER root
RUN chown -R $NB_UID $HOME
USER $NB_UID

RUN conda env update -n root -f environment.yml

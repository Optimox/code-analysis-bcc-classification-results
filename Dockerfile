FROM python:3.10-slim

RUN apt update && apt install curl make git libgtk2.0-dev -y

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV SHELL /bin/bash -l

ENV POETRY_CACHE /work/.cache/poetry

ENV PIP_CACHE_DIR /work/.cache/pip

ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime

ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN /root/.local/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH /root/.local/bin:/bin:/usr/local/bin:/usr/bin

RUN export PATH="/root/.local/bin:$PATH"

CMD ["bash", "-l"]

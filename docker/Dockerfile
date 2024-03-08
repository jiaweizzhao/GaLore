
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    curl \
    git \
    build-essential \
    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip python3.11-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

WORKDIR /code
ENV VIRTUAL_ENV=/code/.venv

RUN pip install poetry==1.6.1

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry config virtualenvs.create true
RUN poetry config virtualenvs.in-project true
RUN poetry env use python3.11

RUN poetry install --no-interaction --no-ansi --no-root

COPY ./galore_torch ./galore_torch

RUN poetry install --no-interaction --no-ansi

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

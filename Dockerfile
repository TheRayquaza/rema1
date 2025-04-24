FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONPATH="/home/default/.local/lib/python3/site-packages/"

RUN apt-get update --no-install-recommends \
    && apt-get install --no-install-recommends -y \
    python3 git make cron bzip2 tzdata locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && cp /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && useradd --create-home --shell /bin/bash default

WORKDIR /home/default
COPY requirements.txt ./
RUN chown default:default requirements.txt

USER default
RUN python3 -m venv .venv \
    && . .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install pipenv \
    && pip install -r requirements.txt

USER root
COPY --chown=default:default ./src ./src
USER default

CMD ["python3", "src/__main__.py"]

FROM python:3

ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_DIR=/models
ENV TRITON_HOST=triton_server
ENV TRITON_PORT=8000

RUN useradd --create-home config_daemon
USER config_daemon
WORKDIR /home/config_daemon
ENV PATH="/home/config_daemon/.local/bin:${PATH}"

COPY --chown=config_daemon:config_daemon ./ ./

RUN pip install --user --no-cache-dir .

CMD uvicorn config_daemon.api:app --host ${HOST} --port ${PORT}
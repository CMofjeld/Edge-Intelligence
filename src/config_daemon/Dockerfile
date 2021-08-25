FROM python:3

ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_DIR=/models
ENV TRITON_HOST=triton_server
ENV TRITON_PORT=8000

WORKDIR /app

ADD ./requirements.txt ./requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

ADD ./ ./

CMD uvicorn api:app --host ${HOST} --port ${PORT}
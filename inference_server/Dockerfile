# Reference: https://github.com/triton-inference-server/server/issues/1468#issuecomment-829522264
FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        autoconf \
        automake \
        build-essential \
        cmake \
        git \
        libb64-dev \
        libre2-dev \
        libssl-dev \
        libtool \
        libboost-dev \
        libcurl4-openssl-dev \
        libopenblas-dev \
        rapidjson-dev \
        patchelf \
        zlib1g-dev

WORKDIR /tritonserver

RUN wget https://github.com/triton-inference-server/server/releases/download/v2.12.0/tritonserver2.12.0-jetpack4.6.tgz && \
    tar -xzf tritonserver2.12.0-jetpack4.6.tgz && \
    rm tritonserver2.12.0-jetpack4.6.tgz

ENTRYPOINT ["/tritonserver/bin/tritonserver"]
CMD ["--model-repository", "/models", "--backend-directory", "/tritonserver/backends", "--backend-config", "tensorflow,version=2", "--strict-model-config", "false"]

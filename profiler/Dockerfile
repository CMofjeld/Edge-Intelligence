FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

WORKDIR /usr/src/app

RUN pip3 install --no-cache-dir scipy
RUN pip3 install --no-cache-dir scikit-learn

ADD ./profiler/models ./models
COPY ./profiler/profiler.py ./

CMD [ "python", "profiler.py" ]
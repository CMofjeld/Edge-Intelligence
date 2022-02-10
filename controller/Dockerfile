# Use base image with IPOPT installed
FROM coinor/coin-or-optimization-suite

# Install AMPL
RUN cd /opt/ && \
    wget https://ampl.com/demo/ampl.linux64.tgz && \
    tar xzf ampl.linux64.tgz && \
    rm ampl.linux64.tgz

# Install Python and pip
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip setuptools wheel

# Copy app files over
ADD ./ /usr/src/app
WORKDIR /usr/src/app

# Install controller package
RUN python3 -m pip install .
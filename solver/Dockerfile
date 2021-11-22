# Use base image with IPOPT installed
FROM coinor/coin-or-optimization-suite

# Install AMPL
RUN cd /opt/ && \
    wget https://ampl.com/demo/ampl.linux64.tgz && \
    tar xzf ampl.linux64.tgz && \
    rm ampl.linux64.tgz

# Install Python
RUN apt-get update -y && \
    apt-get install -y python3

# Copy app files over
ADD ./solver /usr/src/app
WORKDIR /usr/src/app
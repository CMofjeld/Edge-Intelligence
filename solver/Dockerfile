# Use base image with IPOPT installed
FROM coinor/coin-or-optimization-suite

# Install AMPL
RUN cd /opt/ && \
    wget https://ampl.com/demo/ampl.linux64.tgz && \
    tar xzf ampl.linux64.tgz && \
    rm ampl.linux64.tgz
COPY ./ampl.sh /bin/ampl.sh
RUN chmod 777 /bin/ampl.sh
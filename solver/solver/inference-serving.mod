##  Model file for Inference Serving problem
option solver bonmin;

##  Sets
set R;  # set of requests
set M;  # set of DNN models
set N;  # set of edge nodes
set SERVED within {M,N};    # edge node model deployments

##  Parameters
param lamda{i in R};        # arrival rate of request i
param tau{i in R};          # transmission speed of request i
param accuracy{i in M};     # accuracy score of model i
param min_accuracy{i in R}; # minimum accuracy score accepted by request i
param input_size{i in M};   # input size in bits for model i
param max_throughput{SERVED};   # maximum throughput of model i on edge node j
param alpha{SERVED};    # batching coefficient for model i on edge node j
param beta{SERVED};     # batching coefficient for model i on edge node j

##  Outputs
# Indicator variable I:
#   1 indicates that request i is served with model j on edge node k
#   and 0 indicates that it is not
var I{i in R, (j, k) in SERVED} binary;

##  Objective
# Uses Inoue's formula to estimate serving latency
maximize reward:
    sum{i in R, (j,k) in SERVED} (I[i,j,k]*accuracy[j] / (input_size[j]*tau[i]
        + (3/2) * (beta[j,k] / (1 - (sum{l in R}(I[l,j,k]*lamda[l])) * alpha[j,k]))
        + (alpha[j,k]/2) * ((sum{l in R}(I[l,j,k]*lamda[l]))*alpha[j,k] + 2) / (1 - ((sum{l in R}(I[l,j,k]*lamda[l]))^2)*(alpha[j,k]^2))));

## Constraints
subject to UniqueDestination{i in R}:
    sum{(j,k) in SERVED}(I[i,j,k]) = 1;
subject to Accuracy{i in R}:
    sum{(j,k) in SERVED}(I[i,j,k]*accuracy[j]) >= min_accuracy[i];
subject to MaxThroughput{(j,k) in SERVED}:
    sum{i in R}(I[i,j,k]*lamda[i]) <= max_throughput[j,k];
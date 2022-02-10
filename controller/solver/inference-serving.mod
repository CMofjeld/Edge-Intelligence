##  Model file for Inference Serving problem
option solver bonmin;

##  Sets
set R;  # set of requests
set M;  # set of DNN models
set N;  # set of edge nodes
set SERVED within {M,N};    # edge node model deployments

##  Parameters
param gamma;                # parameter that weights latency vs error rate in cost function
param lambda{i in R};       # arrival rate of request i
param tau{i in R};          # transmission speed of request i
param prop_delay{i in R};   # propagation delay of request i
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
var Cost{i in R};
var ServerArrivalRate{(j, k) in SERVED};
var ModelLatency{(j,k) in SERVED};
var ServingLatency{k in N};

##  Objective
minimize total_cost:
    sum{i in R} Cost[i];

## Constraints
subject to cost{i in R}:
    Cost[i] = sum{(j,k) in SERVED}(I[i,j,k]*(1-gamma)*((1 - accuracy[j])^2 + gamma*(input_size[j]/tau[i] + prop_delay[i] + ServingLatency[k])^2));
subject to UniqueDestination{i in R}:
    sum{(j,k) in SERVED}(I[i,j,k]) = 1;
subject to Accuracy{i in R}:
    sum{(j,k) in SERVED}(I[i,j,k]*accuracy[j]) >= min_accuracy[i];
subject to MaxThroughput{k in N}:
    sum{i in R, j in M}(I[i,j,k]*lambda[i]/max_throughput[j,k]) <= 1;
subject to server_arrival_rate{(j,k) in SERVED}:
    ServerArrivalRate[j,k] = sum{i in R}(I[i,j,k]*lambda[i]);
subject to model_latency {(j,k) in SERVED}:
    ModelLatency[j,k] = (3/2) * (beta[j,k] / (1 - ServerArrivalRate[j,k] * alpha[j,k]))
        + (alpha[j,k]/2) * ((ServerArrivalRate[j,k])*alpha[j,k] + 2) / (1 - ((ServerArrivalRate[j,k])^2)*(alpha[j,k]^2));
subject to serving_latency {k in N}:
    ServingLatency[k] = sum{j in };
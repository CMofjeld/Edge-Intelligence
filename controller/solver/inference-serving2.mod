##  Model file for Inference Serving problem
option solver bonmin;

##  Sets
set R;  # set of requests
set M;  # set of DNN models
set N;  # set of edge nodes
set SERVED {N};    # edge node model deployments

##  Parameters
param gamma;                # parameter that weights latency vs error rate in cost function
param lambda{i in R};       # arrival rate of request i
param tau{i in R};          # transmission speed of request i
param prop_delay{i in R};   # propagation delay of request i
param accuracy{i in M};     # accuracy score of model i
param min_accuracy{i in R}; # minimum accuracy score accepted by request i
param input_size{i in M};   # input size in bits for model i
param max_throughput{ k in N, j in SERVED[k]};   # maximum throughput of model i on edge node j
param alpha{k in N, j in SERVED[k]};    # batching coefficient for model i on edge node j
param beta{k in N, j in SERVED[k]};     # batching coefficient for model i on edge node j

##  Outputs
# Indicator variable I:
#   1 indicates that request i is served with model j on edge node k
#   and 0 indicates that it is not
var I{i in R, k in N, j in SERVED[k]} binary;
var Cost{i in R};
var ServerArrivalRate{k in N, j in SERVED[k]};
var ModelLatency{k in N, j in SERVED[k]};
var ServingLatency{k in N};

##  Objective
minimize total_cost:
    sum{i in R} Cost[i];

## Constraints
subject to cost{i in R}:
    Cost[i] = sum{k in N, j in SERVED[k]}(I[i,k,j]*((1-gamma)*(1 - accuracy[j])^2 + gamma*(input_size[j]/tau[i] + prop_delay[i] + ServingLatency[k])^2));
subject to UniqueDestination{i in R}:
    sum{k in N, j in SERVED[k]}(I[i,k,j]) = 1;
subject to Accuracy{i in R}:
    sum{k in N, j in SERVED[k]}(I[i,k,j]*accuracy[j]) >= min_accuracy[i];
subject to MaxThroughput{k in N}:
    sum{i in R, j in SERVED[k]}(I[i,k,j]*lambda[i]/max_throughput[k,j]) <= 1;
subject to server_arrival_rate{k in N, j in SERVED[k]}:
    ServerArrivalRate[k,j] = sum{i in R}(I[i,k,j]*lambda[i]);
subject to model_latency {k in N, j in SERVED[k]}:
    ModelLatency[k,j] = min(ServerArrivalRate[k,j] * 1000,
        (3/2) * (beta[k,j] / (1 - ServerArrivalRate[k,j] * alpha[k,j]))
        + (alpha[k,j]/2) * ((ServerArrivalRate[k,j])*alpha[k,j] + 2) / (1 - ((ServerArrivalRate[k,j])^2)*(alpha[k,j]^2)),
        (alpha[k,j] + beta[k,j]) / (2 * (1 - ServerArrivalRate[k,j] * alpha[k,j]))
        * (1 + 2 * ServerArrivalRate[k,j] * beta[k,j] + (1 - ServerArrivalRate[k,j] * beta[k,j]) / (1 + ServerArrivalRate[k,j] * alpha[k,j]))
    );
subject to serving_latency {k in N}:
    ServingLatency[k] = sum{j in SERVED[k]}(ModelLatency[k,j]);
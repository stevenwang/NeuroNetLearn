# Multi-Experiment Network Estimation and Inference for High-Dimensional Point Process Data

Xu Wang, Mladen Kolar and Ali Shojaie. "Statistical Inference for Networks of High-Dimensional Point Processes." https://arxiv.org/abs/2007.07448

Xu Wang and Ali Shojaie. "Joint Estimation and Inference for Multi-ExperimentNetworks of High-Dimensional Point Processes." [forthcoming]


## Installation:
Require install `scipy`, `numpy`, `math`, `numba`

Code was run using python 3.8.5

## Primary files:
* `net_est_auto.py`: joint estimation for multi-experiment networks where tuning parameters are automatically chosen based on eBIC (via `getBIC.py`)
* `net_inf.py`: high-dimensional statistical inference for point process network
* `spg_genlasso_solver_jit.py`: generalized lasso solver using smoothing proximal gradient descent algorithm
* `simu_net.py`: simulate multi-experiment point process data where the settings of networks are generated using `genSetting.py`

## Example
* `examples/demo_network_estimation.html`: examples of estimating multi/single-experiment point process network(s) using `net_est` function in `net_est_auto.py`
* `examples/demo_network_inference.html`: examples of statistical inference over multi/single-experiment point process network(s) using `net_inf` function in `net_inf.py`



# Multi-Experiment Network Estimation and Inference for High-Dimensional Point Process Data

[![PyPI version](https://badge.fury.io/py/neuronetlearn.svg)](https://pypi.org/project/neuronetlearn/)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg



**Python package:** https://github.com/stevenwang/NeuroNetLearn/


Modern high-dimensional point process data, especially those from neuroscience experiments, often involve observations from multiple conditions and/or experiments. Networks of interactions corresponding to these conditions are expected to share many edges, but also exhibit unique, condition-specific ones. However, the degree of similarity among the networks from different conditions is generally unknown. Existing approaches for multivariate point processes do not take these structures into account and do not provide inference for jointly estimated networks. To address these needs, we develop the `neuronetlearn` package that includes estimation and inference tools for networks of high-dimensional Hawkes processes over multiple experiments. Specifically, `neuronetlearn` includes functions that implement a joint estimation procedure for networks of high-dimensional point processes that incorporates easy-to-compute weights in order to data-adaptively encourage similarity between the estimated networks. It also includes functions that implement a powerful hierarchical multiple testing procedure for edges of all estimated networks, which takes into account the data-driven similarity structure of the multi-experiment networks. 

For more details, please see the accompanying manuscripts: ["Statistical Inference for Networks of High-Dimensional Point Processes."](https://arxiv.org/abs/2007.07448) by Xu Wang, Mladen Kolar and Ali Shojaie, and ["Joint Estimation and Inference for Multi-ExperimentNetworks of High-Dimensional Point Processes."](https://arxiv.org/abs/2109.11634) by Xu Wang and Ali Shojaie. 


## Installation:

You can install a stable release of `neuronetlearn` using `pip3` by running `python pip3 install neuronetlearn` from a Terminal window. 

Dependencies include `scipy`, `numpy`, `math`, `numba`, `igraph`, `sklearn`

Code was run using python 3.8.5

## Primary files:
* `net_est_auto.py`: joint estimation for multi-experiment networks where tuning parameters are automatically chosen based on eBIC (via `getBIC.py`)
* `net_inf.py`: high-dimensional statistical inference for point process network
* `net_inf_threshold.py`: fast version of 'net_inf' by first identifying sub-graphs and then applying 'net_inf' to each sub-graph
* `spg_genlasso_solver_jit.py`: generalized lasso solver using smoothing proximal gradient descent algorithm
* `simu_net.py`: simulate multi-experiment point process data where the settings of networks are generated using `genSetting.py`
* `ht.py`: multi-experiment hierarchical testing controlling FWER

## Example
We recommend starting with one of the following examples (hosted at the package github [page](https://github.com/stevenwang/NeuroNetLearn)) which demonstrate various features of the package.

* `examples/demo_network_estimation.ipynb`: examples of estimating multi/single-experiment point process network(s) using `net_est` function in `net_est_auto.py`
* `examples/demo_network_inference.ipynb`: examples of statistical inference over multi/single-experiment point process network(s) using `net_inf` function in `net_inf.py`



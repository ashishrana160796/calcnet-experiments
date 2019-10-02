# Experiment Description & Setup

In this section experiments related to comparing accuracy of NAC and NALU units. It contains a script that extensively generates models
trained on different binary or unary operations and saves them. Along with that it also stores the histories in runtime for analysis
with jupyter-notebook and output logs for debugging are also stored.

__Note:__ The experimental setup doesn't exactly match with the one used in paper. For replication of results remove the negative train targets from the generated data for improved results.

# Usage Instructions

* Simply, run the script with following command on your linux machine `python3 arithmatic_experiments.py`.
* Also, for running the jupyter notebook just start the jupyter server in the given directory and use the generated history variables to observe the plots for different models during different epoch trainings.

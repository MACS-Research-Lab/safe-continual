# On the Design of Safe Continual RL Methods for Control of Nonlinear Systems

Submitted to ECC 2025.

### Repository Structure

- [`results/`](./results/) contains `.csv` files for all results

- [`Safe-Policy-Optimization/`](./Safe-Policy-Optimization/) contains the code from [this respository](https://github.com/PKU-Alignment/Safe-Policy-Optimization).
    - The algorithms we used are inside the [`/safepo/single_agent/`](./Safe-Policy-Optimization/safepo/single_agent/) directory. These are [`cpo.py`](./Safe-Policy-Optimization/safepo/single_agent/cpo.py) (CPO), [`ppo_ewc_cost.py`](./Safe-Policy-Optimization/safepo/single_agent/ppo_ewc_cost.py) (Safe EWC), and [`ppo_ewc.py`](./Safe-Policy-Optimization/safepo/single_agent/ppo_ewc.py) (PPO+EWC). [`ppo_ewc_lambda.py`](./Safe-Policy-Optimization/safepo/single_agent/ppo_ewc_lambda.py) is used for tuning the $\lambda$ hyperparameter.

- ['safety-gymnasium/'](./safety-gymnasium/) contains the code from [this repository](https://github.com/PKU-Alignment/safety-gymnasium).
    - The continual RL environments we created that are used in the paper are in ['/safety_gymnasium/tasks/safe_velocity/'](./safety-gymnasium/safety_gymnasium/tasks/safe_velocity/). Specifically, the [`safety_half_cheetah_valocity_v4.py`](./safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_half_cheetah_velocity_v4.py) is the HalfCheetah nonstationary safety constrained task and [`safety_ant_velocity_v2.py`](./safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_ant_velocity_v2.py) is the Ant.

- [`Analyze Results.ipynb`](./Analyze%20Results.ipynb) contains the analysis of the results.

- [`Lambda Experiment.ipynb`](./Lambda%20Experiment.ipynb) contains a hyperparameter experiment to choose EWC $\lambda$.

- [`Environment Test`](./Environment%20Test.ipynb) can be used to test the environments and visualize them.


### Running Your Own Experiments

1. Enter the `/Safe-Policy-Optimization/safepo/single_agent/` directory. (e.g., `cd /Safe-Policy-Optimization/safepo/single_agent/`)
2. Train an agent by running the chosen algorithm as follows
    - `python algorithm.py --task taskname -- experiment experiment_name`
        - `algorithm` is one of `cpo`, `ppo_ewc`, `ppo_ewc_cost`, or `ppo_ewc_lambda`.
        - `taskname` is `SafeHalfCheetahVelocity-v4` or `SafeAntVelocity-v2`.
        - `experiment` is your experiment name which will be saved in the `runs/` folder. 
        - `--ewc_lambda num` will set the value of $\lambda$, the tradeoff between remembering previous tasks and learning on old tasks to `num`.
        - `--task-length num` is the number of environment observations for each nonstationary task.
        - `--tasks 'task_list'` is the task sequence. Ex: '[0, 1, 0, 1, 2, 0]'.
    - For a comprehensive list of command line arguments, check the `single_agent_args()` function in [this file](./Safe-Policy-Optimization/safepo/utils/config.py).


### Results of the Paper

The results of the paper can be reproduced by running the above commands for seeds 0-4 for each. As detailed in the paper, use `ewc_lambda=10`, `task-length=1_000_000`, `task_list='[0, 1, 0, 2, 1, 0, 2]'`, and `total-steps=8_000_000`. These results are saved in the `results/` directory. Use `Analyze Results.ipynb` to see our analysis. 

### Environment Setup



1. From project top directory `conda env create -f environment.yml`
2. `conda activate safe-continual` 
3. `cd safety-gymnasium`
4. `pip install -e .`

If you experience any issues, you may need to setup your own conda env and install safety gymnasium, then add packages as necessary. Alternatively, if your installation is not time-sensitive, please feel free to raise an issue!
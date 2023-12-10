# Quadjax: High performance quadrotor simulator with wide support of RL and control algorithms

## Supported algorithms

* RL
  * PPO + Auto curriculum + Domain randomization
  * [RMA](https://arxiv.org/abs/2106.00091) (Rapid Motor Adaptation)
  * [DATT](https://arxiv.org/abs/2310.09053) (Deep Adaptive Trajectory Tracking)
* Control
  * PID, L1 Adaptive Control
  * MPPI (model predictive path integral)
  * 🌟 [CoVO-MPC](https://panchaoyi.com/covo-mpc-theoretical-analysis-of-sampling-based-mpc-and-optimal-covariance-design) (optimal covariance model predictive control)

## Installation

```shell
conda create -n jax python
conda activate jax
pip install -e .
```

## Run `CoVO-MPC`

Run `CoVO-MPC` in quadrotor environment (if you want to run other controllers, just replace `covo-online` with `pid`, `l1`, `mppi`, `nn`, `RMA` etc. you can also change the task to `tracking` or `hover`): 

```shell
cd quadjax/envs
# note: --noDR means no domain randomization, disable it if running a controller
python quadrotor.py --controller covo-online --task tracking_zigzag --mode render --disturb_type none --noDR 
# run CoVO-MPC offline approximation version
python quadrotor.py --controller covo-offline --task tracking_zigzag --mode render --disturb_type none --noDR 
```

This will generate tested state sequence `state_seq.pkl` and plot figure `plot.png`.

Reproduce the results in the `CoVO-MPC` paper: 

```shell
cd quadjax/scripts
# main results
sh covo_quadrotor.sh
# ablation study for sampling number
sh covo_quadrotor_N.sh
```

## Visualization

```shell
cd quadjax/scripts
python vis.py
```

This will visualize the results in `quadjax/results/state_seq_.pkl` with meshcat, which is generated by `quadjax/envs/quadrotor.py`.

## Train policy

Train `PPO`:
```shell
cd quadjax
python train.py
```

Train `RMA`: 
```shell
cd quadjax
python train.py --RMA
```

Train `DATT`: 
```shell
cd quadjax
python train.py --lower-controller l1_esitimate_only --obs-type quad_l1 --disturb-type periodic --task tracking_zigzag
```

Training debug

```
python train.py --env quad3d_free --debug
```

## Notes

* all action in the environment is normalized into [-1, 1] range.
* for maximum reusability, the environment is designed to be simple single file format.
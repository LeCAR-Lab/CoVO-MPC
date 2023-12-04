# Learning-based Quadrotor Simulator and Controller in Jax

## Code snips

### Quadrotor

Play with the environment

```python
cd quadjax/envs
python quad3d_free.py --dynamics bodyrate --noDR --controller pid --task tracking_zigzag --mode render --disturb_type none
```

This will generate tested state sequence `state_seq.pkl` and plot figure `plot.png`.

Visualize 3d environment:

```python
cd quadjax/test
python vis.py # please make sure you have state_seq.pkl inside the results. 
```

Trajectory tracking with max velocity 2m/s with 3d quadrotor environment.

```
# tracking zigzag
python train.py --env quad3d_free --dynamics bodyrate --noDR --name test

# tracking smoooth trajectory
python train.py --env quad3d_free --task tracking
```

train RMA

```shell
# RMA
python train.py --env quad3d_free --dynamics bodyrate --RMA
# robust PPO
python train.py --env quad3d_free --dynamics bodyrate
# expert PPO (PPO with true parameter)
python train.py --env quad3d_free --dynamics bodyrate --obs_type quad_params
```

run certain controller

```shell
# L1 controller for quadrotor 3d free environment (NOTE: L1 currently only works for bodyrate dynamics)
python quad3d_free.py --dynamics bodyrate --noDR --controller pid --task tracking_zigzag --mode render --disturb_type none
```

![ppo](https://github.com/jc-bao/quadjax/assets/60093981/48220814-8775-4539-b9bc-85f6236b077b)

https://github.com/jc-bao/quadjax/assets/60093981/6f06ab1a-df00-4d8b-8aa1-56008298f0ab

Training debug

```
python train.py --env quad3d_free --debug
```

### modify environment

**add disturbance**

all the environment dynamics is inside `quadjax/dynamics` folder. For `Quad3D` quadrotor only environment, the dynamics is `free.py` with function `get_free_dynamics_3d()`. Please note the coordinate system is in the quadrotor local frame. 

## Notes

* all action in the environment is normalized into [-1, 1] range.
* for maximum reusability, the environment is designed to be simple single file format.
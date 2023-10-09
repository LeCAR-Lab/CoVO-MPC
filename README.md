# Learning-based Quadrotor Simulator and Controller in Jax

## Code snips

### Quadrotor

Play with the environment

```python
cd quadjax/envs
python quad3d_free.py --controller base --task hovering
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
python train.py --env quad3d_free --task tracking_zigzag

# tracking smoooth trajectory
python train.py --env quad3d_free --task tracking
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
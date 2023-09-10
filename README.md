# Learning-based Quadrotor Simulator and Controller in Jax

## Results

### 2D Quadrotor Tracking

Smooth trajectory with max acceleration 16m/s^2 and max velocity 5m/s.

```
python train.py --task tracking
```

![ppo](https://github.com/jc-bao/quadjax/assets/60093981/b9890897-9bcf-48d4-ba83-2ea794999218)


https://github.com/jc-bao/quadjax/assets/60093981/abb72f44-a227-4b04-a2bf-4d8f46e5a90c


Zig-zag trajectory with max acceleration 16m/s^2 and max velocity 3.12m/s.

```
python train.py --task tracking_zigzag
```

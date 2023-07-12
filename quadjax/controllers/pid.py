import jax
from jax import numpy as jnp
from functools import partial

from quadjax.quad3d import Quad3D

class PIDController:
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, kp, ki, kd, ki_max, integral, last_error):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ki_max = ki_max
        self.integral = integral
        self.last_error = last_error
        self.reset()

    def reset(self):
        self.integral *= 0.0
        self.last_error *= 0.0

    @partial(jax.jit, static_argnums=(0,))
    def update(self, error, dt):
        self.integral += error * dt
        self.integral = jnp.clip(self.integral, -self.ki_max, self.ki_max)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class Quad3DPID:
    """PID controller for 3d quadrotor environment
    """

    def __init__(self, env:Quad3D):
        self.quadpos_controller = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            ki_max=0.0,
            integral=0.0,
            last_error=0.0
        )
        self.quadatti_controller 

import numpy as np
import matplotlib.pyplot as plt


M = np.eye(3)
C = -0.5*np.eye(3)
g = - 2.0 * np.ones((3,))

N = 5000
T = 10.0
dt = T/N





class controller_params:
    a_hat: np.ndarray = np.zeros((6,), dtype=np.float32)
    v_hat: np.ndarray = np.zeros((3,), dtype=np.float32)
    last_u: np.ndarray = np.zeros((3,), dtype=np.float32)
    K: np.ndarray = 5.0 * np.eye(3, dtype=np.float32)
    P: np.ndarray = 5.0 * np.eye(6, dtype=np.float32)
    R: np.ndarray = 1.0 * np.eye(3, dtype=np.float32)
    nu: np.ndarray = 0.9 * np.ones((6,), dtype=np.float32)


# def phi(x):
#     v = x[3:6]
#     return ((- 1.0 * v * np.abs(v)).T - np.array([-1.0, 0.0, 1.0])).T

def phi(x):
    v = x[3:6]
    return np.concatenate([np.diag(v*np.abs(v)), np.eye(3)], axis=1)

def disturbance(x):
    return phi(x)@np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 1.0])

def dynamics(x, u):
    p = x[0:3]
    v = x[3:6]
    dot_v = np.linalg.inv(M) @ (u + g - C @ v + disturbance(x))
    # dot_v = np.linalg.inv(M) @ (u + g - C @ v)
    dot_p = v
    return np.concatenate([dot_p, dot_v])


def simulate(x0, u, N, dt):
    x_log = np.zeros((6, N), dtype=np.float32)
    ctrl_log = np.zeros((9, N), dtype=np.float32)

    ctrl_params = controller_params()

    x = x0
    t = 0.0
    for i in range(N):
        x_log[:, i] = x
        ctrl_log[0:3, i] = ctrl_params.v_hat
        ctrl_log[3:9, i] = ctrl_params.a_hat
        x = x + dt * dynamics(x, u(t, x, ctrl_params))
        t += dt
    return x_log, ctrl_log


def reference(t):
    ref = (np.diag([2.0, -1.0, 1.0]) @
           np.sin(np.array([[1.0, 1.5, 2.0]]).T*t)).squeeze()
    return ref


def reference_dot(t):
    ref = ((np.array([1.0, 1.5, 2.0]) * np.diag([2.0, -1.0, 1.0]))
           @ np.cos(np.array([[1.0, 1.5, 2.0]]).T*t)).squeeze()
    return ref


def controller(t, x, ctrl_params: controller_params):
    p = x[0:3]
    v = x[3:6]

    v_hat = ctrl_params.v_hat
    a_hat = ctrl_params.a_hat
    last_u = ctrl_params.last_u
    K = ctrl_params.K
    P = ctrl_params.P
    R = ctrl_params.R
    nu = ctrl_params.nu

    v_hat = v_hat + dt * \
        np.concatenate([v_hat[3:6], np.linalg.inv(
            M) @ (last_u + g - C @ v_hat + phi(x) @ a_hat)]) + dt* K@(v - v_hat)
    
    a_hat_new = - P @ phi(x).T @ np.linalg.inv(R) @ (v_hat - v)* dt + a_hat
    a_hat = nu * a_hat + (1.0 - nu) * a_hat_new

    u = - 4.0 * (p - reference(t)) - 4.0 * (v - reference_dot(t))

    ctrl_params.v_hat = v_hat 
    ctrl_params.a_hat = a_hat   
    ctrl_params.last_u = u
    
    return u






state_log, ctrl_log = simulate(np.array([3.0, 2.0, 1.0, 0.5, 0.6, 0.7]), controller, N, dt)

t = np.linspace(0.0, T, N)
x = state_log[0, :]
x_ref = reference(t)[0]
y = state_log[1, :]
y_ref = reference(t)[1]
z = state_log[2, :]
z_ref = reference(t)[2]
vx = state_log[3, :]
vx_ref = reference_dot(t)[0]
vx_hat = ctrl_log[0, :]
vy = state_log[4, :]
vy_ref = reference_dot(t)[1]
vy_hat = ctrl_log[1, :]
vz = state_log[5, :]
vz_ref = reference_dot(t)[2]
vz_hat = ctrl_log[2, :]
a_hat = ctrl_log[3:9, :]
d = np.array([disturbance(state_log[:, i]) for i in range(N)]).T
d_est = np.array([phi(state_log[:, i]) @ a_hat[:, i] for i in range(N)]).T
dx = d[0]
dx_est = d_est[0]
dy = d[1]
dy_est = d_est[1]
dz = d[2]
dz_est = d_est[2]

plt.figure(figsize=(20, 10))

plt.subplot(5, 3, 1)
plt.plot(t, x, label='x')
plt.plot(t, x_ref, label='x_ref')
plt.legend()

plt.subplot(5, 3, 2)
plt.plot(t, y, label='y')
plt.plot(t, y_ref, label='y_ref')
plt.legend()

plt.subplot(5, 3, 3)
plt.plot(t, z, label='z')
plt.plot(t, z_ref, label='z_ref')
plt.legend()

plt.subplot(5, 3, 4)
plt.plot(t, vx, label='vx')
plt.plot(t, vx_ref, label='vx_ref')
plt.plot(t, vx_hat, label='vx_hat')
plt.legend()

plt.subplot(5, 3, 5)
plt.plot(t, vy, label='vy')
plt.plot(t, vy_ref, label='vy_ref')
plt.plot(t, vy_hat, label='vy_hat')
plt.legend()

plt.subplot(5, 3, 6)
plt.plot(t, vz, label='vz')
plt.plot(t, vz_ref, label='vz_ref')
plt.plot(t, vz_hat, label='vz_hat')
plt.legend()

plt.subplot(5, 3, 7)
plt.plot(t, dx, label='dx')
plt.plot(t, dx_est, label='dx_est')
plt.legend()

plt.subplot(5, 3, 8)
plt.plot(t, dy, label='dy')
plt.plot(t, dy_est, label='dy_est')
plt.legend()

plt.subplot(5, 3, 9)
plt.plot(t, dz, label='dz')
plt.plot(t, dz_est, label='dz_est')
plt.legend()

plt.subplot(5, 3, 10)
plt.plot(t, ctrl_log[3, :], label='a_dot_vx')
plt.plot([t[0], t[-1]], [-1.0, -1.0], label='a_dot_vx_ref')
plt.legend()

plt.subplot(5, 3, 11)
plt.plot(t, ctrl_log[4, :], label='a_dot_vy')
plt.plot([t[0], t[-1]], [-1.0, -1.0], label='a_dot_vx_ref')
plt.legend()

plt.subplot(5, 3, 12)
plt.plot(t, ctrl_log[5, :], label='a_dot_vz')
plt.plot([t[0], t[-1]], [-1.0, -1.0], label='a_dot_vx_ref')
plt.legend()

plt.subplot(5, 3, 13)
plt.plot(t, ctrl_log[6, :], label='a_x')
plt.plot([t[0], t[-1]], [-1.0, -1.0], label='a_x_ref')
plt.legend()

plt.subplot(5, 3, 14)
plt.plot(t, ctrl_log[7, :], label='a_y')
plt.plot([t[0], t[-1]], [-0.0, 0.0], label='a_y_ref')
plt.legend()

plt.subplot(5, 3, 15)
plt.plot(t, ctrl_log[8, :], label='a_z')
plt.plot([t[0], t[-1]], [1.0, 1.0], label='a_z_ref')
plt.legend()


plt.savefig('../../results/nlsq_test.png')


import jax
from jax import numpy as jnp

class NLACParams:
    a_hat: jnp.ndarray = jnp.zeros((6,), dtype=jnp.float32)
    d_hat: jnp.ndarray = jnp.zeros((3,), dtype=jnp.float32)
    vel_hat: jnp.ndarray = jnp.zeros((3,), dtype=jnp.float32)
    last_u: jnp.ndarray = jnp.zeros((3,), dtype=jnp.float32)
    As: jnp.ndarray = -2.0 * jnp.eye(3, dtype=jnp.float32)
    P: jnp.ndarray = 2.0 * jnp.eye(6, dtype=jnp.float32)
    R: jnp.ndarray = 1.0 * jnp.eye(3, dtype=jnp.float32)
    alpha: jnp.ndarray = 0.9 * jnp.ones((6,), dtype=jnp.float32)

    Kp: float = 4.0
    Kd: float = 4.0
    Kp_att: float = 4.0
    
class stateInfo:
    vel: jnp.ndarray = jnp.zeros((3,), dtype=jnp.float32)

control_params = NLACParams()
state = stateInfo()

class NLAdaptiveController():
    def phi(v):
        return jnp.concatenate([jnp.diag(v*jnp.abs(v)), jnp.eye(3)], axis=1)


jax.debug.print("phi@a_hat {phi_a_hat}", phi_a_hat=NLAdaptiveController.phi(state.vel)@ control_params.a_hat)

jax.debug.print("As @")

vel_hat_dot = jnp.array([0.0, 0.0, -1.0]) + \
    NLAdaptiveController.phi(state.vel) @ control_params.a_hat + \
    control_params.As @ (control_params.vel_hat - state.vel)
    
    
vel_hat = control_params.vel_hat + vel_hat_dot 

jax.debug.print("vel_hat_dot {vel_hat_dot} vel_hat {vel_hat}", vel_hat_dot=vel_hat_dot, vel_hat=vel_hat)       
        
a_hat_new = - control_params.P @ NLAdaptiveController.phi(state.vel).T @ jnp.linalg.inv(control_params.R) @ (state.vel - vel_hat) * 0.02 + control_params.a_hat



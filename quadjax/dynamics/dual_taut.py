import sympy as sp
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    RigidBody,
    inertia,
)
from jax import numpy as jnp
from functools import partial

from quadjax.dynamics.utils import angle_normalize
from quadjax.dynamics import geom, EnvParamsDual2D, EnvStateDual2D, ActionDual2D

def get_dual_taut_dynamics_2d():

    # Define symbols
    t = sp.Symbol("t")  # time
    m0 = sp.Symbol("m0", positive=True)  # mass of the quadrotor 0
    m1 = sp.Symbol("m1", positive=True)  # mass of the quadrotor 1
    I0 = sp.Symbol("I0", positive=True)  # moment of inertia of the quadrotor 0
    I1 = sp.Symbol("I1", positive=True)  # moment of inertia of the quadrotor 1
    g = sp.Symbol("g", positive=True)  # gravitational acceleration
    l0 = sp.Symbol("l0", positive=True)  # length of the rod 0
    l1 = sp.Symbol("l1", positive=True)  # length of the rod 1
    # mass of the object attached to the rod
    mo = sp.Symbol("mo", positive=True)
    thrust0 = sp.Function("thrust0")(t)  # thrust force of the quadrotor 0
    thrust1 = sp.Function("thrust1")(t)  # thrust force of the quadrotor 1
    tau0 = sp.Function("tau0")(t)  # torque of the quadrotor 0
    tau1 = sp.Function("tau1")(t)  # torque of the quadrotor 1
    f_rope0 = sp.Symbol("f_rope0")  # force in the rope 0
    f_rope1 = sp.Symbol("f_rope1")  # force in the rope 1
    # y displacement of the hook from the quadrotor center of quadrotor 0, 1
    delta_yh0 = sp.Symbol("delta_yh0")
    delta_yh1 = sp.Symbol("delta_yh1")
    # z displacement of the hook from the quadrotor center
    delta_zh0 = sp.Symbol("delta_zh")
    delta_zh1 = sp.Symbol("delta_zh1")
    params = [m0, m1, I0, I1, g, l0, l1, mo, delta_yh0, delta_zh0, delta_yh1, delta_zh1]
    action = [thrust0, tau0, thrust1, tau1]

    # Define state variables and their derivatives
    y_obj, z_obj = dynamicsymbols("y_obj z_obj")
    theta0, phi0 = dynamicsymbols("theta0 phi0")
    theta1, phi1 = dynamicsymbols("theta1 phi1")
    y_obj_dot, z_obj_dot = sp.diff(y_obj, t), sp.diff(z_obj, t)
    y_obj_ddot, z_obj_ddot = sp.diff(y_obj_dot, t), sp.diff(z_obj_dot, t)
    theta0_dot, phi0_dot = sp.diff(theta0, t), sp.diff(phi0, t)
    theta1_dot, phi1_dot = sp.diff(theta1, t), sp.diff(phi1, t)
    theta0_ddot, phi0_ddot = sp.diff(theta0_dot, t), sp.diff(phi0_dot, t)
    theta1_ddot, phi1_ddot = sp.diff(theta1_dot, t), sp.diff(phi1_dot, t)
    # Define the values of the state variables for later expression substitution
    y_obj_dot_val, z_obj_dot_val = sp.symbols("y_obj_dot_val z_obj_dot_val")
    y_obj_ddot_val, z_obj_ddot_val = sp.symbols("y_obj_ddot_val z_obj_ddot_val")
    theta0_dot_val, phi0_dot_val = sp.symbols("theta0_dot_val phi0_dot_val")
    theta1_dot_val, phi1_dot_val = sp.symbols("theta1_dot_val phi1_dot_val")
    theta0_ddot_val, phi0_ddot_val = sp.symbols("theta0_ddot_val phi0_ddot_val")
    theta1_ddot_val, phi1_ddot_val = sp.symbols("theta1_ddot_val phi1_ddot_val")

    states = [y_obj, z_obj, y_obj_dot, z_obj_dot, theta0, phi0, theta0_dot, phi0_dot,
            theta1, phi1, theta1_dot, phi1_dot]
    states_val = [y_obj, z_obj, y_obj_dot_val, z_obj_dot_val, theta0, phi0, theta0_dot_val, phi0_dot_val,
            theta1, phi1, theta1_dot_val, phi1_dot_val]
    states_dot = [y_obj_ddot, z_obj_ddot, theta0_ddot, phi0_ddot, f_rope0, theta1_ddot, phi1_ddot, f_rope1]
    states_dot_val = [y_obj_ddot_val, z_obj_ddot_val, theta0_ddot_val, phi0_ddot_val, f_rope0, theta1_ddot_val, phi1_ddot_val, f_rope1]

    # intermediate variables
    # for quadrotor 0
    delta_yh0_world = delta_yh0 * sp.cos(theta0) - delta_zh0 * sp.sin(theta0)
    delta_zh0_world = delta_yh0 * sp.sin(theta0) + delta_zh0 * sp.cos(theta0)
    y_hook0 = y_obj - l0 * sp.sin(phi0)
    z_hook0 = z_obj + l0 * sp.cos(phi0)
    y_hook0_dot = sp.diff(y_hook0, t)
    z_hook0_dot = sp.diff(z_hook0, t)
    y0 = y_hook0 - delta_yh0_world 
    z0 = z_hook0 - delta_zh0_world
    y0_dot = sp.diff(y0, t)
    z0_dot = sp.diff(z0, t)
    y0_ddot = sp.diff(y0_dot, t)
    z0_ddot = sp.diff(z0_dot, t)
    f_rope0_y = - f_rope0 * sp.sin(phi0)
    f_rope0_z = f_rope0 * sp.cos(phi0)
    # for quadrotor 1
    delta_yh1_world = delta_yh1 * sp.cos(theta1) - delta_zh1 * sp.sin(theta1)
    delta_zh1_world = delta_yh1 * sp.sin(theta1) + delta_zh1 * sp.cos(theta1)
    y_hook1 = y_obj - l1 * sp.sin(phi1)
    z_hook1 = z_obj + l1 * sp.cos(phi1)
    y_hook1_dot = sp.diff(y_hook1, t)
    z_hook1_dot = sp.diff(z_hook1, t)
    y1 = y_hook1 - delta_yh1_world
    z1 = z_hook1 - delta_zh1_world
    y1_dot = sp.diff(y1, t)
    z1_dot = sp.diff(z1, t)
    y1_ddot = sp.diff(y1_dot, t)
    z1_ddot = sp.diff(z1_dot, t)
    f_rope1_y = - f_rope1 * sp.sin(phi1)
    f_rope1_z = f_rope1 * sp.cos(phi1)
    obs_eqs = [y0, z0, y0_dot, z0_dot, y0_ddot, z0_ddot, f_rope0_y, f_rope0_z, y_hook0, z_hook0, y_hook0_dot, z_hook0_dot,
            y1, z1, y1_dot, z1_dot, y1_ddot, z1_ddot, f_rope1_y, f_rope1_z, y_hook1, z_hook1, y_hook1_dot, z_hook1_dot]

    # Newton's law
    # quadrotor 0
    eq_quad0_y = -thrust0 * sp.sin(theta0) - f_rope0_y - m0 * y0_ddot
    eq_quad0_z = thrust0 * sp.cos(theta0) - f_rope0_z - m0 * g - m0 * z0_ddot
    eq_quad0_theta = tau0 + delta_yh0_world * f_rope0_z - \
        delta_zh0_world * f_rope0_y - I0 * theta0_ddot
    # quadrotor 1
    eq_quad1_y = -thrust1 * sp.sin(theta1) - f_rope1_y - m1 * y1_ddot
    eq_quad1_z = thrust1 * sp.cos(theta1) - f_rope1_z - m1 * g - m1 * z1_ddot
    eq_quad1_theta = tau1 + delta_yh1_world * f_rope1_z - \
        delta_zh1_world * f_rope1_y - I1 * theta1_ddot
    # object
    eq_obj_y = f_rope0_y + f_rope1_y - mo * y_obj_ddot
    eq_obj_z = f_rope0_z + f_rope1_z - mo * g - mo * z_obj_ddot

    # convert equations to the form of A * states_dot = b
    eqs = [eq_quad0_y, eq_quad0_z, eq_quad0_theta, eq_obj_y, eq_obj_z, eq_quad1_y, eq_quad1_z, eq_quad1_theta]
    eqs = [eq.expand() for eq in eqs]
    eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                    for i in range(len(states_dot))]) for eq in eqs]
    eqs = [eq.subs([(states[i], states_val[i])
                    for i in range(len(states))]) for eq in eqs]
    # Solve for the acceleration
    A_taut_dyn = sp.zeros(8, 8)
    b_taut_dyn = sp.zeros(8, 1)
    for i in range(8):
        for j in range(8):
            A_taut_dyn[i, j] = eqs[i].coeff(states_dot_val[j])
        b_taut_dyn[i] = -eqs[i].subs([(states_dot_val[j], 0) for j in range(8)])

    # lambda A_taut_dyn
    A_taut_dyn_func = sp.lambdify(
        params + states_val + action, A_taut_dyn, "jax")
    b_taut_dyn_func = sp.lambdify(
        params + states_val + action, b_taut_dyn, "jax")

    # Solve for equations for other state variables
    # replace states with states_val, states_dot with states_dot_val
    obs_eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                       for i in range(len(states_dot))]) for eq in obs_eqs]
    obs_eqs = [eq.subs([(states[i], states_val[i])
                       for i in range(len(states))]) for eq in obs_eqs]
    # lambda obs_eqs to get other related state variables
    obs_eqs_func = sp.lambdify(
        params + states_val + states_dot_val + action, obs_eqs, "jax")

    # dynamics (params, states) -> states_dot
    def dual_taut_dynamics_2d(env_params: EnvParamsDual2D, env_state: EnvStateDual2D, env_action: ActionDual2D):
        params = [env_params.m0, env_params.m1, env_params.I0, env_params.I1, env_params.g, env_params.l0, env_params.l1, env_params.mo, env_params.delta_yh0, env_params.delta_zh0, env_params.delta_yh1, env_params.delta_zh1]
        states = [env_state.y_obj, env_state.z_obj, env_state.y_obj_dot, env_state.z_obj_dot, env_state.theta0, env_state.phi0, env_state.theta0_dot, env_state.phi0_dot,
                env_state.theta1, env_state.phi1, env_state.theta1_dot, env_state.phi1_dot]
        action = [env_action.thrust0, env_action.tau0, env_action.thrust1, env_action.tau1]
        A = A_taut_dyn_func(*params, *states, *action)
        b = b_taut_dyn_func(*params, *states, *action)
        states_dot = jnp.linalg.solve(A, b).squeeze()
        y_obj_ddot, z_obj_ddot, theta0_ddot, phi0_ddot, f_rope0, theta1_ddot, phi1_ddot, f_rope1 = states_dot

        # Calculate updated state variables
        new_y_obj_dot = env_state.y_obj_dot + y_obj_ddot * env_params.dt
        new_z_obj_dot = env_state.z_obj_dot + z_obj_ddot * env_params.dt
        new_theta0_dot = env_state.theta0_dot + theta0_ddot * env_params.dt
        new_phi0_dot = env_state.phi0_dot + phi0_ddot * env_params.dt
        new_theta1_dot = env_state.theta1_dot + theta1_ddot * env_params.dt
        new_phi1_dot = env_state.phi1_dot + phi1_ddot * env_params.dt
        new_y_obj = env_state.y_obj + new_y_obj_dot * env_params.dt
        new_z_obj = env_state.z_obj + new_z_obj_dot * env_params.dt
        new_theta0 = angle_normalize(
            env_state.theta0 + new_theta0_dot * env_params.dt)
        new_phi0 = angle_normalize(env_state.phi0 + new_phi0_dot * env_params.dt)
        new_theta1 = angle_normalize(
            env_state.theta1 + new_theta1_dot * env_params.dt)
        new_phi1 = angle_normalize(env_state.phi1 + new_phi1_dot * env_params.dt)

        # Update states list
        states = [new_y_obj, new_z_obj, new_y_obj_dot, new_z_obj_dot, new_theta0, new_phi0, new_theta0_dot, new_phi0_dot,
                new_theta1, new_phi1, new_theta1_dot, new_phi1_dot]

        # Compute other state variables
        y0, z0, y0_dot, z0_dot, y0_ddot, z0_ddot, f_rope0_y, f_rope0_z, y_hook0, z_hook0, y_hook0_dot, z_hook0_dot, \
            y1, z1, y1_dot, z1_dot, y1_ddot, z1_ddot, f_rope1_y, f_rope1_z, y_hook1, z_hook1, y_hook1_dot, z_hook1_dot \
                 = obs_eqs_func(*params, *states, *states_dot, *action)

        # Update all state variables at once using replace()
        env_state = env_state.replace(
            y0 = y0,
            z0 = z0,
            theta0 = new_theta0,
            phi0 = new_phi0,
            y0_dot = y0_dot,
            z0_dot = z0_dot,
            theta0_dot = new_theta0_dot,
            phi0_dot = new_phi0_dot,
            last_thrust0 = env_action.thrust0,
            last_tau0 = env_action.tau0,
            y_hook0 = y_hook0,
            z_hook0 = z_hook0,
            y_hook0_dot = y_hook0_dot,
            z_hook0_dot = z_hook0_dot,
            f_rope0 = f_rope0,
            f_rope0_y = f_rope0_y,
            f_rope0_z = f_rope0_z,
            l_rope0 = env_params.l0,

            y1 = y1,
            z1 = z1,
            theta1 = new_theta1,
            phi1 = new_phi1,
            y1_dot = y1_dot,
            z1_dot = z1_dot,
            theta1_dot = new_theta1_dot,
            phi1_dot = new_phi1_dot,
            last_thrust1 = env_action.thrust1,
            last_tau1 = env_action.tau1,
            y_hook1 = y_hook1,
            z_hook1 = z_hook1,
            y_hook1_dot = y_hook1_dot,
            z_hook1_dot = z_hook1_dot,
            f_rope1 = f_rope1,
            f_rope1_y = f_rope1_y,
            f_rope1_z = f_rope1_z,
            l_rope1 = env_params.l1,

            y_obj = new_y_obj,
            z_obj = new_z_obj,
            y_obj_dot = new_y_obj_dot,
            z_obj_dot = new_z_obj_dot,

            time=env_state.time + 1,
            y_tar=env_state.y_traj[env_state.time],
            z_tar=env_state.z_traj[env_state.time],
            y_dot_tar=env_state.y_dot_traj[env_state.time],
            z_dot_tar=env_state.z_dot_traj[env_state.time],
        )

        return env_state

    return dual_taut_dynamics_2d
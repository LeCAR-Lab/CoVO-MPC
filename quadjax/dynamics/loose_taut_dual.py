import sympy as sp
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    RigidBody,
    inertia,
)
import jax
from jax import numpy as jnp
from functools import partial

from quadjax.dynamics.utils import angle_normalize
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import EnvParams, EnvState, Action, EnvParams3D, EnvState3D, Action3D
from typing import Tuple
from jax import jit
import numpy as np


def get_loose_taut_dynamics():

    # Define symbols
    t = sp.Symbol("t")  # time
    m = sp.Symbol("m", positive=True)  # mass of the quadrotor
    I = sp.Symbol("I", positive=True)  # moment of inertia
    g = sp.Symbol("g", positive=True)  # gravitational acceleration
    l = sp.Symbol("l", positive=True)  # length of the rod
    # mass of the object attached to the rod
    mo = sp.Symbol("mo", positive=True)
    thrust = sp.Function("thrust")(t)  # thrust force
    tau = sp.Function("tau")(t)  # torque
    f_rope = sp.Symbol("f_rope")  # force in the rope
    thrust2 = sp.Function("thrust2")(t)  # thrust force
    tau2 = sp.Function("tau2")(t)  # torque
    f_rope2 = sp.Symbol("f_rope2")  # force in the rope
    # y displacement of the hook from the quadrotor center
    delta_yh = sp.Symbol("delta_yh")
    # z displacement of the hook from the quadrotor center
    delta_zh = sp.Symbol("delta_zh")
    # y displacement of the hook from the quadrotor center
    delta_yh2 = sp.Symbol("delta_yh2")
    # z displacement of the hook from the quadrotor center
    delta_zh2 = sp.Symbol("delta_zh2")
    # TODO...
    params = [m, I, g, l, mo, delta_yh, delta_zh]
    # action = [thrust, tau]
    action = [thrust, tau, thrust2, tau2]

    # Define state variables and their derivatives
    y, z, theta, phi = dynamicsymbols("y z theta phi")
    y_dot, z_dot, theta_dot, phi_dot = sp.diff(y, t), sp.diff(
        z, t), sp.diff(theta, t), sp.diff(phi, t)
    y_ddot, z_ddot, theta_ddot, phi_ddot = sp.diff(y_dot, t), sp.diff(
        z_dot, t), sp.diff(theta_dot, t), sp.diff(phi_dot, t)
    y_dot_val, z_dot_val, theta_dot_val, phi_dot_val = sp.symbols(
        "y_dot_val z_dot_val theta_dot_val phi_dot_val")
    y_ddot_val, z_ddot_val, theta_ddot_val, phi_ddot_val = sp.symbols(
        "y_ddot_val z_ddot_val theta_ddot_val phi_ddot_val")
    # states = [y, z, theta, phi, y_dot, z_dot, theta_dot, phi_dot]
    # states_val = [y, z, theta, phi, y_dot_val,
    #               z_dot_val, theta_dot_val, phi_dot_val]
    # states_dot = [y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope]
    # states_dot_val = [y_ddot_val, z_ddot_val,
    #                   theta_ddot_val, phi_ddot_val, f_rope]

    y2, z2, theta2, phi2 = dynamicsymbols("y2 z2 theta2 phi2")
    y2_dot, z2_dot, theta2_dot, phi2_dot = sp.diff(y2, t), sp.diff(
        z2, t), sp.diff(theta2, t), sp.diff(phi2, t)
    y2_ddot, z2_ddot, theta2_ddot, phi2_ddot = sp.diff(y2_dot, t), sp.diff(
        z2_dot, t), sp.diff(theta2_dot, t), sp.diff(phi2_dot, t)
    y2_dot_val, z2_dot_val, theta2_dot_val, phi2_dot_val = sp.symbols(
        "y2_dot_val z2_dot_val theta2_dot_val phi2_dot_val")
    y2_ddot_val, z2_ddot_val, theta2_ddot_val, phi2_ddot_val = sp.symbols(
        "y2_ddot_val z2_ddot_val theta2_ddot_val phi2_ddot_val")

    states = [y, z, theta, phi, y_dot, z_dot, theta_dot, phi_dot,
            y2, z2, theta2, phi2, y2_dot, z2_dot, theta2_dot, phi2_dot]
    states_val = [y, z, theta, phi, y_dot_val,
                    z_dot_val, theta_dot_val, phi_dot_val, 
                    y2, z2, theta2, phi2, y2_dot_val, z2_dot_val, theta2_dot_val, phi2_dot_val]
    states_dot = [y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope, y2_ddot, z2_ddot, theta2_ddot, phi2_ddot, f_rope2]
    states_dot_val = [y_ddot_val, z_ddot_val,
                        theta_ddot_val, phi_ddot_val, f_rope, 
                        y2_ddot_val, z2_ddot_val,
                        theta2_ddot_val, phi2_ddot_val, f_rope2]
    # states_dot = [y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope, theta2_ddot, phi2_ddot, f_rope2]
    # states_dot_val = [y_ddot_val, z_ddot_val,
    #                     theta_ddot_val, phi_ddot_val, f_rope,
    #                     theta2_ddot_val, phi2_ddot_val, f_rope2]

    # intermediate variables
    delta_yh_global = delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta)
    delta_zh_global = delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)
    f_rope_y = f_rope * sp.sin(theta+phi)
    f_rope_z = -f_rope * sp.cos(theta+phi)
    y_hook = y + delta_yh_global
    z_hook = z + delta_zh_global
    y_hook_dot = sp.diff(y_hook, t)
    z_hook_dot = sp.diff(z_hook, t)
    y_obj = y_hook + l * sp.sin(theta+phi)
    z_obj = z_hook - l * sp.cos(theta+phi)
    y_obj_dot = sp.diff(y_obj, t)
    z_obj_dot = sp.diff(z_obj, t)
    y_obj_ddot = sp.diff(y_obj_dot, t)
    z_obj_ddot = sp.diff(z_obj_dot, t)
    obses = [y_obj, z_obj, y_obj_dot, z_obj_dot,
                y_obj_ddot, z_obj_ddot, f_rope_y, f_rope_z]

    delta_yh2_global = delta_yh2 * sp.cos(theta2) - delta_zh2 * sp.sin(theta2)
    delta_zh2_global = delta_yh2 * sp.sin(theta2) + delta_zh2 * sp.cos(theta2)
    f_rope2_y = f_rope2 * sp.sin(theta2+phi2)
    f_rope2_z = -f_rope2 * sp.cos(theta2+phi2)
    y_hook2 = y2 + delta_yh2_global
    z_hook2 = z2 + delta_zh2_global
    y_hook2_dot = sp.diff(y_hook2, t)
    z_hook2_dot = sp.diff(z_hook2, t)


    # Define inertial reference frame
    N = ReferenceFrame("N")
    N_origin = Point('N_origin')
    A = N.orientnew("A", "Axis", [theta, N.x])
    B = A.orientnew("B", "Axis", [phi, A.x])

    # Define point
    drone = Point("drone")
    drone.set_pos(N_origin, y * N.y + z * N.z)
    hook = drone.locatenew("hook", delta_yh * A.y + delta_zh * A.z)
    drone2 = Point("drone2")
    drone2.set_pos(N_origin, y2 * N.y + z2 * N.z)
    hook2 = drone2.locatenew("hook2", delta_yh2 * A.y + delta_zh2 * A.z)
    obj = hook.locatenew("obj", -l * B.z)
    drone.set_vel(N, y_dot * N.y + z_dot * N.z)
    drone2.set_vel(N, y2_dot * N.y + z2_dot * N.z)

    # Inertia
    inertia_quadrotor = inertia(N, I, 0, 0)
    quadrotor = RigidBody("quadrotor", drone, A, m, (inertia_quadrotor, drone))
    quadrotor2 = RigidBody("quadrotor2", drone2, A, m, (inertia_quadrotor, drone2))
    obj_particle = Particle("obj_particle", obj, mo)

    # Newton's law
    # 8 equations
    eq_quad_y = -thrust * sp.sin(theta) + f_rope_y - m * y_ddot
    eq_quad_z = thrust * sp.cos(theta) + f_rope_z - m * g - m * z_ddot
    eq_quad_theta = tau + delta_yh_global * f_rope_z - \
        delta_zh_global * f_rope_y - I * theta_ddot
    eq_obj_y = -f_rope_y - mo * y_obj_ddot
    eq_obj_z = -f_rope_z - mo * g - mo * z_obj_ddot

   
    # TODO...
    eqs = [eq_quad_y, eq_quad_z, eq_quad_theta, eq_obj_y, eq_obj_z]
    eqs = [eq.expand() for eq in eqs]
    eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                    for i in range(len(states_dot))]) for eq in eqs]
    eqs = [eq.subs([(states[i], states_val[i])
                    for i in range(len(states))]) for eq in eqs]
    # Solve for the acceleration
    A_taut_dyn = sp.zeros(5, 5)
    b_taut_dyn = sp.zeros(5, 1)
    for i in range(5):
        for j in range(5):
            A_taut_dyn[i, j] = eqs[i].coeff(states_dot_val[j])
        b_taut_dyn[i] = -eqs[i].subs([(states_dot_val[j], 0)
                                        for j in range(5)])

    # lambda A_taut_dyn
    A_taut_dyn_func = sp.lambdify(
        params + states_val + action, A_taut_dyn, "jax")
    b_taut_dyn_func = sp.lambdify(
        params + states_val + action, b_taut_dyn, "jax")

    # Solve for equation
    obs_eqs = [y_obj, z_obj, y_obj_dot, z_obj_dot, y_hook,
               z_hook, y_hook_dot, z_hook_dot, f_rope_y, f_rope_z]
    # replace states with states_val, states_dot with states_dot_val
    obs_eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                       for i in range(len(states_dot))]) for eq in obs_eqs]
    obs_eqs = [eq.subs([(states[i], states_val[i])
                       for i in range(len(states))]) for eq in obs_eqs]
    # lambda obs_eqs
    obs_eqs_func = sp.lambdify(
        params + states_val + states_dot_val + action, obs_eqs, "jax")

    # dynamics (params, states) -> states_dot
    def loose_taut_dynamics(env_params: EnvParams, env_state: EnvState, env_action: Tuple[Action, Action], taut_index: bool):
        if taut_index:
            # the 2rd drone's rope is taut
            params = [env_params.m, env_params.I, env_params.g, env_params.l,
                  env_params.mo,env_params.delta_yh2, env_params.delta_zh2]
            states = [env_state.y2, env_state.z2, env_state.theta2, env_state.phi2,
                    env_state.y2_dot, env_state.z2_dot, env_state.theta2_dot, env_state.phi2_dot]
            action = [env_action[1].thrust, env_action[1].tau,
                env_action[0].thrust, env_action[0].tau]
        else:
            # the 1st drone's rope is taut
            params = [env_params.m, env_params.I, env_params.g, env_params.l,
                    env_params.mo, env_params.delta_yh, env_params.delta_zh]
            states = [env_state.y, env_state.z, env_state.theta, env_state.phi,
                    env_state.y_dot, env_state.z_dot, env_state.theta_dot, env_state.phi_dot,]
            action = [env_action[0].thrust, env_action[0].tau,
                    env_action[1].thrust, env_action[1].tau]
        
        jax.debug.print("taut_params: {}\n", params)
        jax.debug.print("taut_states: {}\n", states)
        jax.debug.print("taut_action: {}\n", action)
        A = A_taut_dyn_func(*params, *states, *action)
        b = b_taut_dyn_func(*params, *states, *action)
        states_dot = jnp.linalg.solve(A, b).squeeze()
        # y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope, y2_ddot, z2_ddot, theta2_ddot, phi2_ddot, f_rope2 = states_dot
        y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope = states_dot
        jax.debug.print("result: {}\n", states_dot)

        # Calculate updated state variables
        new_y_dot = env_state.y_dot + y_ddot * env_params.dt
        new_z_dot = env_state.z_dot + z_ddot * env_params.dt
        new_theta_dot = env_state.theta_dot + theta_ddot * env_params.dt
        new_phi_dot = env_state.phi_dot + phi_ddot * env_params.dt
        new_y = env_state.y + new_y_dot * env_params.dt
        new_z = env_state.z + new_z_dot * env_params.dt
        new_theta = angle_normalize(
            env_state.theta + new_theta_dot * env_params.dt)
        new_phi = angle_normalize(env_state.phi + new_phi_dot * env_params.dt)
        
        
        thrust2 = action[2]
        tau2 = action[3]
        y2_ddot = -thrust2 * jnp.sin(env_state.theta2) / env_params.m
        z2_ddot = thrust2 * \
            jnp.cos(env_state.theta2) / env_params.m - env_params.g
        theta2_ddot = tau2 / env_params.I

        new_y2_dot = env_state.y2_dot + y2_ddot * env_params.dt
        new_z2_dot = env_state.z2_dot + z2_ddot * env_params.dt
        new_theta2_dot = env_state.theta2_dot + theta2_ddot * env_params.dt
        new_phi2_dot = env_state.phi2_dot + phi2_ddot * env_params.dt
        new_y2 = env_state.y2 + new_y2_dot * env_params.dt
        new_z2 = env_state.z2 + new_z2_dot * env_params.dt
        new_theta2 = angle_normalize(
            env_state.theta2 + new_theta2_dot * env_params.dt)
        new_phi2 = angle_normalize(env_state.phi2 + new_phi2_dot * env_params.dt)
        
        delta_y_hook2 = env_params.delta_yh2 * \
            jnp.cos(new_theta2) - env_params.delta_zh2 * jnp.sin(new_theta2)
        delta_z_hook2 = env_params.delta_yh2 * \
            jnp.sin(new_theta2) + env_params.delta_zh2 * jnp.cos(new_theta2)
        
        y_hook2 = new_y2 + delta_y_hook2
        z_hook2 = new_z2 + delta_z_hook2
        y_hook2_dot = new_y2_dot - new_theta2_dot * delta_z_hook2
        z_hook2_dot = new_z2_dot + new_theta2_dot * delta_y_hook2

        # Update states list
        states = [new_y, new_z, new_theta, new_phi,
                  new_y_dot, new_z_dot, new_theta_dot, new_phi_dot]

        # Compute other state variables
        y_obj, z_obj, y_obj_dot, z_obj_dot, y_hook, z_hook, y_hook_dot, z_hook_dot, f_rope_y, f_rope_z = obs_eqs_func(
            *params, *states, *states_dot, *action)
        
        new_y_obj = env_state.y_obj + env_params.dt * y_obj_dot
        new_z_obj = env_state.z_obj + env_params.dt * z_obj_dot
        
        # phi related values
        phi2_th = -jnp.arctan2(y_hook2 - new_y_obj, z_hook2 - new_z_obj)
        new_phi2 = angle_normalize(phi2_th - new_theta2)
        y_obj2hook2_dot = y_obj_dot - y_hook2_dot
        z_obj2hook2_dot = y_obj_dot - z_hook2_dot
        phi2_th_dot = y_obj2hook2_dot * \
            jnp.cos(phi2_th) + z_obj2hook2_dot * jnp.sin(phi2_th)
        new_phi2_dot = phi2_th_dot - new_theta2_dot

        new_l_rope = jnp.sqrt((y_hook2 - y_obj) **2 + (z_hook2 - z_obj) ** 2)

        if taut_index:
            env_state = env_state.replace(
            y2=new_y,
            z2=new_z,
            theta2=new_theta,
            phi2=new_phi,
            y2_dot=new_y_dot,
            z2_dot=new_z_dot,
            theta2_dot=new_theta_dot,
            phi2_dot=new_phi_dot,
            y_obj=y_obj,
            z_obj=z_obj,
            y_obj_dot=y_obj_dot,
            z_obj_dot=z_obj_dot,
            y_hook2=y_hook,
            z_hook2=z_hook,
            y_hook2_dot=y_hook_dot,
            z_hook2_dot=z_hook_dot,
            y=new_y2,
            z=new_z2,
            theta=new_theta2,
            phi=new_phi2,
            y_dot=new_y2_dot,
            z_dot=new_z2_dot,
            theta_dot=new_theta2_dot,
            phi_dot=new_phi2_dot,
            y_hook=y_hook2,
            z_hook=z_hook2,
            y_hook_dot=y_hook2_dot,
            z_hook_dot=z_hook2_dot,
            f_rope2_y=f_rope_y,
            f_rope2_z=f_rope_z,
            f_rope2=f_rope,
            l_rope2=env_params.l,
            last_thrust=env_action[1].thrust,
            last_tau=env_action[1].tau,
            f_rope_y=0.0,
            f_rope_z=0.0,
            f_rope=0.0,
            l_rope=new_l_rope,
            last_thrust2=env_action[0].thrust,
            last_tau2=env_action[0].tau,
            time=env_state.time + 1,
            y_tar=env_state.y_traj[env_state.time],
            z_tar=env_state.z_traj[env_state.time],
            y_dot_tar=env_state.y_dot_traj[env_state.time],
            z_dot_tar=env_state.z_dot_traj[env_state.time],
        )
        else:
            # Update all state variables at once using replace()
            env_state = env_state.replace(
                y=new_y,
                z=new_z,
                theta=new_theta,
                phi=new_phi,
                y_dot=new_y_dot,
                z_dot=new_z_dot,
                theta_dot=new_theta_dot,
                phi_dot=new_phi_dot,
                y_obj=y_obj,
                z_obj=z_obj,
                y_obj_dot=y_obj_dot,
                z_obj_dot=z_obj_dot,
                y_hook=y_hook,
                z_hook=z_hook,
                y_hook_dot=y_hook_dot,
                z_hook_dot=z_hook_dot,
                y2=new_y2,
                z2=new_z2,
                theta2=new_theta2,
                phi2=new_phi2,
                y2_dot=new_y2_dot,
                z2_dot=new_z2_dot,
                theta2_dot=new_theta2_dot,
                phi2_dot=new_phi2_dot,
                y_hook2=y_hook2,
                z_hook2=z_hook2,
                y_hook2_dot=y_hook2_dot,
                z_hook2_dot=z_hook2_dot,
                f_rope_y=f_rope_y,
                f_rope_z=f_rope_z,
                f_rope=f_rope,
                l_rope=env_params.l,
                last_thrust=env_action[0].thrust,
                last_tau=env_action[0].tau,
                f_rope2_y=0.0,
                f_rope2_z=0.0,
                f_rope2=0.0,
                l_rope2=new_l_rope,
                last_thrust2=env_action[1].thrust,
                last_tau2=env_action[1].tau,
                time=env_state.time + 1,
                y_tar=env_state.y_traj[env_state.time],
                z_tar=env_state.z_traj[env_state.time],
                y_dot_tar=env_state.y_dot_traj[env_state.time],
                z_dot_tar=env_state.z_dot_traj[env_state.time],
            )

        return env_state

    return loose_taut_dynamics
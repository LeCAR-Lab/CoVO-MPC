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

# Enable debug mode with print statements
jax.config.update("jax_debug_nans", True)

def get_taut_dynamics():

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
    params = [m, I, g, l, mo, delta_yh, delta_zh, delta_yh2, delta_zh2]
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
    # y_hook2 = y_obj - l * sp.sin(theta2+phi2)
    # z_hook2 = z_obj + l * sp.cos(theta2+phi2)
    y_hook2 = y2 + delta_yh2_global
    z_hook2 = z2 + delta_zh2_global
    # y2 = y_hook2 - delta_yh2_global
    # z2 = z_hook2 - delta_zh2_global
    y_hook2_dot = sp.diff(y_hook2, t)
    z_hook2_dot = sp.diff(z_hook2, t)
    # obses = [y_obj, z_obj, y_obj_dot, z_obj_dot,
    #          y_obj_ddot, z_obj_ddot, f_rope_y, f_rope_z]


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
    # eq_obj_y = -f_rope_y - mo * y_obj_ddot
    # eq_obj_z = -f_rope_z - mo * g - mo * z_obj_ddot

    eq_quad_y2 = -thrust2 * sp.sin(theta2) + f_rope2_y - m * y2_ddot
    eq_quad_z2 = thrust2 * sp.cos(theta2) + f_rope2_z - m * g - m * z2_ddot
    eq_quad_theta2 = tau2 + delta_yh2_global * f_rope2_z - \
        delta_zh2_global * f_rope2_y - I * theta2_ddot
    eq_obj_y = -f_rope_y -f_rope2_y - mo * y_obj_ddot
    eq_obj_z = -f_rope_z -f_rope2_z - mo * g - mo * z_obj_ddot

    # eq_quad1_obj = (y-y_obj)*(y_dot-y_obj_dot)+(z-z_obj)*(z_dot-z_obj_dot)
    # eq_quad2_obj = (y2-y_obj)*(y2_dot-y_obj_dot)+(z2-z_obj)*(z2_dot-z_obj_dot)
    # eq_quad1_obj = (y_dot-y_obj_dot)**2+(y-y_obj)*(y_ddot-y_obj_ddot)+(z_dot-z_obj_dot)**2+(z-z_obj)*(z_ddot-z_obj_ddot)
    # eq_quad2_obj = (y2_dot-y_obj_dot)**2+(y2-y_obj)*(y2_ddot-y_obj_ddot)+(z2_dot-z_obj_dot)**2+(z2-z_obj)*(z2_ddot-z_obj_ddot)
    # eq_quad1_obj = y_ddot + l * (sp.cos(theta + phi) * (theta_ddot + phi_ddot) - sp.sin(theta + phi) * (theta_dot + phi_dot)**2) -\
    #      theta_ddot * (delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)) - theta_dot**2 * (delta_yh * sp.cos(theta - delta_zh * sp.sin(theta))) -\
    #         y2_ddot + l * (sp.cos(theta2 + phi2) * (theta2_ddot + phi2_ddot) - sp.sin(theta2 + phi2) * (theta2_dot + phi2_dot)**2) -\
    #      theta2_ddot * (delta_yh2 * sp.sin(theta2) + delta_zh2 * sp.cos(theta2)) - theta2_dot**2 * (delta_yh2 * sp.cos(theta2 - delta_zh2 * sp.sin(theta2))) 

    # eq_quad2_obj = z_ddot + l * (theta_dot + phi_dot) * (sp.sin(theta + phi) * (theta_ddot + phi_ddot) + sp.cos(theta + phi) * (theta_dot + phi_dot)) - \
    #                 z2_ddot - l * (theta_dot + phi_dot) * (sp.sin(theta + phi) * (theta_ddot + phi_ddot) + sp.cos(theta + phi) * (theta_dot + phi_dot))

    eq_quad1_obj = y_ddot + l * (sp.cos(theta + phi) * (theta_ddot + phi_ddot) - sp.sin(theta + phi) * (theta_dot + phi_dot)**2) -\
         theta_ddot * (delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)) - theta_dot**2 * (delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta)) -\
            y2_ddot - l * (sp.cos(theta2 + phi2) * (theta2_ddot + phi2_ddot) - sp.sin(theta2 + phi2) * (theta2_dot + phi2_dot)**2) +\
         theta2_ddot * (delta_yh2 * sp.sin(theta2) + delta_zh2 * sp.cos(theta2)) + theta2_dot**2 * (delta_yh2 * sp.cos(theta2) - delta_zh2 * sp.sin(theta2))

    eq_quad2_obj = z_ddot + l * (sp.sin(theta + phi) * (theta_ddot + phi_ddot) + sp.cos(theta + phi) * (theta_dot + phi_dot)**2) -\
         theta_ddot * (delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta)) - theta_dot**2 * (delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)) -\
            z2_ddot - l * (sp.sin(theta2 + phi2) * (theta2_ddot + phi2_ddot) + sp.cos(theta2 + phi2) * (theta2_dot + phi2_dot)**2) +\
         theta2_ddot * (delta_yh2 * sp.cos(theta2) - delta_zh2 * sp.sin(theta2)) + theta2_dot**2 * (delta_yh2 * sp.sin(theta2) + delta_zh2 * sp.cos(theta2))

    # TODO...
    eqs = [eq_quad_y, eq_quad_z, eq_quad_theta, eq_obj_y, eq_obj_z, eq_quad_y2, eq_quad_z2, eq_quad_theta2, eq_quad1_obj, eq_quad2_obj]
    # eqs = [eq_quad_y, eq_quad_z, eq_quad_theta, eq_obj_y, eq_obj_z, eq_quad_y2, eq_quad_z2, eq_quad_theta2]
    eqs = [eq.expand() for eq in eqs]
    eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                    for i in range(len(states_dot))]) for eq in eqs]
    eqs = [eq.subs([(states[i], states_val[i])
                    for i in range(len(states))]) for eq in eqs]
    # Solve for the acceleration
    A_taut_dyn = sp.zeros(10, 10)
    b_taut_dyn = sp.zeros(10, 1)
    for i in range(10):
        for j in range(10):
            A_taut_dyn[i, j] = eqs[i].coeff(states_dot_val[j])
        b_taut_dyn[i] = -eqs[i].subs([(states_dot_val[j], 0)
                                        for j in range(10)])

    # lambda A_taut_dyn
    A_taut_dyn_func = sp.lambdify(
        params + states_val + action, A_taut_dyn, "jax")
    b_taut_dyn_func = sp.lambdify(
        params + states_val + action, b_taut_dyn, "jax")

    # Solve for equation
    obs_eqs = [y_obj, z_obj, y_obj_dot, z_obj_dot, y_hook,
               z_hook, y_hook_dot, z_hook_dot, f_rope_y, f_rope_z,
               y_hook2, z_hook2, y_hook2_dot, z_hook2_dot, f_rope2_y, f_rope2_z]
    # replace states with states_val, states_dot with states_dot_val
    obs_eqs = [eq.subs([(states_dot[i], states_dot_val[i])
                       for i in range(len(states_dot))]) for eq in obs_eqs]
    obs_eqs = [eq.subs([(states[i], states_val[i])
                       for i in range(len(states))]) for eq in obs_eqs]
    # lambda obs_eqs
    obs_eqs_func = sp.lambdify(
        params + states_val + states_dot_val + action, obs_eqs, "jax")
    # lambda some other obs functions
    y_hook_func = sp.lambdify(params + states_val, obs_eqs[4], "jax")
    z_hook_func = sp.lambdify(params + states_val, obs_eqs[5], "jax")
    y_hook_dot_func = sp.lambdify(params + states_val, obs_eqs[6], "jax")
    z_hook_dot_func = sp.lambdify(params + states_val, obs_eqs[7], "jax")

    # dynamics (params, states) -> states_dot
    def taut_dynamics(env_params: EnvParams, env_state: EnvState, env_action: Action):
        params = [env_params.m, env_params.I, env_params.g, env_params.l,
                  env_params.mo, env_params.delta_yh, env_params.delta_zh,env_params.delta_yh2, env_params.delta_zh2]
        states = [env_state.y, env_state.z, env_state.theta, env_state.phi,
                  env_state.y_dot, env_state.z_dot, env_state.theta_dot, env_state.phi_dot,
                  env_state.y2, env_state.z2, env_state.theta2, env_state.phi2,
                  env_state.y2_dot, env_state.z2_dot, env_state.theta2_dot, env_state.phi2_dot]
        # action = [env_action.thrust, env_action.tau]
        action = [env_action[0].thrust, env_action[0].tau,
                env_action[1].thrust, env_action[1].tau]
        jax.debug.print("taut_params: {}\n", params)
        jax.debug.print("taut_states: {}\n", states)
        jax.debug.print("taut_action: {}\n", action)
        A = A_taut_dyn_func(*params, *states, *action)
        b = b_taut_dyn_func(*params, *states, *action)
        states_dot = jnp.linalg.solve(A, b).squeeze()
        # y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope = states_dot
        # y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope, y2_ddot, z2_ddot, theta2_ddot, phi2_ddot, f_rope2 = states_dot
        y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope, y2_ddot, z2_ddot, theta2_ddot, phi2_ddot, f_rope2 = states_dot
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
        
        new_y2_dot = env_state.y2_dot + y2_ddot * env_params.dt
        new_z2_dot = env_state.z2_dot + z2_ddot * env_params.dt
        new_theta2_dot = env_state.theta2_dot + theta2_ddot * env_params.dt
        new_phi2_dot = env_state.phi2_dot + phi2_ddot * env_params.dt
        new_y2 = env_state.y2 + new_y2_dot * env_params.dt
        new_z2 = env_state.z2 + new_z2_dot * env_params.dt
        new_theta2 = angle_normalize(
            env_state.theta2 + new_theta2_dot * env_params.dt)
        new_phi2 = angle_normalize(env_state.phi2 + new_phi2_dot * env_params.dt)
        

        # Update states list
        states = [new_y, new_z, new_theta, new_phi,
                  new_y_dot, new_z_dot, new_theta_dot, new_phi_dot,
                  new_y2, new_z2, new_theta2, new_phi2,
                  new_y2_dot, new_z2_dot, new_theta2_dot, new_phi2_dot]

        # Compute other state variables
        y_obj, z_obj, y_obj_dot, z_obj_dot, y_hook, z_hook, y_hook_dot, z_hook_dot, f_rope_y, f_rope_z, y_hook2, z_hook2, y_hook2_dot, z_hook2_dot, f_rope2_y, f_rope2_z = obs_eqs_func(
            *params, *states, *states_dot, *action)

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
            f_rope2_y=f_rope2_y,
            f_rope2_z=f_rope2_z,
            f_rope2=f_rope2,
            l_rope2=env_params.l,
            last_thrust2=env_action[1].thrust,
            last_tau2=env_action[1].tau,
            time=env_state.time + 1,
            y_tar=env_state.y_traj[env_state.time],
            z_tar=env_state.z_traj[env_state.time],
            y_dot_tar=env_state.y_dot_traj[env_state.time],
            z_dot_tar=env_state.z_dot_traj[env_state.time],
        )

        return env_state

    return taut_dynamics


def get_taut_dynamics_3d():
    # define parameter symbols
    g, m, mo, l = sp.symbols("g m mo l")
    I = sp.Matrix(sp.symbols(
        'I_11 I_12 I_13 I_21 I_22 I_23 I_31 I_32 I_33')).reshape(3, 3)
    hook_offset = sp.Matrix(sp.symbols(
        'hook_offset_x hook_offset_y hook_offset_z')).reshape(3, 1)
    params = [g, m, mo, l, I[0, 0], I[0, 1], I[0, 2], I[1, 0], I[1, 1], I[1, 2],
            I[2, 0], I[2, 1], I[2, 2], hook_offset[0], hook_offset[1], hook_offset[2]]
    # define dynamic state
    pos = sp.Matrix(sp.symbols('pos_x pos_y pos_z')).reshape(3, 1)
    vel = sp.Matrix(sp.symbols('vel_x vel_y vel_z')).reshape(3, 1)
    quat = sp.Matrix(sp.symbols('quat_x quat_y quat_z quat_w')).reshape(4, 1)
    omega = sp.Matrix(sp.symbols('omega_x omega_y omega_z')).reshape(3, 1)
    theta_rope = sp.Symbol("theta_rope")
    theta_rope_dot = sp.Symbol("theta_rope_dot")
    phi_rope = sp.Symbol("phi_rope")
    phi_rope_dot = sp.Symbol("phi_rope_dot")
    states = [
        pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2],
        quat[0], quat[1], quat[2], quat[3],
        omega[0], omega[1], omega[2],
        theta_rope, theta_rope_dot, phi_rope, phi_rope_dot
    ]
    # dynamic variables to solve
    acc = sp.Matrix(sp.symbols('acc_x acc_y acc_z')).reshape(3, 1)
    alpha = sp.Matrix(sp.symbols('alpha_x alpha_y alpha_z')).reshape(3, 1)
    theta_rope_ddot = sp.Symbol("theta_rope_ddot")
    phi_rope_ddot = sp.Symbol("phi_rope_ddot")
    f_rope_norm = sp.Symbol("f_rope_norm")
    states_dot = [
        acc[0, 0], acc[1, 0], acc[2, 0],
        alpha[0, 0], alpha[1, 0], alpha[2, 0],
        theta_rope_ddot, phi_rope_ddot, 
        f_rope_norm
    ]
    # get other state variables
    rotmat = quat2rot(quat)
    hook_offset_world = rotmat @ hook_offset
    pos_hook = pos + hook_offset_world
    vel_hook = vel + sp.Matrix.cross(omega, hook_offset_world)
    acc_hook = acc + sp.Matrix.cross(alpha, hook_offset_world) + sp.Matrix.cross(
        omega, sp.Matrix.cross(omega, hook_offset_world))
    zeta = sp.Matrix([sp.sin(theta_rope) * sp.cos(phi_rope),
                    sp.sin(theta_rope) * sp.sin(phi_rope), sp.cos(theta_rope)])
    zeta_dot = sp.Matrix([-sp.sin(theta_rope) * sp.sin(phi_rope) * theta_rope_dot + sp.cos(theta_rope) * sp.cos(phi_rope) * phi_rope_dot,
                        sp.sin(phi_rope) * sp.cos(theta_rope) * theta_rope_dot + sp.cos(phi_rope) * sp.sin(theta_rope) * phi_rope_dot, -sp.sin(theta_rope) * theta_rope_dot])
    zeta_ddot = sp.Matrix([[-sp.sin(phi_rope)*sp.sin(theta_rope)*phi_rope_ddot - 2*sp.sin(phi_rope)*sp.cos(theta_rope)*phi_rope_dot*theta_rope_dot - sp.sin(theta_rope)*sp.cos(phi_rope)*phi_rope_dot**2 - sp.sin(theta_rope)*sp.cos(phi_rope)*theta_rope_dot**2 + sp.cos(phi_rope)*sp.cos(theta_rope)*theta_rope_ddot], [-sp.sin(phi_rope)*sp.sin(
        theta_rope)*phi_rope_dot**2 - sp.sin(phi_rope)*sp.sin(theta_rope)*theta_rope_dot**2 + sp.sin(phi_rope)*sp.cos(theta_rope)*theta_rope_ddot + sp.sin(theta_rope)*sp.cos(phi_rope)*phi_rope_ddot + 2*sp.cos(phi_rope)*sp.cos(theta_rope)*phi_rope_dot*theta_rope_dot], [-(sp.sin(theta_rope)*theta_rope_ddot + sp.cos(theta_rope)*theta_rope_dot**2)]])
    pos_obj = pos_hook + l * zeta
    vel_obj = vel_hook + l * zeta_dot
    acc_obj = acc_hook + l * zeta_ddot
    f_rope = f_rope_norm * zeta
    # define action
    thrust = sp.Symbol("thrust")
    thrust_local = sp.Matrix([0, 0, thrust])
    thrust_world = rotmat @ thrust_local
    torque = sp.Matrix(sp.symbols('torque_x torque_y torque_z')).reshape(3, 1)
    action = [thrust, torque[0], torque[1], torque[2]]

    # newton's law (9 equations)
    # quadrotor
    eq_quad_pos = sp.Matrix([0, 0, -m*g]) + f_rope + thrust_world - m * acc
    eq_quad_rot = torque + \
        sp.Matrix.cross(hook_offset_world, f_rope) - \
        sp.Matrix.cross(omega, I @ omega) - I @ alpha
    # object
    eq_obj_pos = -f_rope + sp.Matrix([0, 0, -mo*g]) - mo * acc_obj

    # expand all equations
    eqs = []
    for i in range(3):
        eqs.append(eq_quad_pos[i].expand())
        eqs.append(eq_quad_rot[i].expand())
        eqs.append(eq_obj_pos[i].expand())
    unknowns = [acc[0], acc[1], acc[2], alpha[0], alpha[1],
                alpha[2], theta_rope_ddot, phi_rope_ddot, f_rope_norm]

    # Solve for the acceleration
    A_taut_dyn = sp.zeros(9, 9)
    b_taut_dyn = sp.zeros(9, 1)
    for i in range(9):
        for j in range(9):
            A_taut_dyn[i, j] = eqs[i].coeff(unknowns[j])
        b_taut_dyn[i] = -eqs[i].subs([(unknowns[j], 0) for j in range(9)])

    # Define matrix function
    A_taut_dyn_func = sp.lambdify(params + states + action, A_taut_dyn, "jax")
    b_taut_dyn_func = sp.lambdify(params + states + action, b_taut_dyn, "jax")

    # Define other dynamic variable functions
    pos_hook_func = sp.lambdify(params + states + states_dot + action, pos_hook, "jax")
    vel_hook_func = sp.lambdify(params + states + states_dot + action, vel_hook, "jax")
    pos_obj_func = sp.lambdify(params + states + states_dot + action, pos_obj, "jax")
    vel_obj_func = sp.lambdify(params + states + states_dot + action, vel_obj, "jax")
    f_rope_func = sp.lambdify(params + states + states_dot + action, f_rope, "jax")
    zeta_func = sp.lambdify(params + states + states_dot + action, zeta, "jax")
    zeta_dot_func = sp.lambdify(params + states + states_dot + action, zeta_dot, "jax")

    def taut_dynamics_3d(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D):
        params = [env_params.g, env_params.m, env_params.mo, env_params.l, env_params.I[0, 0], env_params.I[0, 1], env_params.I[0, 2], env_params.I[1, 0], env_params.I[1, 1], env_params.I[1, 2],
                    env_params.I[2, 0], env_params.I[2, 1], env_params.I[2, 2], env_params.hook_offset[0], env_params.hook_offset[1], env_params.hook_offset[2]]
        states = [
            env_state.pos[0], env_state.pos[1], env_state.pos[2],
            env_state.vel[0], env_state.vel[1], env_state.vel[2],
            env_state.quat[0], env_state.quat[1], env_state.quat[2], env_state.quat[3],
            env_state.omega[0], env_state.omega[1], env_state.omega[2],
            env_state.theta_rope, env_state.theta_rope_dot, env_state.phi_rope, env_state.phi_rope_dot
        ]
        action = [env_action.thrust, env_action.torque[0], env_action.torque[1], env_action.torque[2]]
        A = A_taut_dyn_func(*params, *states, *action)
        b = b_taut_dyn_func(*params, *states, *action)
        A_inv = jnp.linalg.inv(A)
        states_dot = jnp.dot(A_inv, b).squeeze()
        acc_x, acc_y, acc_z, alpha_x, alpha_y, alpha_z, theta_rope_ddot, phi_rope_ddot, f_rope_norm = states_dot
        acc = jnp.array([acc_x, acc_y, acc_z])
        alpha = jnp.array([alpha_x, alpha_y, alpha_z])

        # calculate updated state variables
        new_vel = env_state.vel + acc * env_params.dt
        new_pos = env_state.pos + new_vel * env_params.dt
        new_omega = env_state.omega + alpha * env_params.dt
        new_quat = geom.integrate_quat(env_state.quat, new_omega, env_params.dt)
        new_theta_rope_dot = env_state.theta_rope_dot + theta_rope_ddot * env_params.dt
        new_theta_rope = env_state.theta_rope + new_theta_rope_dot * env_params.dt
        new_phi_rope_dot = angle_normalize(env_state.phi_rope_dot + phi_rope_ddot * env_params.dt)
        new_phi_rope = angle_normalize(env_state.phi_rope + new_phi_rope_dot * env_params.dt)

        # Update states list
        states_new = [
            new_pos[0], new_pos[1], new_pos[2],
            new_vel[0], new_vel[1], new_vel[2],
            new_quat[0], new_quat[1], new_quat[2], new_quat[3],
            new_omega[0], new_omega[1], new_omega[2],
            new_theta_rope, new_theta_rope_dot, new_phi_rope, new_phi_rope_dot
        ]

        # Compute other state variables
        time = env_state.time + 1
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]

        # replace state
        env_state = env_state.replace(
            pos=new_pos, vel=new_vel, quat=new_quat, omega=new_omega,
            theta_rope=new_theta_rope, theta_rope_dot=new_theta_rope_dot,
            phi_rope=new_phi_rope, phi_rope_dot=new_phi_rope_dot, 
            pos_tar = pos_tar, vel_tar = vel_tar, 
            pos_hook = pos_hook_func(*params, *states_new, *states_dot, *action).squeeze(),
            vel_hook = vel_hook_func(*params, *states_new, *states_dot, *action).squeeze(),
            pos_obj = pos_obj_func(*params, *states_new, *states_dot, *action).squeeze(),
            vel_obj = vel_obj_func(*params, *states_new, *states_dot, *action).squeeze(),
            f_rope_norm = f_rope_norm, 
            f_rope = f_rope_func(*params, *states_new, *states_dot, *action).squeeze(),
            l_rope = env_params.l,
            zeta = zeta_func(*params, *states_new, *states_dot, *action).squeeze(),
            zeta_dot = zeta_dot_func(*params, *states_new, *states_dot, *action).squeeze(),
            last_thrust = env_action.thrust,
            last_torque = env_action.torque,
            time = time
        )
        
        return env_state
    
    return taut_dynamics_3d


def quat2rot(quat):
    x, y, z, w = quat
    return sp.Matrix([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
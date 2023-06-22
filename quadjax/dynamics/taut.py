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

from adaptive_control_gym.envs.jax_env.dynamics.utils import angle_normalize, EnvParams, EnvState, Action


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
    # y displacement of the hook from the quadrotor center
    delta_yh = sp.Symbol("delta_yh")
    # z displacement of the hook from the quadrotor center
    delta_zh = sp.Symbol("delta_zh")
    params = [m, I, g, l, mo, delta_yh, delta_zh]
    action = [thrust, tau]

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
    states = [y, z, theta, phi, y_dot, z_dot, theta_dot, phi_dot]
    states_val = [y, z, theta, phi, y_dot_val,
                  z_dot_val, theta_dot_val, phi_dot_val]
    states_dot = [y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope]
    states_dot_val = [y_ddot_val, z_ddot_val,
                      theta_ddot_val, phi_ddot_val, f_rope]

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

    # Define inertial reference frame
    N = ReferenceFrame("N")
    N_origin = Point('N_origin')
    A = N.orientnew("A", "Axis", [theta, N.x])
    B = A.orientnew("B", "Axis", [phi, A.x])

    # Define point
    drone = Point("drone")
    drone.set_pos(N_origin, y * N.y + z * N.z)
    hook = drone.locatenew("hook", delta_yh * A.y + delta_zh * A.z)
    obj = hook.locatenew("obj", -l * B.z)
    drone.set_vel(N, y_dot * N.y + z_dot * N.z)

    # Inertia
    inertia_quadrotor = inertia(N, I, 0, 0)
    quadrotor = RigidBody("quadrotor", drone, A, m, (inertia_quadrotor, drone))
    obj_particle = Particle("obj_particle", obj, mo)

    # Newton's law
    eq_quad_y = -thrust * sp.sin(theta) + f_rope_y - m * y_ddot
    eq_quad_z = thrust * sp.cos(theta) + f_rope_z - m * g - m * z_ddot
    eq_quad_theta = tau + delta_yh_global * f_rope_z - \
        delta_zh_global * f_rope_y - I * theta_ddot
    eq_obj_y = -f_rope_y - mo * y_obj_ddot
    eq_obj_z = -f_rope_z - mo * g - mo * z_obj_ddot

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

    # Solve for tion
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
    # lambda some other obs functions
    y_hook_func = sp.lambdify(params + states_val, obs_eqs[4], "jax")
    z_hook_func = sp.lambdify(params + states_val, obs_eqs[5], "jax")
    y_hook_dot_func = sp.lambdify(params + states_val, obs_eqs[6], "jax")
    z_hook_dot_func = sp.lambdify(params + states_val, obs_eqs[7], "jax")

    # dynamics (params, states) -> states_dot
    def taut_dynamics(env_params: EnvParams, env_state: EnvState, env_action: Action):
        params = [env_params.m, env_params.I, env_params.g, env_params.l,
                  env_params.mo, env_params.delta_yh, env_params.delta_zh]
        states = [env_state.y, env_state.z, env_state.theta, env_state.phi,
                  env_state.y_dot, env_state.z_dot, env_state.theta_dot, env_state.phi_dot]
        action = [env_action.thrust, env_action.tau]
        A = A_taut_dyn_func(*params, *states, *action)
        b = b_taut_dyn_func(*params, *states, *action)
        states_dot = jnp.linalg.solve(A, b).squeeze()
        y_ddot, z_ddot, theta_ddot, phi_ddot, f_rope = states_dot

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

        # Update states list
        states = [new_y, new_z, new_theta, new_phi,
                  new_y_dot, new_z_dot, new_theta_dot, new_phi_dot]

        # Compute other state variables
        y_obj, z_obj, y_obj_dot, z_obj_dot, y_hook, z_hook, y_hook_dot, z_hook_dot, f_rope_y, f_rope_z = obs_eqs_func(
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
            f_rope_y=f_rope_y,
            f_rope_z=f_rope_z,
            f_rope=f_rope,
            l_rope=env_params.l,
            last_thrust=env_action.thrust,
            last_tau=env_action.tau,
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
    I = sp.MatrixSymbol("I", 3, 3)
    hook_offset = sp.MatrixSymbol("hook_offset", 3, 1)
    # define dynamic state
    pos = sp.MatrixSymbol("pos", 3, 1)
    vel = sp.MatrixSymbol("vel", 3, 1)
    quat = sp.MatrixSymbol("quat", 4, 1)
    omega = sp.MatrixSymbol("omega", 3, 1)
    theta_rope = sp.Symbol("theta_rope")
    theta_rope_dot = sp.Symbol("theta_rope_dot")
    phi_rope = sp.Symbol("phi_rope")
    phi_rope_dot = sp.Symbol("phi_rope_dot")
    # dynamic variables to solve
    acc = sp.MatrixSymbol("acc", 3, 1)
    alpha = sp.MatrixSymbol("alpha", 3, 1)
    theta_rope_ddot = sp.Symbol("theta_rope_ddot")
    phi_rope_ddot = sp.Symbol("phi_rope_ddot")
    f_rope_norm = sp.Symbol("f_rope_norm")
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
    thrust_local = sp.Matrix([0, 0, f_rope_norm])
    thrust_world = rotmat @ thrust_local
    # define action
    thrust = sp.Symbol("thrust")
    torque = sp.MatrixSymbol("torque", 3, 1)

    # newton's law (9 equations)
    # quadrotor
    eq_quad_pos = sp.Matrix([0, 0, -m*g]) + f_rope + thrust_world - m * acc
    eq_quad_rot = torque + \
        sp.Matrix.cross(hook_offset_world, f_rope) - \
        sp.Matrix.cross(omega, I @ omega) - I @ alpha
    # object
    eq_obj_pos = -f_rope + sp.Matrix([0, 0, -mo*g]) - mo * acc_obj

    # Solve for the acceleration
    A_taut_dyn = sp.zeros(9, 9)
    b_taut_dyn = sp.zeros(9, 1)
    return None


def quat2rot(quat):
    x, y, z, w = quat
    return sp.Matrix([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
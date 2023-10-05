import sympy as sp
import jax
from jax import numpy as jnp

from quadjax.dynamics.utils import angle_normalize
from quadjax.dynamics.dataclass import EnvParams2D, EnvState2D, Action2D, EnvParams3D, EnvState3D, Action3D


def get_dynamic_transfer():

    # Define the variables
    I, m, l, mo, delta_yh, delta_zh, Imp, t = \
        sp.symbols('I m l mo delta_yh delta_zh Imp t')
    theta, phi, y, z, \
        loose_y_dot, loose_z_dot, loose_theta_dot, loose_y_obj_dot, loose_z_obj_dot, \
        taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot = \
        sp.symbols('theta phi y z \
                        loose_y_dot loose_z_dot loose_theta_dot loose_y_obj_dot loose_z_obj_dot \
                        taut_y_dot taut_z_dot taut_theta_dot taut_phi_dot')

    # Define velocities
    # get the linear velocity of the object in the world frame
    taut_xyz_obj_dot = sp.Matrix([0, taut_y_dot, taut_z_dot]) + \
        sp.Matrix([taut_theta_dot, 0, 0]).cross(sp.Matrix([0, delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta), delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)])) + \
        sp.Matrix([taut_phi_dot+taut_theta_dot, 0, 0]).cross(sp.Matrix([0,
                                                                        l * sp.sin(phi+theta), -l * sp.cos(phi+theta)]))
    taut_y_obj_dot, taut_z_obj_dot = taut_xyz_obj_dot[1], taut_xyz_obj_dot[2]

    # linear momentum balance for the quadrotor
    lin_quad_y = m * loose_y_dot + Imp * sp.sin(theta+phi) - m * taut_y_dot
    lin_quad_z = m * loose_z_dot - Imp * sp.cos(theta+phi) - m * taut_z_dot

    # linear momentum balance for the object
    lin_obj_y = sp.expand(mo * loose_y_obj_dot - Imp *
                          sp.sin(theta+phi) - mo * taut_y_obj_dot)
    lin_obj_z = sp.expand(mo * loose_z_obj_dot + Imp *
                          sp.cos(theta+phi) - mo * taut_z_obj_dot)

    # angular momentum balance for the quadrotor
    M = delta_yh * (- Imp * sp.cos(phi+theta)) - \
        delta_zh * (Imp * sp.sin(phi+theta))
    ang_quad = I * loose_theta_dot + M - I * taut_theta_dot

    equations = [lin_quad_y, lin_quad_z, lin_obj_y, lin_obj_z, ang_quad]
    num_equations = len(equations)
    taut_state = [taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp]

    Atrans = sp.Matrix(num_equations, len(taut_state),
                       lambda i, j: equations[i].coeff(taut_state[j]))
    btrans = -sp.Matrix(num_equations, 1, lambda i,
                        j: equations[i].subs([(var, 0) for var in taut_state]))

    params = [I, m, l, mo, delta_yh, delta_zh]
    loose_state = [theta, phi, y, z, loose_y_dot, loose_z_dot,
                   loose_theta_dot, loose_y_obj_dot, loose_z_obj_dot]

    # lambdify the matrix
    Atrans_func = sp.lambdify(params+loose_state, Atrans, modules='jax')
    btrans_func = sp.lambdify(params+loose_state, btrans, modules='jax')

    # Define the function to compute A^-1 (x-b)
    def loose2taut_transfer(env_params: EnvParams2D, loose_state: EnvState2D, loose2taut: bool):
        params = [env_params.I, env_params.m, env_params.l,
                  env_params.mo, env_params.delta_yh, env_params.delta_zh]
        loose_state_values = [loose_state.theta, loose_state.phi, loose_state.y, loose_state.z, loose_state.y_dot,
                              loose_state.z_dot, loose_state.theta_dot, loose_state.y_obj_dot, loose_state.z_obj_dot]
        A_val = Atrans_func(*params, *loose_state_values)
        b_val = btrans_func(*params, *loose_state_values)
        taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp = jnp.linalg.solve(
            A_val, b_val).squeeze()

        new_y_dot = taut_y_dot * loose2taut + \
            loose_state.y_dot * (1 - loose2taut)
        new_z_dot = taut_z_dot * loose2taut + \
            loose_state.z_dot * (1 - loose2taut)
        new_theta_dot = taut_theta_dot * loose2taut + \
            loose_state.theta_dot * (1 - loose2taut)
        new_phi_dot = taut_phi_dot * loose2taut + \
            loose_state.phi_dot * (1 - loose2taut)
        new_l_rope = env_params.l * loose2taut + \
            loose_state.l_rope * (1 - loose2taut)

        loose_state = loose_state.replace(
            y_dot=new_y_dot,
            z_dot=new_z_dot,
            theta_dot=new_theta_dot,
            phi_dot=new_phi_dot,
            l_rope=new_l_rope
        )

        return loose_state

    def dynamic_transfer(env_params: EnvParams2D, loose_state: EnvState2D, taut_state: EnvState2D, old_loose_state: bool):
        new_loose_state = loose_state.l_rope < (
            env_params.l - env_params.rope_taut_therehold)
        taut2loose = (taut_state.f_rope < 0.0) & (~old_loose_state)
        loose2taut = (~new_loose_state) & (old_loose_state)

        # taut2loose dynamics
        new_taut_l_rope = taut_state.l_rope - taut2loose * \
            env_params.rope_taut_therehold * 2.0
        taut_state = taut_state.replace(l_rope=new_taut_l_rope)

        # loose2taut dynamics
        loose_state = loose2taut_transfer(env_params, loose_state, loose2taut)

        # use loose_state when old_loose_state is True, else use taut_state
        new_state = {}
        for k in loose_state.__dict__.keys():
            new_state[k] = jnp.where(old_loose_state, loose_state.__dict__[
                                     k], taut_state.__dict__[k])

        loose_state = loose_state.replace(**new_state)

        return loose_state

    return dynamic_transfer


def get_dynamic_transfer_3d():
    # define parameter symbols
    g, m, mo, l = sp.symbols("g m mo l")
    I = sp.Matrix(sp.symbols("I_11 I_12 I_13 I_21 I_22 I_23 I_31 I_32 I_33")).reshape(
        3, 3
    )
    hook_offset = sp.Matrix(
        sp.symbols("hook_offset_x hook_offset_y hook_offset_z")
    ).reshape(3, 1)
    params = [
        g,
        m,
        mo,
        l,
        I[0, 0],
        I[0, 1],
        I[0, 2],
        I[1, 0],
        I[1, 1],
        I[1, 2],
        I[2, 0],
        I[2, 1],
        I[2, 2],
        hook_offset[0],
        hook_offset[1],
        hook_offset[2],
    ]
    # define dynamic state
    quat = sp.Matrix(sp.symbols("quat_x quat_y quat_z quat_w")).reshape(4, 1)
    zeta = sp.Matrix(sp.symbols("zeta_x zeta_y zeta_z")).reshape(3, 1)
    loose_vel = sp.Matrix(sp.symbols(
        "loose_vel_x loose_vel_y loose_vel_z")).reshape(3, 1)
    loose_omega = sp.Matrix(sp.symbols(
        "loose_omega_x loose_omega_y loose_omega_z")).reshape(3, 1)
    loose_vel_obj = sp.Matrix(sp.symbols(
        "loose_vel_obj_x loose_vel_obj_y loose_vel_obj_z")).reshape(3, 1)
    loose_state = [
        quat[0], quat[1], quat[2], quat[3], 
        zeta[0], zeta[1], zeta[2], 
        loose_vel[0], loose_vel[1], loose_vel[2], 
        loose_omega[0], loose_omega[1], loose_omega[2], 
        loose_vel_obj[0], loose_vel_obj[1], loose_vel_obj[2],
    ]
    
    taut_vel = sp.Matrix(sp.symbols(
        "taut_vel_x taut_vel_y taut_vel_z")).reshape(3, 1)
    taut_omega = sp.Matrix(sp.symbols(
        "taut_omega_x taut_omega_y taut_omega_z")).reshape(3, 1)
    taut_zeta_dot = sp.Matrix(sp.symbols(
        "zeta_dot_x zeta_dot_y zeta_dot_z")).reshape(3, 1)
    hook_offset_world = quat2rot(quat) @ hook_offset
    taut_vel_hook = taut_vel + sp.Matrix.cross(taut_omega, hook_offset_world)
    taut_vel_obj = taut_vel_hook + l * taut_zeta_dot

    # define the rope impulse
    imp = sp.Symbol("imp")
    Imp = imp * zeta

    # state to solve
    taut_state = [
        taut_vel[0], taut_vel[1], taut_vel[2], 
        taut_omega[0], taut_omega[1], taut_omega[2], 
        taut_zeta_dot[0], taut_zeta_dot[1], taut_zeta_dot[2], 
        imp
    ]

    quat_conj = sp.Matrix([-quat[0], -quat[1], -quat[2], quat[3]])
    Imp_local = quat2rot(quat_conj) @ Imp

    # Moment balance
    # for the quadrotor
    lin_quad = m * loose_vel + Imp - m * taut_vel
    ang_quad = I @ loose_omega + hook_offset.cross(Imp_local) - I @ taut_omega
    # for the object
    lin_obj = mo * loose_vel_obj - Imp - mo * taut_vel_obj

    # all equations
    eqs = []
    for ee in [lin_quad, ang_quad, lin_obj]:
        for i in range(3):
            eqs.append(ee[i].expand()) # NOTE expand is important!
    eq_zeta = zeta[0] * taut_zeta_dot[0] + zeta[1] * taut_zeta_dot[1] + zeta[2] * taut_zeta_dot[2]
    eqs.append(eq_zeta)
    
    num_equations = len(eqs)
    Atrans = sp.Matrix(num_equations, len(taut_state),
                       lambda i, j: eqs[i].coeff(taut_state[j]))
    btrans = -sp.Matrix(num_equations, 1, lambda i,
                        j: eqs[i].subs([(var, 0) for var in taut_state]))

    # lambdify the matrix
    Atrans_func = sp.lambdify(params+loose_state, Atrans, modules='jax')
    btrans_func = sp.lambdify(params+loose_state, btrans, modules='jax')

    def loose2taut_transfer_3d(env_params: EnvParams3D, loose_state: EnvState3D, loose2taut: bool):
        params = [
            env_params.g,
            env_params.m,
            env_params.mo,
            env_params.l,
            env_params.I[0, 0],
            env_params.I[0, 1],
            env_params.I[0, 2],
            env_params.I[1, 0],
            env_params.I[1, 1],
            env_params.I[1, 2],
            env_params.I[2, 0],
            env_params.I[2, 1],
            env_params.I[2, 2],
            env_params.hook_offset[0],
            env_params.hook_offset[1],
            env_params.hook_offset[2],
        ]
        loose_state_values = [
            *loose_state.quat,
            *loose_state.zeta,
            *loose_state.vel,
            *loose_state.omega,
            *loose_state.vel_obj
        ]

        A_val = Atrans_func(*params, *loose_state_values)
        b_val = btrans_func(*params, *loose_state_values)
        taut_state_values = jnp.linalg.solve(
            A_val, b_val).squeeze()
        taut_vel = taut_state_values[:3]
        taut_omega = taut_state_values[3:6]
        taut_zeta_dot = taut_state_values[6:9]
        imp = taut_state_values[9]

        new_vel = taut_vel * loose2taut + \
            loose_state.vel * (1 - loose2taut)
        new_omega = taut_omega * loose2taut + \
            loose_state.omega * (1 - loose2taut)
        new_zeta_dot = taut_zeta_dot * loose2taut + \
            loose_state.zeta_dot * (1 - loose2taut)
        new_l_rope = env_params.l * loose2taut + \
            loose_state.l_rope * (1 - loose2taut)

        loose_state = loose_state.replace(
            vel = new_vel,
            omega = new_omega,
            zeta_dot = new_zeta_dot,
            l_rope = new_l_rope
        )

        return loose_state
    
    def dynamic_transfer_3d(env_params: EnvParams3D, loose_state: EnvState3D, taut_state: EnvState3D, old_loose_state: bool):
        new_loose_state = loose_state.l_rope < (
            env_params.l - env_params.rope_taut_therehold)
        taut2loose = (taut_state.f_rope_norm < 0.0) & (~old_loose_state)
        loose2taut = (~new_loose_state) & (old_loose_state)

        # taut2loose dynamics
        new_taut_l_rope = taut_state.l_rope - taut2loose * \
            env_params.rope_taut_therehold * 2.0
        taut_state = taut_state.replace(l_rope=new_taut_l_rope)

        # loose2taut dynamics
        loose_state = loose2taut_transfer_3d(env_params, loose_state, loose2taut)

        # use loose_state when old_loose_state is True, else use taut_state
        new_state = {}
        keys = loose_state.__dict__.keys()
        # exclude control_params
        keys = [k for k in keys if k != 'control_params']
        for k in keys:
            # jax.debug.print('[DEBUG] {x}',x=old_loose_state)
            # jax.debug.print('[DEBUG] {k} loose {x.shape}', k=k, x=jnp.asarray(loose_state.__dict__[
            #                          k]))
            # jax.debug.print('[DEBUG] {k} taut {x.shape}', k=k, x=jnp.asarray(taut_state.__dict__[
            #                             k]))
            new_state[k] = jnp.where(old_loose_state, loose_state.__dict__[
                                     k], taut_state.__dict__[k])

        loose_state = loose_state.replace(**new_state)

        return loose_state

    return dynamic_transfer_3d

def quat2rot(q):  # q in [x, y, z, w]
    x, y, z, w = q
    return sp.Matrix([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]
    ])
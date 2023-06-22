import sympy as sp
from jax import numpy as jnp

from adaptive_control_gym.envs.jax_env.dynamics.utils import angle_normalize, EnvParams, EnvState, Action

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
    taut_xyz_obj_dot = sp.Matrix([0,taut_y_dot, taut_z_dot]) + \
                        sp.Matrix([taut_theta_dot,0,0]).cross(sp.Matrix([0,delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta), delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)])) + \
                            sp.Matrix([taut_phi_dot+taut_theta_dot, 0,0]).cross(sp.Matrix([0,l * sp.sin(phi+theta), -l * sp.cos(phi+theta)])) 
    taut_y_obj_dot, taut_z_obj_dot = taut_xyz_obj_dot[1], taut_xyz_obj_dot[2]

    # linear momentum balance for the quadrotor
    lin_quad_y = m * loose_y_dot + Imp * sp.sin(theta+phi) - m * taut_y_dot
    lin_quad_z = m * loose_z_dot - Imp * sp.cos(theta+phi) - m * taut_z_dot

    # linear momentum balance for the object
    lin_obj_y = sp.expand(mo * loose_y_obj_dot - Imp * sp.sin(theta+phi) - mo * taut_y_obj_dot)
    lin_obj_z = sp.expand(mo * loose_z_obj_dot + Imp * sp.cos(theta+phi) - mo * taut_z_obj_dot)

    # angular momentum balance for the quadrotor
    M = delta_yh * (- Imp * sp.cos(phi+theta)) - delta_zh * (Imp * sp.sin(phi+theta))
    ang_quad = I * loose_theta_dot + M - I * taut_theta_dot

    equations = [lin_quad_y, lin_quad_z, lin_obj_y, lin_obj_z, ang_quad]
    num_equations = len(equations)
    taut_state = [taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp]

    Atrans = sp.Matrix(num_equations, len(taut_state), lambda i, j: equations[i].coeff(taut_state[j]))
    btrans = -sp.Matrix(num_equations, 1, lambda i, j: equations[i].subs([(var, 0) for var in taut_state]))

    params = [I, m, l, mo, delta_yh, delta_zh]
    loose_state = [theta, phi, y, z, loose_y_dot, loose_z_dot, loose_theta_dot, loose_y_obj_dot, loose_z_obj_dot]

    # lambdify the matrix
    Atrans_func = sp.lambdify(params+loose_state, Atrans, modules='jax')
    btrans_func = sp.lambdify(params+loose_state, btrans, modules='jax')

    # Define the function to compute A^-1 (x-b)
    def loose2taut_transfer(env_params: EnvParams, loose_state: EnvState, loose2taut:bool):
        params = [env_params.I, env_params.m, env_params.l, env_params.mo, env_params.delta_yh, env_params.delta_zh]
        loose_state_values = [loose_state.theta, loose_state.phi, loose_state.y, loose_state.z, loose_state.y_dot, loose_state.z_dot, loose_state.theta_dot, loose_state.y_obj_dot, loose_state.z_obj_dot]
        A_val = Atrans_func(*params, *loose_state_values)
        b_val = btrans_func(*params, *loose_state_values)
        taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp = jnp.linalg.solve(A_val, b_val).squeeze()

        new_y_dot = taut_y_dot * loose2taut + loose_state.y_dot * (1 - loose2taut)
        new_z_dot = taut_z_dot * loose2taut + loose_state.z_dot * (1 - loose2taut)
        new_theta_dot = taut_theta_dot * loose2taut + loose_state.theta_dot * (1 - loose2taut)
        new_phi_dot = taut_phi_dot * loose2taut + loose_state.phi_dot * (1 - loose2taut)
        new_l_rope = env_params.l * loose2taut + loose_state.l_rope * (1 - loose2taut)

        loose_state = loose_state.replace(
            y_dot=new_y_dot,
            z_dot=new_z_dot,
            theta_dot=new_theta_dot,
            phi_dot=new_phi_dot,
            l_rope=new_l_rope
        )

        return loose_state

    def dynamic_transfer(env_params:EnvParams, loose_state:EnvState, taut_state:EnvState, old_loose_state:bool):
        new_loose_state = loose_state.l_rope < (env_params.l - env_params.rope_taut_therehold)
        taut2loose = (taut_state.f_rope < 0.0) & (~old_loose_state)
        loose2taut = (~new_loose_state) & (old_loose_state)

        # taut2loose dynamics
        new_taut_l_rope = taut_state.l_rope - taut2loose * env_params.rope_taut_therehold * 2.0
        taut_state = taut_state.replace(l_rope=new_taut_l_rope)

        # loose2taut dynamics
        loose_state = loose2taut_transfer(env_params, loose_state, loose2taut)

        # use loose_state when old_loose_state is True, else use taut_state
        new_state = {}
        for k in loose_state.__dict__.keys():
            new_state[k] = jnp.where(old_loose_state, loose_state.__dict__[k], taut_state.__dict__[k])

        loose_state = loose_state.replace(**new_state)

        return loose_state
    
    return dynamic_transfer
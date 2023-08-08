import sympy as sp
from jax import numpy as jnp
from typing import Tuple
from quadjax.dynamics.utils import angle_normalize
from quadjax.dynamics.dataclass import EnvParams, EnvState, Action

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
    def loose2taut_transfer(env_params: EnvParams, loose_state: EnvState, loose2taut: bool):
        if ~loose2taut:
            params = [env_params.I, env_params.m, env_params.l, env_params.mo, env_params.delta_yh, env_params.delta_zh]
            loose_state_values = [loose_state.theta, loose_state.phi, loose_state.y, loose_state.z, loose_state.y_dot, loose_state.z_dot, loose_state.theta_dot, loose_state.y_obj_dot, loose_state.z_obj_dot]
        else:
            params = [env_params.I, env_params.m, env_params.l, env_params.mo, env_params.delta_yh2, env_params.delta_zh2]
            loose_state_values = [loose_state.theta2, loose_state.phi2, loose_state.y2, loose_state.z2, loose_state.y2_dot, loose_state.z2_dot, loose_state.theta2_dot, loose_state.y_obj_dot, loose_state.z_obj_dot]
        
        A_val = Atrans_func(*params, *loose_state_values)
        b_val = btrans_func(*params, *loose_state_values)
        taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp = jnp.linalg.solve(A_val, b_val).squeeze()

        new_y_dot = loose_state.y_dot * loose2taut + taut_y_dot * (1 - loose2taut)
        new_z_dot = loose_state.z_dot * loose2taut + taut_z_dot * (1 - loose2taut)
        new_theta_dot = loose_state.theta_dot * loose2taut + taut_theta_dot * (1 - loose2taut)
        new_phi_dot = loose_state.phi_dot * loose2taut + taut_phi_dot * (1 - loose2taut)
        new_l_rope = loose_state.l_rope * loose2taut +  env_params.l * (1 - loose2taut)
            

        if ~loose2taut:
            loose_state = loose_state.replace(
                y_dot=new_y_dot,
                z_dot=new_z_dot,
                theta_dot=new_theta_dot,
                phi_dot=new_phi_dot,
                l_rope=new_l_rope
            )
        else:
            loose_state = loose_state.replace(
                y2_dot=new_y_dot,
                z2_dot=new_z_dot,
                theta2_dot=new_theta_dot,
                phi2_dot=new_phi_dot,
                l_rope2=new_l_rope
            )

        return loose_state

    # Define the function to compute A^-1 (x-b)
    def both_loose2taut_transfer(env_params: EnvParams, loose_state: EnvState):
        # # Define the variables
        # delta_yh2, delta_zh2, Imp2 = \
        #     sp.symbols('delta_yh2 delta_zh2 Imp2')
        # theta2, phi2, y2, z2, \
        #     loose_y2_dot, loose_z2_dot, loose_theta2_dot, \
        #     taut_y2_dot, taut_z2_dot, taut_theta2_dot, taut_phi2_dot= \
        #     sp.symbols('theta2 phi2 y2 z2 \
        #                     loose_y2_dot loose_z2_dot loose_theta2_dot \
        #                     taut_y2_dot taut_z2_dot taut_theta2_dot taut_phi2_dot')
        # Define the variables
        I, m, l, mo, delta_yh, delta_zh, Imp, delta_yh2, delta_zh2, Imp2, t = \
            sp.symbols('I m l mo delta_yh delta_zh Imp delta_yh2 delta_zh2 Imp2 t')
        theta, phi, y, z, \
            loose_y_dot, loose_z_dot, loose_theta_dot, loose_y_obj_dot, loose_z_obj_dot, \
            taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, \
        theta2, phi2, y2, z2, \
            loose_y2_dot, loose_z2_dot, loose_theta2_dot, \
            taut_y2_dot, taut_z2_dot, taut_theta2_dot, taut_phi2_dot= \
            sp.symbols('theta phi y z \
                            loose_y_dot loose_z_dot loose_theta_dot loose_y_obj_dot loose_z_obj_dot \
                            taut_y_dot taut_z_dot taut_theta_dot taut_phi_dot \
                            theta2 phi2 y2 z2 \
                            loose_y2_dot loose_z2_dot loose_theta2_dot \
                            taut_y2_dot taut_z2_dot taut_theta2_dot taut_phi2_dot')

        # Define velocities
        # get the linear velocity of the object in the world frame
        taut_xyz_obj_dot = sp.Matrix([0,taut_y_dot, taut_z_dot]) + \
                            sp.Matrix([taut_theta_dot,0,0]).cross(sp.Matrix([0,delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta), delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)])) + \
                                sp.Matrix([taut_phi_dot+taut_theta_dot, 0,0]).cross(sp.Matrix([0,l * sp.sin(phi+theta), -l * sp.cos(phi+theta)])) 
        taut_y_obj_dot, taut_z_obj_dot = taut_xyz_obj_dot[1], taut_xyz_obj_dot[2]

        # Define velocities
        # get the linear velocity of the object in the world frame
        # taut_xyz_obj_dot = sp.Matrix([0,taut_y_dot, taut_z_dot]) + \
        #                     sp.Matrix([taut_theta_dot,0,0]).cross(sp.Matrix([0,delta_yh * sp.cos(theta) - delta_zh * sp.sin(theta), delta_yh * sp.sin(theta) + delta_zh * sp.cos(theta)])) + \
        #                         sp.Matrix([taut_phi_dot+taut_theta_dot, 0,0]).cross(sp.Matrix([0,l * sp.sin(phi+theta), -l * sp.cos(phi+theta)])) 
        # taut_y_obj_dot, taut_z_obj_dot = taut_xyz_obj_dot[1], taut_xyz_obj_dot[2]

        ##### Both loose2taut #####
        # linear momentum balance for the quadrotor
        lin_quad_y = m * loose_y_dot + Imp * sp.sin(theta+phi) - m * taut_y_dot
        lin_quad_z = m * loose_z_dot - Imp * sp.cos(theta+phi) - m * taut_z_dot

        lin_quad2_y = m * loose_y2_dot + Imp2 * sp.sin(theta2+phi2) - m * taut_y2_dot
        lin_quad2_z = m * loose_z2_dot - Imp2 * sp.cos(theta2+phi2) - m * taut_z2_dot

        # linear momentum balance for the object
        lin_obj_y = sp.expand(mo * loose_y_obj_dot - Imp * sp.sin(theta+phi)- Imp2 * sp.sin(theta2+phi2) - mo * taut_y_obj_dot)
        lin_obj_z = sp.expand(mo * loose_z_obj_dot + Imp * sp.cos(theta+phi)+ Imp2 * sp.cos(theta2+phi2) - mo * taut_z_obj_dot)

        # angular momentum balance for the quadrotor
        M = delta_yh * (- Imp * sp.cos(phi+theta)) - delta_zh * (Imp * sp.sin(phi+theta))
        ang_quad = I * loose_theta_dot + M - I * taut_theta_dot

        M2 = delta_yh2 * (- Imp2 * sp.cos(phi2+theta2)) - delta_zh2 * (Imp2 * sp.sin(phi2+theta2))
        ang_quad2 = I * loose_theta2_dot + M2 - I * taut_theta2_dot

        # geometric relation
        lin_quads_y = taut_y_dot - taut_theta_dot * (delta_yh * sp.sin(theta) + delta_zh *sp.cos(theta)) + l * sp.cos(theta + phi) * (taut_theta_dot + taut_phi_dot) - \
            taut_y2_dot + taut_theta2_dot * (delta_yh2 * sp.sin(theta2) + delta_zh2 *sp.cos(theta2)) - l * sp.cos(theta2 + phi2) * (taut_theta2_dot + taut_phi2_dot)

        lin_quads_z = taut_z_dot + taut_theta_dot * (delta_yh * sp.cos(theta) - delta_zh *sp.sin(theta)) + l * sp.sin(theta + phi) * (taut_theta_dot + taut_phi_dot) - \
            taut_z2_dot - taut_theta2_dot * (delta_yh2 * sp.cos(theta2) - delta_zh2 *sp.sin(theta2)) - l * sp.sin(theta2 + phi2) * (taut_theta2_dot + taut_phi2_dot)

        # equations = [lin_quad_y, lin_quad_z, lin_obj_y, lin_obj_z, ang_quad]
        equations = [lin_quad_y, lin_quad_z, lin_quad2_y, lin_quad2_z, lin_obj_y, lin_obj_z, ang_quad, ang_quad2, lin_quads_y, lin_quads_z]
        num_equations = len(equations)
        taut_state = [taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp, taut_y2_dot, taut_z2_dot, taut_theta2_dot, taut_phi2_dot, Imp2]

        #################################


        Atrans = sp.Matrix(num_equations, len(taut_state), lambda i, j: equations[i].coeff(taut_state[j]))
        btrans = -sp.Matrix(num_equations, 1, lambda i, j: equations[i].subs([(var, 0) for var in taut_state]))

        fun_params = [I, m, l, mo, delta_yh, delta_zh, delta_yh2, delta_zh2]
        fun_loose_state = [theta, phi, y, z, loose_y_dot, loose_z_dot, loose_theta_dot, loose_y_obj_dot, loose_z_obj_dot, theta2, phi2, y2, z2, loose_y2_dot, loose_z2_dot, loose_theta2_dot]

        # lambdify the matrix
        Atrans_func = sp.lambdify(fun_params+fun_loose_state, Atrans, modules='jax')
        btrans_func = sp.lambdify(fun_params+fun_loose_state, btrans, modules='jax')

        params = [env_params.I, env_params.m, env_params.l, env_params.mo, env_params.delta_yh, env_params.delta_zh, env_params.delta_yh2, env_params.delta_zh2]
        loose_state_values = [loose_state.theta, loose_state.phi, loose_state.y, loose_state.z, loose_state.y_dot, loose_state.z_dot, loose_state.theta_dot, loose_state.y_obj_dot, loose_state.z_obj_dot,
                            loose_state.theta2, loose_state.phi2, loose_state.y2, loose_state.z2, loose_state.y2_dot, loose_state.z2_dot, loose_state.theta2_dot]
        A_val = Atrans_func(*params, *loose_state_values)
        b_val = btrans_func(*params, *loose_state_values)
        # taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp = jnp.linalg.solve(A_val, b_val).squeeze()
        taut_y_dot, taut_z_dot, taut_theta_dot, taut_phi_dot, Imp, taut_y2_dot, taut_z2_dot, taut_theta2_dot, taut_phi2_dot, Imp2 = jnp.linalg.solve(A_val, b_val).squeeze()

        loose_state = loose_state.replace(
            y_dot=taut_y_dot,
            z_dot=taut_z_dot,
            theta_dot=taut_theta_dot,
            phi_dot=taut_phi_dot,
            l_rope=env_params.l,
            y2_dot=taut_y2_dot,
            z2_dot=taut_z2_dot,
            theta2_dot=taut_theta2_dot,
            phi2_dot=taut_phi2_dot,
            l_rope2=env_params.l
        )
        return loose_state

        
    def dynamic_transfer(env_params:EnvParams, loose_state:EnvState, taut_state:EnvState, loose_taut_state:EnvState, taut_loose_state:EnvState, old_loose_state: Tuple[bool,bool]):
        taut2loose1 = (taut_state.f_rope < 0.0) & (~old_loose_state[0])
        taut2loose2 = (taut_state.f_rope2 < 0.0) & (~old_loose_state[1])

        # old_tt,tl,lt,ll new_tt,tl,lt,tt
        # 定义改变状态量，4个状态之间的转移，共12种
        # tt2ll = taut2loose1 & taut2loose2
        # tt2tl = (~old_loose_state[0]) &  (~old_loose_state[1]) & (taut_state.f_rope >= 0.0) & (taut_state.f_rope2 < 0.0)
        # tt2lt = (~old_loose_state[0]) &  (~old_loose_state[1]) & (taut_state.f_rope < 0.0) & (taut_state.f_rope2 >= 0.0)

        lt2ll = (old_loose_state[0]) & (~old_loose_state[1]) & (loose_taut_state.f_rope2 < 0.0) & (loose_taut_state.l_rope < (env_params.l - env_params.rope_taut_threshold))
        tl2ll = (~old_loose_state[0]) & (old_loose_state[1]) & (taut_loose_state.f_rope < 0.0) & (taut_loose_state.l_rope < (env_params.l - env_params.rope_taut_threshold))

        lt2tl = (old_loose_state[0]) & (~old_loose_state[1]) & (loose_taut_state.f_rope2 < 0.0) & (loose_taut_state.l_rope < (env_params.l - env_params.rope_taut_threshold))
        lt2tt = (old_loose_state[0]) & (~old_loose_state[1]) & (loose_taut_state.f_rope2 >= 0.0) & (loose_taut_state.l_rope >= (env_params.l - env_params.rope_taut_threshold))

        tl2lt = (~old_loose_state[0]) & (old_loose_state[1]) & (taut_loose_state.f_rope < 0.0) & (taut_loose_state.l_rope2 >= (env_params.l - env_params.rope_taut_threshold))
        tl2tt = (~old_loose_state[0]) & (old_loose_state[1]) & (taut_loose_state.f_rope >= 0.0) & (taut_loose_state.l_rope2 >= (env_params.l - env_params.rope_taut_threshold))

        ll2lt = (old_loose_state[0]) & (old_loose_state[1]) & (loose_state.l_rope < (env_params.l - env_params.rope_taut_threshold)) & (loose_state.l_rope2 >= (env_params.l - env_params.rope_taut_threshold))
        ll2tl = (old_loose_state[0]) & (old_loose_state[1]) & (loose_state.l_rope >= (env_params.l - env_params.rope_taut_threshold)) & (loose_state.l_rope2 < (env_params.l - env_params.rope_taut_threshold))
        ll2tt = (old_loose_state[0]) & (old_loose_state[1]) & (loose_state.l_rope < (env_params.l - env_params.rope_taut_threshold)) & (loose_state.l_rope2 < (env_params.l - env_params.rope_taut_threshold))
        
        # taut2loose dynamics: both taut2loose, or one taut2loose
        new_taut_l_rope1 = taut_state.l_rope - taut2loose1 * env_params.rope_taut_threshold * 2.0
        new_taut_l_rope2 = taut_state.l_rope2 - taut2loose2 * env_params.rope_taut_threshold * 2.0
        taut_state = taut_state.replace(l_rope=new_taut_l_rope1, l_rope2=new_taut_l_rope2)

        new_lt_l_rope2 = loose_taut_state.l_rope2 - (lt2ll | lt2tl) * env_params.rope_taut_threshold * 2.0
        loose_taut_state = loose_taut_state.replace(l_rope2=new_lt_l_rope2)

        new_tl_l_rope = taut_loose_state.l_rope - (tl2ll | tl2lt) * env_params.rope_taut_threshold * 2.0
        taut_loose_state = taut_loose_state.replace(l_rope=new_tl_l_rope)

        # loose2taut dynamics
        # if ll2tt:
        #     loose_state = both_loose2taut_transfer(env_params, loose_state)
        # elif ll2tl:
        #     loose_state = loose2taut_transfer(env_params, loose_state, 0)
        # elif ll2lt:
        #     loose_state = loose2taut_transfer(env_params, loose_state, 1)
        # elif (lt2tl | lt2tt):
        #     loose_taut_state = loose2taut_transfer(env_params, loose_taut_state, 0)
        # elif (tl2lt | tl2tt):
        #     taut_loose_state = loose2taut_transfer(env_params, taut_loose_state, 1)
        
        l1 = old_loose_state[0]
        l2 = old_loose_state[1]
        # if (l1 & l2):
        #     return loose_state
        # elif(l1 & ~l2):
        #     return loose_taut_state
        # elif(~l1 & l2):
        #     return taut_loose_state
        # else:
        #     return taut_state
        # new_state = {}
        # loose_state1 = both_loose2taut_transfer(env_params, loose_state)
        # loose_state2 = loose2taut_transfer(env_params, loose_state, 0)
        # loose_state3 = loose2taut_transfer(env_params, loose_state, 1)
        # for k in loose_state.__dict__.keys():
        #     new_state[k] = jnp.where(ll2tt, loose_state1.__dict__[k], loose_state.__dict__[k])
        #     new_state[k] = jnp.where(ll2tl, loose_state2.__dict__[k], loose_state.__dict__[k])
        #     new_state[k] = jnp.where(ll2lt, loose_state3.__dict__[k], loose_state.__dict__[k])
        
        # loose_taut_state1 = loose2taut_transfer(env_params, loose_taut_state, 0)
        # for k in loose_taut_state.__dict__.keys():
        #     new_state[k] = jnp.where(lt2tl | lt2tt, loose_taut_state1.__dict__[k], loose_taut_state.__dict__[k])

        # taut_loose_state1 = loose2taut_transfer(env_params, taut_loose_state, 1)
        # for k in taut_loose_state.__dict__.keys():
        #     new_state[k] = jnp.where(tl2lt | tl2tt, taut_loose_state1.__dict__[k], taut_loose_state.__dict__[k])



        # loose_state = jnp.where(ll2tt, both_loose2taut_transfer(env_params, loose_state), loose_state)
        # loose_state = jnp.where(ll2tl, loose2taut_transfer(env_params, loose_state, 0), loose_state)
        # loose_state = jnp.where(ll2lt, loose2taut_transfer(env_params, loose_state, 1), loose_state)
        # loose_taut_state = jnp.where(lt2tl | lt2tt, loose2taut_transfer(env_params, loose_taut_state, 0), loose_taut_state)
        # taut_loose_state = jnp.where(tl2lt | tl2tt, loose2taut_transfer(env_params, taut_loose_state, 1), taut_loose_state)

        # return jnp.where(l1 & l2, loose_state, jnp.where(l1 & ~l2, loose_taut_state, jnp.where(~l1 & l2, taut_loose_state, taut_state)))
        # loose_state = jnp.where(ll2tt, both_loose2taut_transfer(env_params, loose_state),
        #                jnp.where(ll2tl, loose2taut_transfer(env_params, loose_state, 0),
        #                         jnp.where(ll2lt, loose2taut_transfer(env_params, loose_state, 1), loose_state)))

        # return jnp.where(l1 & l2, loose_state,
        #                 jnp.where(l1 & ~l2, loose_taut_state,
        #                         jnp.where(~l1 & l2, taut_loose_state, taut_state)))

        new_state = {}
        for k in taut_state.__dict__.keys():
            new_state[k] = jnp.where(ll2tt, loose_state.__dict__[k], 
                            jnp.where(ll2tl, loose_state.__dict__[k],
                            jnp.where(ll2lt, loose_state.__dict__[k],
                            jnp.where(lt2tl | lt2tt, loose_taut_state.__dict__[k],
                            jnp.where(tl2lt | tl2tt, taut_loose_state.__dict__[k],
                            taut_state.__dict__[k])))))

        for k in taut_state.__dict__.keys():
            new_state[k] = jnp.where(l1 & l2, loose_state.__dict__[k],
                            jnp.where(l1 & ~l2, loose_taut_state.__dict__[k],
                            jnp.where(~l1 & l2, taut_loose_state.__dict__[k],
                            taut_state.__dict__[k])))
        
        taut_state = taut_state.replace(**new_state)
        return taut_state

    return dynamic_transfer
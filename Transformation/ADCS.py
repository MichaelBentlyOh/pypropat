import numpy as np
from kinematics import *

def rigbody (x, ext_torque, tensin,
             teninv, mag_moment=np.zeros(3), magnetic_field=np.zeros(3)):
    q = x[:4]
    w = x[4:7]
    xp = 0.5 * sangvel(w) * q

    torque = ext_torque + np.cross(mag_moment, quatrmx(q)*magnetic_field)

    wp = teninv*(np.cross(tensin*w,w)+torque)

    return np.array([[xp],[wp]])

def rb_nutation_damper (x, ext_torque, tensin, teninv,
                nd_axis, nd_inertia, nd_spring, nd_damper):
    q = x[:4]
    w = x[4:7]
    nd_momentum = x[8]  
    nd_angle = x[9]     

    nd_ang_vel = (nd_momentum - nd_inertia * np.dot(nd_axis.T, w)) / nd_inertia

    xp = 0.5 * sangvel(w) * q
    torque = ext_torque

    nd_momentum_dot = -nd_spring * nd_angle - nd_damper * nd_ang_vel

    nd_angle_dot = nd_ang_vel

    wp = np.dot(teninv, (torque - np.cross(w, np.dot(tensin, w) + nd_momentum * nd_axis) -
                         nd_momentum_dot * nd_axis))

    dxdt = np.concatenate([xp, wp, [nd_momentum_dot], [nd_angle_dot]])
    return dxdt

def rb_reaction_wheel_n (time, x, flag, ext_torque, tinerb,
                         tinerbinv, n, rw_torque, an):
    q = x[:4]
    w = x[4:7]
    hwn = an*x[8:7+n]
    tqrw = an * rw_torque

    xp = 0.5*sangvel(w)*q
    wp = tinerbinv*(ext_torque + np.cross(tinerb*w + hwn, w) - tqrw)
    hwnp = tqrw

    dxdt = np.concatenate([xp, wp,hwnp])
    return dxdt


def rb_reaction_wheel (time, x, flag, ext_torque,
                       tinerb,tinerbinv,rw_torque):
    q = x[:4]
    w = x[4:7]
    hwn = x[8:10]
    xp = 0.5 * sangvel(w) * q

    wp = tinerbinv*(ext_torque + np.cross(tinerb*w + hwn, w) - rw_torque)
    hwnp = rw_torque
    dxdt = np.concatenate([xp, wp, hwnp])
    return dxdt

def rw_speed_n(x,rw_iner,an):
    n = len(rw_iner)
    wns = x[7:7+n] / rw_iner - np.dot(an.T, x[4:7])
    return wns

def rw_speed(w, hwn, rw_iner):
    wns = hwn/rw_iner - w
    return wns
import coal.coal_pywrap
import numpy as np

def rpy_to_quaternion(rpy, array_format=False):
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if array_format:
        return np.array([x,y,z,w])
    else:
        return coal.coal_pywrap.Quaternion(w,x,y,z) # w,x,y,z

import casadi
from casadi import vertcat
def csquat_to_rpy(quat):
    x,y,z,w = quat[0],quat[1],quat[2],quat[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    r = casadi.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    p = casadi.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    y = casadi.arctan2(t3, t4)

    return vertcat(r,p,y)
def csrpy_to_quat(rpy):
    roll, pitch, yaw = rpy[0],rpy[1],rpy[2]
    cy = casadi.cos(yaw * 0.5)
    sy = casadi.sin(yaw * 0.5)
    cp = casadi.cos(pitch * 0.5)
    sp = casadi.sin(pitch * 0.5)
    cr = casadi.cos(roll * 0.5)
    sr = casadi.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return vertcat(x,y,z,w)

def quat_to_rpy(quat):
    x,y,z,w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    r = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    p = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    y = np.arctan2(t3, t4)

    return r,p,y
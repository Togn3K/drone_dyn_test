import numpy as np
from scipy.integrate import odeint 
import time

def sKw(x):
    x = x.flatten()
    Y = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)
    return Y


def drone_dyn(state, t, g, m, inertia_matrix, w_r, f_thrust, kw, kv):
    # Drone state
    state = np.expand_dims(state, axis=1)

    # rpy
    w_r = w_r.reshape(3, 1)
    kv = kv.reshape(3, 1)

    # Variables and Parameters
    zeta = np.array([0, 0, 1]).reshape(3, 1)
    gv = np.array([0, 0, -1]).reshape(3, 1) * g
    p = state[0: 3, 0]
    v = state[3: 6, 0].reshape(3, 1)
    R = state[6: 15].reshape(3, 3)
    w = state[15: 18, 0].reshape(3, 1)

    # Low-level attitude controller
    tau = -kw * (w - w_r)
    f_drag = kv * v

    # Drone Dynamics
    dp = v
    dv = (np.dot(R, zeta) * f_thrust - f_drag) / m + gv
    dR = np.dot(R, sKw(w))
    dw = np.dot(np.linalg.inv(inertia_matrix), np.dot(sKw(np.dot(inertia_matrix, w)), w) + tau)

    # Output
    d_state = np.concatenate((dp.reshape(-1, 1), dv, dR.reshape(-1, 1), dw.reshape(-1, 1)), axis=0).squeeze()
    #print("d_state: {}\n".format(d_state))
    return d_state


def lambda_dyn(dlam, lk, Ts):
    lk1 = lk + dlam * Ts
    return lk1


def yaw_control(R, yp, kth):
    # print(R)
    # print(yp)
    yh = np.arctan2(np.dot(R[:,1:2].T, yp), np.dot(R[:,0:1].T, yp))
    return kth * np.sin(yh)


if __name__ == "__main__":
    g = 9.8
    Tin = 0
    Ts = 1./100
    Tend = 10 
    N = Tend/Ts 
    p = np.vstack(np.array([0,0,0], dtype=np.float64))
    v = np.vstack(np.array([0,0,0], dtype=np.float64))
    R = np.eye(3)
    w = np.vstack(np.array([0,0,0], dtype=np.float64))
    xr = np.vstack(np.array([0,0,0.4,0,0,0,0,0,0,0,0,0], dtype=np.float64))
    m = 1
    mnom = 1
    im = np.diag([1,1,1])*0.05
    kth = 5 
    kw = np.linalg.norm(im)*50
    lk = np.log(m/mnom*g)
    rp = np.vstack(np.array([0,0,0], dtype=np.float64))
    KdA = np.array([
        [ 0.9793, -0.1393, -0.0083, -0.0007,  0.0002,  0.0000,  0.0001],
        [ 0.1393,  0.3086,  0.1195,  0.0086, -0.0019, -0.0004, -0.0008],
        [ 0.0083,  0.1195,  0.9657, -0.0030,  0.0011,  0.0002,  0.0004],
        [ 0.0007,  0.0086, -0.0030,  0.9997,  0.0001,  0.0000,  0.0000],
        [ 0.0002,  0.0019, -0.0011, -0.0001,  0.9999, -0.0000, -0.0000],
        [-0.0000, -0.0004,  0.0002,  0.0000,  0.0000,  1.0000, -0.0008],
        [ 0.0001,  0.0008, -0.0004, -0.0000, -0.0000,  0.0008,  0.9998]
    ], dtype=np.float64)
    KdB = np.array([
        [-0.0552],
        [ 0.2768],
        [-0.0209],
        [-0.0011],
        [-0.0001],
        [ 0.0000],
        [-0.0000]
    ], dtype=np.float64)
    kdC = np.array([1.5577, 47.8875, -5.6419, -0.3678, 0.0652, 0.0148, 0.0273], dtype=np.float64)
    kdD = 0
    xest1 = np.zeros((len(KdA), 1))
    xest2 = np.zeros((len(KdA), 1))
    xest3 = np.zeros((len(KdA), 1))
    Af = np.array([
        [ 0.9703, -0.0148, -0.0099],
        [ 0.0197,  0.9999, -0.0001],
        [ 0.0000,  0.0050,  1.0000]
    ], dtype=np.float64)
    Bf = np.array([
        [ 0.0099],
        [ 0.0001],
        [ 0.0000]
    ], dtype=np.float64)
    Cf = np.array([0, 0, 1], dtype=np.float64)
    Df = 0
    xf1 = np.array([
        [0], 
        [0], 
        [xr[0,0] - p[0,0]]
    ], dtype=np.float64)
    xf2 = np.array([
        [0], 
        [0], 
        [xr[1,0] - p[1,0]]
    ], dtype=np.float64)
    xf3 = np.array([
        [0], 
        [0], 
        [xr[2,0] - p[2,0]]
    ], dtype=np.float64)
    Xin = np.concatenate((p, v, R.reshape(9,1), w))
    for i in range(int(N)):
        pr = xr[0:3,:]
        yp = pr - p 
        rf1 = np.dot(Cf, xf1) + np.dot(Df, rp[0][0])
        xf1 = np.dot(Af, xf1) + np.dot(Bf, rp[0][0])
        rf2 = np.dot(Cf, xf2) + np.dot(Df, rp[1][0])
        xf2 = np.dot(Af, xf2) + np.dot(Bf, rp[1][0])
        rf3 = np.dot(Cf, xf3) + np.dot(Df, rp[2][0])
        xf3 = np.dot(Af, xf3) + np.dot(Bf, rp[2][0])
        rf = np.vstack([rf1[0], rf2[0], rf3[0]])
        inpcop = rf - yp
        u1 = np.dot(kdC, xest1) + np.dot(kdD, inpcop[0][0])
        xest1 = np.dot(KdA, xest1) + np.dot(KdB, inpcop[0][0])
        u2 = np.dot(kdC, xest2) + np.dot(kdD, inpcop[1][0])
        xest2 = np.dot(KdA, xest2) + np.dot(KdB, inpcop[1][0])
        u3 = np.dot(kdC, xest3) + np.dot(kdD, inpcop[2][0])
        xest3 = np.dot(KdA, xest3) + np.dot(KdB, inpcop[2][0])
        u = - np.vstack([u1, u2, u3])
        omega3 = yaw_control(R, yp, kth)
        fk = np.exp(lk) * mnom
        # print("lk: {}".format(lk))
        # print(R.T)
        # print(u)
        q = np.exp(-lk) * np.dot(R.T, u)
        wtr = np.zeros((3, 1))
        wtr[0][0] = -q[1][0]
        wtr[1][0] = q[0][0]
        wtr[2][0] = omega3
        # wtr[2] = 0.0
        dlam = q[2][0]
        # print("dlam: {}, lk: {}, Ts: {}".format(dlam, lk, Ts))
        lk = lambda_dyn(dlam, lk, Ts)
        t = [Tin, Tin + Ts]
        # print(Xin)
        Xout = odeint(drone_dyn, Xin.squeeze(), t, args=(g, m, im, wtr, fk, w, v))
        Xin = np.vstack(Xout[-1])
        # print(Xin)
        # print("----------------------------")
        Tin = Tin + Ts
        p = Xin[0:3][:]
        v = Xin[3:6][:]
        R = Xin[6:15][:].reshape((3,3))
        w = Xin[15:18][:]
        print(p)
        # time.sleep(0.1)

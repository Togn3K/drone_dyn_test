import numpy as np
from scipy.integrate import ode 
import time

def skw(x):
    return np.array([
        [0, -x[2,0], x[1,0]],
        [x[2,0], 0, -x[0,0]],
        [-x[1,0], x[0,0], 0]
    ])


def dynamic(t, y, m, im, g, f, wh, kw):
    zeta = np.vstack(np.array([0, 0, 1]))
    gv = np.vstack(np.array([0, 0, -1])) * g 
    p = y[:3].reshape((3,1))
    v = y[3:6].reshape((3,1))
    R = y[6:15].reshape((3,3))
    w = y[15:18].reshape((3,1))

    tau = - kw * (w-wh)

    dp = v 
    dv = (np.dot(R, zeta) * f) / m + gv
    dR = np.dot(R, skw(w))
    dw = np.dot(np.linalg.inv(im), (np.dot(skw(np.dot(im, w)), w)) + tau)

    dx = np.concatenate((dp, dv, dR.reshape((9,1)), dw))
    return dx


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
    Tend = 100
    N = Tend/Ts 
    p = np.vstack(np.array([0,0,0]))
    v = np.vstack(np.array([0,0,0]))
    R = np.eye(3)
    w = np.vstack(np.array([0,0,0]))
    xr = np.vstack(np.array([0,0,0.4,0,0,0,0,0,0,0,0,0]))
    m = 1
    mnom = 1
    im = np.diag([1,1,1])*0.05
    kth = 5 
    kw = np.linalg.norm(im)*50
    lk = np.log(m/mnom*g)
    rp = np.vstack(np.array([0,0,0]))
    KdA = np.array([
        [ 0.9793, -0.1393, -0.0083, -0.0007,  0.0002,  0.0000,  0.0001],
        [ 0.1393,  0.3086,  0.1195,  0.0086, -0.0019, -0.0004, -0.0008],
        [ 0.0083,  0.1195,  0.9657, -0.0030,  0.0011,  0.0002,  0.0004],
        [ 0.0007,  0.0086, -0.0030,  0.9997,  0.0001,  0.0000,  0.0000],
        [ 0.0002,  0.0019, -0.0011, -0.0001,  0.9999, -0.0000, -0.0000],
        [-0.0000, -0.0004,  0.0002,  0.0000,  0.0000,  1.0000, -0.0008],
        [ 0.0001,  0.0008, -0.0004, -0.0000, -0.0000,  0.0008,  0.9998]
    ])
    KdB = np.array([
        [-0.0552],
        [ 0.2768],
        [-0.0209],
        [-0.0011],
        [-0.0001],
        [ 0.0000],
        [-0.0000]
    ])
    kdC = np.array([1.5577, 47.8875, -5.6419, -0.3678, 0.0652, 0.0148, 0.0273])
    kdD = 0
    xest1 = np.zeros((len(KdA), 1))
    xest2 = np.zeros((len(KdA), 1))
    xest3 = np.zeros((len(KdA), 1))
    Af = np.array([
        [ 0.9703, -0.0148, -0.0099],
        [ 0.0197,  0.9999, -0.0001],
        [ 0.0000,  0.0050,  1.0000]
    ])
    Bf = np.array([
        [ 0.0099],
        [ 0.0001],
        [ 0.0000]
    ])
    Cf = np.array([0, 0, 1])
    Df = 0
    xf1 = np.array([
        [0], 
        [0], 
        [xr[0,0] - p[0,0]]
    ])
    xf2 = np.array([
        [0], 
        [0], 
        [xr[1,0] - p[1,0]]
    ])
    xf3 = np.array([
        [0], 
        [0], 
        [xr[2,0] - p[2,0]]
    ])
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
        # print(lk)
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
        solver = ode(dynamic).set_integrator('dopri5')
        solver.set_initial_value(Xin, Tin).set_f_params(m, im, g, fk, wtr, kw)
        while solver.successful() and solver.t < Tin+Ts:
            solver.integrate(solver.t + Ts)
        Xin = solver.y 
        Tin = solver.t 
        p = Xin[0:3][:]
        v = Xin[3:6][:]
        R = Xin[6:15][:].reshape((3,3))
        w = Xin[15:18][:]
        # time.sleep(0.1)
        print(p)

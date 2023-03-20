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


def yaw_control(R, yh, kth):
    # print(R)
    # print(yp)
    # yh = np.arctan2(np.dot(R[:,1:2].T, yp), np.dot(R[:,0:1].T, yp))
    return kth * np.sin(yh)


class DroneLQR:
    def __init__(self, optimal_distance=0.0):

        self._optimal_distance = optimal_distance

        # Get Parameters
        global_parameter_name = "/drone_controller"

        # TrustAsTrust Hovering value
        self.TRUST_STATIC = 0.375 
        self.ts = 1. / 100
        
        # drone parameters
        self.g = 9.8
        self.m = 1   # actual drone mass [kg]
        self.mnom = 1   # lqr design mass [kg]

        # control parameters 
        self.kth = 5
        self.lk = np.log(self.m / self.mnom * self.g)
        self.rp = np.vstack(np.array([self._optimal_distance, 0, 0]))

        if self.ts == 0.01:
            self.KdA = np.array([
                [ 0.9793, -0.1393, -0.0083, -0.0007,  0.0002,  0.0000,  0.0001],
                [ 0.1393,  0.3086,  0.1195,  0.0086, -0.0019, -0.0004, -0.0008],
                [ 0.0083,  0.1195,  0.9657, -0.0030,  0.0011,  0.0002,  0.0004],
                [ 0.0007,  0.0086, -0.0030,  0.9997,  0.0001,  0.0000,  0.0000],
                [ 0.0002,  0.0019, -0.0011, -0.0001,  0.9999, -0.0000, -0.0000],
                [-0.0000, -0.0004,  0.0002,  0.0000,  0.0000,  1.0000, -0.0008],
                [ 0.0001,  0.0008, -0.0004, -0.0000, -0.0000,  0.0008,  0.9998]
            ])
            self.KdB = np.array([
                [-0.0552],
                [ 0.2768],
                [-0.0209],
                [-0.0011],
                [-0.0001],
                [ 0.0000],
                [-0.0000]
            ])
            self.kdC = np.array([1.5577, 47.8875, -5.6419, -0.3678, 0.0652, 0.0148, 0.0273])
            self.kdD = 0
            self.xest1 = np.zeros((len(self.KdA), 1))
            self.xest2 = np.zeros((len(self.KdA), 1))
            self.xest3 = np.zeros((len(self.KdA), 1))
            self.Af = np.array([
                [ 0.9703, -0.0148, -0.0099],
                [ 0.0197,  0.9999, -0.0001],
                [ 0.0000,  0.0050,  1.0000]
            ])
            self.Bf = np.array([
                [ 0.0099],
                [ 0.0001],
                [ 0.0000]
            ])
            self.Cf = np.array([0, 0, 1])
            self.Df = 0
            self.xf1 = np.array([
                [0], 
                [0], 
                [0]
                # [xr[0,0] - p[0,0]]
            ])
            self.xf2 = np.array([
                [0], 
                [0],
                [0] 
                # [xr[1,0] - p[1,0]]
            ])
            self.xf3 = np.array([
                [0], 
                [0], 
                [0]
                # [xr[2,0] - p[2,0]]
            ])
        else:
            print("The LQR Agents only supports the following ts: 0.01 (100Hz)")
            exit(-1)

    def perform_action(self, observation, take_off=False) -> tuple:
        yp = observation["p"]
        R = observation["R"]
        yaw = observation["yaw"]
        # print("observation: {}".format(observation))
        rf1 = np.dot(self.Cf, self.xf1) + np.dot(self.Df, self.rp[0][0])
        self.xf1 = np.dot(self.Af, self.xf1) + np.dot(self.Bf, self.rp[0][0])
        rf2 = np.dot(self.Cf, self.xf2) + np.dot(self.Df, self.rp[1][0])
        self.xf2 = np.dot(self.Af, self.xf2) + np.dot(self.Bf, self.rp[1][0])
        rf3 = np.dot(self.Cf, self.xf3) + np.dot(self.Df, self.rp[2][0])
        self.xf3 = np.dot(self.Af, self.xf3) + np.dot(self.Bf, self.rp[2][0])
        # print("rf1: {}".format(rf1))
        # print("rf2: {}".format(rf2))
        # print("rf3: {}".format(rf3))
        rf = np.vstack([rf1[0], rf2[0], rf3[0]])
        # print("rf: {}".format(rf))
        # print("p0: {}".format(p0))
        inpcop = rf - yp
        # print("inpcop: {}".format(inpcop))
        u1 = np.dot(self.kdC, self.xest1) + np.dot(self.kdD, inpcop[0][0]) 
        self.xest1 = np.dot(self.KdA, self.xest1) + np.dot(self.KdB, inpcop[0][0]) 
        u2 = np.dot(self.kdC, self.xest2) + np.dot(self.kdD, inpcop[1][0])
        self.xest2 = np.dot(self.KdA, self.xest2) + np.dot(self.KdB, inpcop[1][0]) 
        u3 = np.dot(self.kdC, self.xest3) + np.dot(self.kdD, inpcop[2][0]) 
        self.xest3 = np.dot(self.KdA, self.xest3) + np.dot(self.KdB, inpcop[2][0]) 
        u = - np.vstack([u1, u2, u3])

        omega3 = yaw_control(R, yaw, self.kth)
        f = np.exp(self.lk) * self.mnom
        q = np.exp(-self.lk) * np.dot(R.T, u)
        # print("q: {}".format(q))
        wtr = np.zeros((3, 1))
        wtr[0][0] = -q[1][0]
        wtr[1][0] = q[0][0]
        wtr[2][0] = omega3
        # wtr[2] = 0.0
        dlam = q[2][0]

        self.lk = lambda_dyn(dlam, self.lk, self.ts)

        return wtr, f


if __name__ == "__main__":
    drone = DroneLQR()
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

    Xin = np.concatenate((p, v, R.reshape(9,1), w))

    for i in range(int(N)):
        pr = xr[0:3,:]
        yp = pr - p 

        yh = np.arctan2(np.dot(R[:,1:2].T, yp), np.dot(R[:,0:1].T, yp))
        obs = {"p": yp, "R": R, "yaw": yh}

        wtr, fk = drone.perform_action(obs)

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
        # time.sleep(0.01)
        print("[{}/{}] {}".format(i+1, int(N), p.flatten()))

"""

Unscented kalman filter (UKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import pandas as pd

from utils.angle import rot_mat_2d

# Covariance for UKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

#  UKF Parameter
ALPHA = 0.001
BETA = 2
KAPPA = 0

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawRate = 0.1  # [rad/s]
    u = np.array([[v, yawRate]]).T
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def generate_sigma_points(xEst, PEst, gamma):
    sigma = xEst
    Psqrt = scipy.linalg.sqrtm(PEst)
    n = len(xEst[:, 0])
    # Positive direction
    for i in range(n):
        sigma = np.hstack((sigma, xEst + gamma * Psqrt[:, i:i + 1]))

    # Negative direction
    for i in range(n):
        sigma = np.hstack((sigma, xEst - gamma * Psqrt[:, i:i + 1]))

    return sigma


def predict_sigma_motion(u, sigma):
    """
        Sigma Points prediction with motion model
    """
    for i in range(sigma.shape[1]):
        sigma[:, i:i + 1] = motion_model(sigma[:, i:i + 1], u)

    return sigma


def predict_sigma_observation(sigma):
    """
        Sigma Points prediction with observation model
    """
    for i in range(sigma.shape[1]):
        sigma[0:2, i] = observation_model(sigma[:, i])

    sigma = sigma[0:2, :]

    return sigma


def calc_sigma_covariance(x, sigma, wc, Pi):
    nSigma = sigma.shape[1]
    d = sigma - x[0:sigma.shape[0]]
    P = Pi
    for i in range(nSigma):
        P = P + wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
    return P


def calc_pxz(sigma, x, z_sigma, zb, wc):
    nSigma = sigma.shape[1]
    dx = sigma - x
    dz = z_sigma - zb[0:2]
    P = np.zeros((dx.shape[0], dz.shape[0]))

    for i in range(nSigma):
        P = P + wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

    return P


# def ukf_estimation(mu_tm1, Sigma_tm1, u_t, z_t, wm, wc, gamma):
#     """
#     ------- iteration params -------
#     mu_tm1, Sigma_tm1, u_t, z_t
    
#     ------- weighting params -------
#     wm, wc, gamma
#     """
#     #  Predict
#     X_tm1 = generate_sigma_points(mu_tm1, Sigma_tm1, gamma)                 # alg line 2
#     X_t_barstar = predict_sigma_motion(u_t, X_tm1)                          # alg line 3
#     mu_t_bar = (wm @ X_t_barstar.T).T                                       # alg line 4
#     Sigma_t_bar = calc_sigma_covariance(mu_t_bar, X_t_barstar, wc, R)       # alg line 5

#     #  Update
#     X_t_bar = generate_sigma_points(mu_t_bar, Sigma_t_bar, gamma)           # alg line 6
#     Zeta_t_bar = predict_sigma_observation(X_t_bar)                         # alg line 7: a predicted obsv is computed for each sigma point (to verify)
#     z_t_hat = (wm @ Zeta_t_bar.T).T                                         # alg line 8
#     S_t = calc_sigma_covariance(z_t_hat, Zeta_t_bar, wc, Q)                 # alg line 9 (to verify...)
#     Sigma_t_bar_xz = calc_pxz(X_t_bar, mu_t_bar, Zeta_t_bar, z_t_hat, wc)   # alg line 10 (to verify...)
#     K_t = Sigma_t_bar_xz @ np.linalg.inv(S_t)                               # alg line 11
#     mu_t = mu_t_bar + K_t @ (z_t - z_t_hat)                                 # alg line 12
#     Sigma_t = Sigma_t_bar - K_t @ S_t @ K_t.T                               # alg line 13

#     return mu_t, Sigma_t

def ukf_estimation(xEst, PEst, z, u, wm, wc, gamma):
    """
    
    """
    #  Predict
    sigma = generate_sigma_points(xEst, PEst, gamma)            # alg line 2
    sigma = predict_sigma_motion(sigma, u)                      # alg line 3
    xPred = (wm @ sigma.T).T                                    # alg line 4
    PPred = calc_sigma_covariance(xPred, sigma, wc, Q)          # alg line 5

    #  Update
    zPred = observation_model(xPred)                            # alg line 8
    y = z - zPred                                               # (z_t - \hat{z_t}) (used in line 12)
    sigma = generate_sigma_points(xPred, PPred, gamma)          # alg line 6
    zb = (wm @ sigma.T).T                                       # i feel like this is alg line 8 (w_m * z_sigma)
    z_sigma = predict_sigma_observation(sigma)                  # alg line 7
    st = calc_sigma_covariance(zb, z_sigma, wc, R)              # alg line 9
    Pxz = calc_pxz(sigma, xPred, z_sigma, zb, wc)               # alg line 10
    K = Pxz @ np.linalg.inv(st)                                 # alg line 11
    xEst = xPred + K @ y                                        # alg line 12
    PEst = PPred - K @ st @ K.T                                 # alg line 13

    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ np.array([x, y])
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def setup_ukf(nx):
    """
    See PR 3.4.1
    λ = α^2(n+κ)−n
    α, κ = scaling params determining how far sigma points are spread around mean
    n = dimensions of state vector
    """
    
    lamb = ALPHA ** 2 * (nx + KAPPA) - nx
    # calculate weights
    wm = [lamb / (lamb + nx)]
    wc = [(lamb / (lamb + nx)) + (1 - ALPHA ** 2 + BETA)]
    for i in range(2 * nx):
        wm.append(1.0 / (2 * (nx + lamb)))
        wc.append(1.0 / (2 * (nx + lamb)))
    gamma = math.sqrt(nx + lamb)

    wm = np.array([wm])
    wc = np.array([wc])

    return wm, wc, gamma


def read_data():
    # Read in the data
    dataset = "ds1"

    ### Barcode mapping
    subject2barcode = {}
    barcode2subject = {}
    with open(f"{dataset}/{dataset}_Barcodes.dat") as f:
        for _ in range(4):
            next(f)
        
        for line in f:
            subject, barcode = map(int, line.strip().split())
            subject2barcode[subject] = barcode
            barcode2subject[barcode] = subject

    ### Landmark GT
    landmark_gt = pd.read_csv(f"{dataset}/{dataset}_Landmark_Groundtruth.dat", sep="\s+", skiprows=4, 
                            names=["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])

    ### Robot GT
    gt = pd.read_csv(f"{dataset}/{dataset}_Groundtruth.dat", sep="\s+", skiprows=4,
                    names=["Time [s]", "x [m]", "y [m]", "Orientation [rad]"])

    ### Robot Measurements
    msmt = pd.read_csv(f"{dataset}/{dataset}_Measurement.dat", sep="\s+", skiprows=4,
                    names=["Time [s]", "Subject #", "Range [m]", "Bearing [rad]"])

    ### Robot Odometry
    odo = pd.read_csv(f"{dataset}/{dataset}_Odometry.dat", sep="\s+", skiprows=4,
                    names=["Time [s]", "Forward Velocity [m/s]", "Angular Velocity [rad/s]"])
    
    return landmark_gt, gt, msmt, odo


def main():
    print(__file__ + " start!!")

    ### Read Data Files ###
    landmark_gt, gt, msmt, odo = read_data()

    nx = 4  # State Vector [x y yaw v]'
    # nx = 3  # State Vector [x y heading]'
    xEst = np.zeros((nx, 1))
    xTrue = np.zeros((nx, 1))
    PEst = np.eye(nx)
    xDR = np.zeros((nx, 1))  # Dead reckoning

    wm, wc, gamma = setup_ukf(nx)

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    time = 0.0

    while SIM_TIME >= time:
    # for i in range(1, len(odo)):
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ukf_estimation(xEst, PEst, z, ud, wm, wc, gamma)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxDR[0, :]).flatten(),
                     np.array(hxDR[1, :]).flatten(), "-k")
            plt.plot(np.array(hxEst[0, :]).flatten(),
                     np.array(hxEst[1, :]).flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()

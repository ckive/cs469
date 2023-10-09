"""
following this
https://github.com/Al-khwarizmi-780/OpenKF/blob/main/python/examples/Introduction_Unscented_Kalman_Filter.ipynb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints, UnscentedKalmanFilter as UKF
from ukf_model import UKF
from myutils import read_data, residual_x, motion_model, observation_model, _normalize_angle, state_mean, z_mean, is_action_row

landmark_gt, gt, msmt, odo = read_data()

landmarks_np = landmark_gt[["x [m]", "y [m]"]].to_numpy()

all_data = pd.read_csv("action-update.csv")

MAX_UNCERTAINTY = 1e5
OBSERVED_UNCERTAINTY = 0.3


x0 = gt.iloc[0][["x [m]", "y [m]", "Orientation [rad]"]].to_numpy()
nx = np.shape(x0)[0]

# P0 = np.array([[0.01, 0.0, 0.0, 0.0],
#               [0.0, 0.01, 0.0, 0.0],
#               [0.0, 0.0, 0.05, 0.0],
#               [0.0, 0.0, 0.0, 0.05]])

# Sigma_tm1 
P0 = np.diag([0.00001] * nx)

R = np.diag([
        0.1,  # variance of location on x-axis
        0.1,  # variance of location on y-axis
        np.deg2rad(1.0),  # variance of yaw angle
    ]) ** 2  # predict state covariance

Q = np.eye(30) * MAX_UNCERTAINTY

nz = np.shape(R)[0]

# init our UKF
ukf = UKF(dim_x=nx, dim_z=nz, Q=Q, R=R, kappa=(3 - nx))

### Start running on sequence
i = 0
N = all_data.shape[0]
# offset such that subj 6->0, and so on so that 15 landmarks are indexed 0-14
subj_offset = 6
# num actions taken ctr (expect total of 5121)
actions_ctr = 1


# Getting ready!
x, P = x0, P0

# i is always the action index
while i < N:
    # print(f"i: {i}, action {actions_ctr}, state x y theta: {mu_tm1[0]} {mu_tm1[1]} {mu_tm1[2]}")
    # actions_ctr += 1

    # ground truth at this time step
    gt_state = np.array([all_data.iloc[i+1]["x [m]"], all_data.iloc[i+1]["y [m]"], all_data.iloc[i+1]["Orientation [rad]"]]).reshape(-1,1)

    # action at this time step
    u_t = np.array([[all_data.iloc[i]["Forward Velocity [m/s]"], all_data.iloc[i]["Angular Velocity [rad/s]"]]]).T

    # get measurement of this step
    z_t = np.zeros((30, 1))
    j = i+2
    measurements = []
    while j < N and not is_action_row(all_data.iloc[j]):
        # get measurement for each landmark
        jrow = all_data.iloc[j]
        subj = int(jrow["Subject #"])
        ldmk_r, ldmk_phi = jrow["Range [m]"], jrow["Bearing [rad]"]
        z_t[2*(subj-subj_offset)] = ldmk_r
        z_t[2*(subj-subj_offset)+1] = ldmk_phi

        x, P, _ = ukf.predict(motion_model, x, P, dt=dt)
        x, P, _ = ukf.correct(observation_model, x, P, z_t)




        # Trust this measurement by lowering noise in R
        Q[2*(subj-subj_offset), 2*(subj-subj_offset)] = OBSERVED_UNCERTAINTY
        j += 1

    # get the dt from next action
    dt = all_data.iloc[j]["Time [s]"] - all_data.iloc[i]["Time [s]"]
    # move to next action step
    i = j
    
    # get dead reckoning at this time step
    dr_state = motion_model(dr_state, u_t, dt)

    for iteration, z in enumerate(measurements):
        x, P, _ = ukf.predict(f_2, x, P)
        x, P, _ = ukf.correct(h_2, x, P, z)
        visualize_estimate(viewer, f'', 'g', x, P)

    
    
    x, P, _ = ukf.predict(motion_model, x, P, dt=dt)


    x, P, _ = ukf.correct(observation_model, x, P, z_t)





x, P, _ = ukf.predict(f_2, x0, P0)

print(f'x = \n {x.round(3)}')
print(f'P = \n {P.round(3)}')

x, P, _ = ukf.correct(h_2, x, P, z)

print(f'x = \n {x.round(3)}')
print(f'P = \n {P.round(3)}')


x0 = np.array([1.0, 2.0])
P0 = np.array([[1.0, 0.0], [0.0, 1.0]])
# 3



def f(x, v):
    return (x + v)

def h(x, n):
    return (x + n)

nx = 3
nz = 30

ukf = UKF(dim_x=nx, dim_z=nz, Q=Q, R=R, kappa=(3 - nx))

x1, P1, _ = ukf.predict(f, x0, P0)

print(f'x = \n{x1.round(5)}\n')
print(f'P = \n{P1.round(5)}\n')

x2, P2, _ = ukf.correct(h, x1, P1, z)

print(f'x = \n{x2.round(5)}\n')
print(f'P = \n{P2.round(5)}\n')
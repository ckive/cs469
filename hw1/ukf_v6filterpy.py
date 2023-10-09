import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints#, UnscentedKalmanFilter as UKF
from ukf_filterpy import UnscentedKalmanFilter as UKF
from myutils import (
    read_data, residual_x, residual_h, 
    motion_model, get_landmark_measurements_from_state, 
    normalize_angle, 
    state_mean, z_mean, 
    is_action_row,
)
from scipy.linalg import cholesky, lu

landmark_gt, gt, msmt, odo = read_data()

landmarks_np = landmark_gt[["x [m]", "y [m]"]].to_numpy()

all_data = pd.read_csv("action-update.csv")


points = MerweScaledSigmaPoints(n=3, alpha=1e-3, beta=2, kappa=0, 
                                    subtract=residual_x)

### UKF Parameters
# initial starting state is the gt at time of first action
# x0 = np.array([all_data.iloc[i+1]["x [m]"], all_data.iloc[i+1]["y [m]"], all_data.iloc[i+1]["Orientation [rad]"]])
x0 = all_data.iloc[1, -3:].to_numpy()

# initial known z is from the given ground truth at the beginning.
# successive steps will use the previously known Zs and update only those that is seen at a particular time step
z0 = get_landmark_measurements_from_state(x0, landmark_gt)

ukf = UKF(dim_x=3, dim_z=2*len(landmarks_np),
            fx=motion_model, hx=get_landmark_measurements_from_state, points=points,
            sqrt_fn=lu,
            x_mean_fn=state_mean, 
            z_mean_fn=z_mean, 
            residual_x=residual_x, 
            residual_z=residual_h,
            x0=x0, 
            P0=np.eye(3)*1e-5
            )
            # may want to tune the R and Q matrices later

cur_state = x0.copy()
prev_z = z0.copy()

track = []

i = 0
N = all_data.shape[0]
# offset such that subj 6->0, and so on so that 15 landmarks are indexed 0-14
subj_offset = 6
# num actions taken ctr (expect total of 5121)
actions_ctr = 1

# i is always the action index
while i < N:

    print(f"i: {i}, action {actions_ctr}, state x y theta: {cur_state[0]} {cur_state[1]} {cur_state[2]}")
    actions_ctr += 1

    ### get action at this time step (2,)
    u_t = np.array([all_data.iloc[i]["Forward Velocity [m/s]"], all_data.iloc[i]["Angular Velocity [rad/s]"]]).T

    ### using known measurements from prev time step, update measurement of landmarks seen in this time step
    cur_z = prev_z.copy()
    j = i+2
    while j < N and not is_action_row(all_data.iloc[j]):
        # get measurement for each landmark
        jrow = all_data.iloc[j]
        subj = int(jrow["Subject #"])
        ldmk_r, ldmk_phi = jrow["Range [m]"], jrow["Bearing [rad]"]
        cur_z[2*(subj-subj_offset)] = ldmk_r
        cur_z[2*(subj-subj_offset)+1] = ldmk_phi
        j += 1
    
    # get the dt from next action
    dt = all_data.iloc[j]["Time [s]"] - all_data.iloc[i]["Time [s]"]
    # move to next action step
    i = j

    # take action step
    ukf.predict(dt=dt, u=u_t)

    # take correction step
    ukf.update(cur_z, landmark_df=landmark_gt)


track = np.array(track)
plt.plot(track[:, 0], track[:,1], color='k', lw=2)
plt.axis('equal')
plt.title("UKF Robot localization")
plt.show()





# landmarks = np.array([[5, 10], [10, 5], [15, 15]])
# cmds = [np.array([1.1, .01])] * 200

#     cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
#     sigma_range=0.3, sigma_bearing=0.1)
# print('Final P:', ukf.P.diagonal())


# cmds, landmarks, sigma_vel, sigma_steer, sigma_range, 
# sigma_bearing, ellipse_step=1, step=10):

# plt.figure()
# points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
#                                 subtract=residual_x)
# ukf = UKF(dim_x=3, dim_z=2*len(landmarks), fx=move, hx=Hx,
#             dt=dt, points=points, x_mean_fn=state_mean, 
#             z_mean_fn=z_mean, residual_x=residual_x, 
#             residual_z=residual_h)

# ukf.x = np.array([2, 6, .3])
# ukf.P = np.diag([.1, .1, .05])
# ukf.R = np.diag([sigma_range**2, 
#                     sigma_bearing**2]*len(landmarks))
# ukf.Q = np.eye(3)*0.0001

# sim_pos = ukf.x.copy()

# # plot landmarks
# if len(landmarks) > 0:
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], 
#                 marker='s', s=60)

# track = []
# for i, u in enumerate(cmds):     
#     sim_pos = move(sim_pos, dt/step, u, wheelbase)
#     track.append(sim_pos)

#     if i % step == 0:
#         ukf.predict(u=u, wheelbase=wheelbase)

#         if i % ellipse_step == 0:
#             plot_covariance_ellipse(
#                 (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
#                     facecolor='k', alpha=0.3)

#         x, y = sim_pos[0], sim_pos[1]
#         z = []
#         for lmark in landmarks:
#             dx, dy = lmark[0] - x, lmark[1] - y
#             d = sqrt(dx**2 + dy**2) + randn()*sigma_range
#             bearing = atan2(lmark[1] - y, lmark[0] - x)
#             a = (normalize_angle(bearing - sim_pos[2] + 
#                     randn()*sigma_bearing))
#             z.extend([d, a])            
#         ukf.update(z, landmarks=landmarks)

#         if i % ellipse_step == 0:
#             plot_covariance_ellipse(
#                 (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
#                     facecolor='g', alpha=0.8)
# track = np.array(track)
# plt.plot(track[:, 0], track[:,1], color='k', lw=2)
# plt.axis('equal')
# plt.title("UKF Robot localization")
# plt.show()










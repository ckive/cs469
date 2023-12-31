import math

import matplotlib.pyplot as plt
import numpy as np
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
import pandas as pd

np.random.seed(469)
import scipy.linalg

#  UKF Parameter
# typically 1e-3 (0.001)
ALPHA = 0.001
# typically 2 if we don't have prior knowledge of the distribution, assume Gaussian
BETA = 2
# typically 3-n (n is the state dimension)
KAPPA = 0

show_animation = True

def _normalize_angle(angle):
    """
    Normalize an angle to be between -pi and pi.
    """
    # https://samialperenakgun.com/blog/2022/05/scale_radian_range_python/
    return np.arctan2(np.sin(angle),np.cos(angle))


def motion_model(posn, action, dt):
    """
    Update the robot's position using the velocity motion model.
    
    posn = (x, y, theta)
    action = (v, omega)
    dt: Time duration.
    """

    # given posn = (x, y, theta)
    x, y, theta = posn
    
    # given vector u, it contains the velocity and angular velocity
    v, omega = action
    
    # If the robot is moving approximately straight
    if abs(omega) < 1e-10:
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        # Orientation remains the same when omega is zero
    else:
        x += (v/omega) * (math.sin(theta + omega*dt) - math.sin(theta))
        y += (v/omega) * (math.cos(theta) - math.cos(theta + omega*dt))
        theta += omega * dt
        theta = _normalize_angle(theta)

    # make sure it's a n(3) x 1 vector
    return np.array([x, y, theta]).reshape(-1, 1)


# get measurement for a landmark with cur posn
def observation_model(ldmk_x, ldmk_y, x, y, theta):
    """
    measurement model:

    passed in 1 nx1 vector, representing a sigma point
    this functions as our "self.x,y" in previous iteration

    for all landmarks for which we have a measurement this iteration, 
        we compute the expected measurement for this landmark to our given sp
    
    return a 2x1 vector representing the expected measurement for this sp

    """
    # Compute expected range and bearing
    delta_x = ldmk_x - x
    delta_y = ldmk_y - y
    r_expected = math.sqrt(delta_x**2 + delta_y**2)
    phi_expected = _normalize_angle(math.atan2(delta_y, delta_x) - theta)
    
    return r_expected, phi_expected


def generate_sigma_points(mu, sigma, gamma):
    sigmapoints = mu
    Psqrt = scipy.linalg.sqrtm(sigma)
    n = len(mu[:, 0])
    # Positive direction
    for i in range(n):
        sigmapoints = np.hstack((sigmapoints, mu + gamma * Psqrt[:, i:i + 1]))

    # Negative direction
    for i in range(n):
        sigmapoints = np.hstack((sigmapoints, mu - gamma * Psqrt[:, i:i + 1]))

    return sigmapoints


def predict_sigma_motion(u, sigmapoints, dt):
    """
        Sigma Points prediction with motion model
        Alg line 3
    """
    for i in range(sigmapoints.shape[1]):
        # sigmapoints[:, i:i + 1] is 1 sigma point aka the ith column as a nx1 vector
        # motion_model output is a nx1 vector. 
        # with respect to my motion model, the output is a [x, y, theta] vector
        sigmapoints[:, i:i + 1] = motion_model(sigmapoints[:, i:i + 1], u, dt)

    return sigmapoints


def predict_sigma_observation(sigmapoints, landmark_gt):
    """
        Sigma Points prediction with measurement model

        This should return another set of transformed SigmaPoints with the measurement model applied to each point
        Alg line 7
    """

    Zeta = np.zeros((30, sigmapoints.shape[1])) # (2K x 2n+1) matrix
    for i in range(sigmapoints.shape[1]):
        Zeta[:, i:i+1] = get_landmark_measurement_from_robo_posn(landmark_gt, sigmapoints[:, i])

    # returns a (2K x 2n+1) matrix representing r,phi for each landmark k for all 2n+1 sigma points
    return Zeta


def calc_covariance(vector, matrix, wc, noise_covar):
    """
    input:
        vector: nx1 vector      (3x1)
        matrix: nx2n+1 matrix   (3x7)
        wc: 1x2n+1 matrix       (1x7)
        noise_covar: nxn matrix (3x3)


        Calculate the covariance of the sigma points
        Alg line 5, 9
    """
    N = matrix.shape[1]
    d = matrix - vector 
    new_covar = noise_covar
    for i in range(N):
        # what is shape of wc here?: 2n+1 x 1 (still wanna take a look)
        new_covar += wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
    return new_covar


def calc_cross_covariance_xz(X_t_bar, mu_t_bar, Zeta_t_bar, z_t_hat, wc):
    """
        Calculate the cross covariance of the sigma points
        Alg line 10

        ret nx2K matrix
    """
    nSigmapoints = X_t_bar.shape[1]
    dx = X_t_bar - mu_t_bar
    dz = Zeta_t_bar - z_t_hat
    sigma_xz = np.zeros((dx.shape[0], dz.shape[0]))

    for i in range(nSigmapoints):
        sigma_xz = sigma_xz + wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

    return sigma_xz


def ukf_estimation(mu_tm1, Sigma_tm1, z_t, u_t, wm, wc, gamma, R, Q, dt, landmark_gt):
    """
    ------- iteration params -------
    mu_tm1, Sigma_tm1, u_t, z_t
    
    ------- weighting params -------
    wm, wc, gamma

    Line by line by book Table 3.4
    """
    


    #  Predict
    # (3x7)
    X_tm1 = generate_sigma_points(mu_tm1, Sigma_tm1, gamma)                 # alg line 2
    # (3x7)
    X_t_barstar = predict_sigma_motion(u_t, X_tm1, dt)                      # alg line 3
    # (3x1)
    mu_t_bar = (wm @ X_t_barstar.T).T                                       # alg line 4
    # (3x3)
    Sigma_t_bar = calc_covariance(mu_t_bar, X_t_barstar, wc, R)             # alg line 5
    # instability fix
    if np.max(np.abs(Sigma_t_bar)) > 2:
        Sigma_t_bar /= np.max(np.abs(Sigma_t_bar))
    
    # Update 
    # (3x7)
    X_t_bar = generate_sigma_points(mu_t_bar, Sigma_t_bar, gamma)           # alg line 6
    # (30x7)
    Zeta_t_bar = predict_sigma_observation(X_t_bar, landmark_gt)            # alg line 7: a predicted obsv is computed for each sigma point (to verify)
    
    # # (30x1) for z_t
    # # replace the elements in z_t where there were no observation with the average from the r, phi of sigma points estimates from line 7
    # for i in range(z_t.shape[0]):
    #     if z_t[i, 0] == 0:
    #         z_t[i, 0] = Zeta_t_bar[i, :].mean()        # takes the average
    
    # continue with update
    # (30x1)
    z_t_hat = (wm @ Zeta_t_bar.T).T                                         # alg line 8
    # (30x30)
    S_t = calc_covariance(z_t_hat, Zeta_t_bar, wc, Q)                       # alg line 9 
    # (3x30)
    Sigma_t_bar_xz = calc_cross_covariance_xz(X_t_bar, mu_t_bar, Zeta_t_bar, z_t_hat, wc)   # alg line 10
    # (3x30)
    K_t = Sigma_t_bar_xz @ np.linalg.inv(S_t)                               # alg line 11
    # (3x1)
    mu_t = mu_t_bar + K_t @ (z_t - z_t_hat)                                 # alg line 12
    # (3x3)
    Sigma_t = Sigma_t_bar - K_t @ S_t @ K_t.T                               # alg line 13

    return mu_t, Sigma_t



def setup_ukf(nx):
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




# get initial measurement of all landmarks from starting position (K is sorted by subj number)
def get_landmark_measurement_from_robo_posn(landmark_df, robo_posn):
    """
    input: takes the df and a (3,) vector representing the robot's position
    for each landmark, compute the expected measurement from the robot's position
    final res is a 2K x 1 vector (30x1)
    """
    robo_x, robo_y, robo_theta = robo_posn
    msmnts = []

    for _, row in landmark_df.iterrows():
        ldmk_x, ldmk_y = row["x [m]"], row["y [m]"]
        r, phi = observation_model(ldmk_x, ldmk_y, robo_x, robo_y, robo_theta)
        phi = np.arctan2(ldmk_y - robo_y, ldmk_x - robo_y)
        msmnts += [r, phi]
    return np.array(msmnts).reshape(-1, 1)


# checks if a row from the combined df is an action (next action) or not
def is_action_row(row):
    return not pd.isna(row['Forward Velocity [m/s]']) and not pd.isna(row['Angular Velocity [rad/s]'])



def run_ukf(dataset, mu_tm1, Sigma_tm1, z_t, u_t, wm, wc, gamma, R, Q, dt, landmark_gt, M):
    """
    mu_tm1, Sigma_tm1, z_t, u_t, wm, wc, gamma, R, Q, dt, 
    landmark_gt, 
    M: last M steps to plot
    
    """
    # i_actions means run up to the ith_action step and stop
    if dataset == "v2":
        # this is the dataset with 5121 actions
        all_data = pd.read_csv("action-update_v2.csv")


    elif dataset == "v3":
        # this is the dataset with u_t = [0,0] removed
        all_data = pd.read_csv("action-update_v3.csv")
    else:
        print("Invalid dataset")
        return


def draw_animation(hist_gt, hist_dr, hist_mu, ldmk_xcoords, ldmk_ycoords, M=0, **kwargs):
    """
    draws the animation of the robot's path with last M steps where M=0 means all steps
    and **kwargs may hold the parameters for the run (used in title)
    """
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
    # plot the true posn of robot
    plt.plot(np.array(hist_gt[0, -M:]).flatten(),
                np.array(hist_gt[1, -M:]).flatten(), "-b", label="Ground Truth")
    # plot posn with symbol
    plt.plot(hist_gt[0, -1], hist_gt[1, -1], 'D', markersize=3, color='b')

    # plot the dead reckoning posn of robot
    plt.plot(np.array(hist_dr[0, -M:]).flatten(),
                np.array(hist_dr[1, -M:]).flatten(), "-k", label="Dead Reckoning")
    # plot posn with symbol
    plt.plot(hist_dr[0, -1], hist_dr[1, -1], 'o', markersize=3, color='k')

    # plot the ukf posn of robot
    plt.plot(np.array(hist_mu[0, -M:]).flatten(),
                np.array(hist_mu[1, -M:]).flatten(), "-r", label="UKF")
    # plot posn with symbol
    plt.plot(hist_mu[0, -1], hist_mu[1, -1], 's', markersize=3, color='r')

    # plot the landmarks
    plt.scatter(ldmk_xcoords, ldmk_ycoords, marker='^', c='orange')

    # plt.axis("equal")
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.gcf().canvas.manager.set_window_title('Timelapse Plot')
    plt.title(f'Run with alpha=, beta=, kappa=, sigma=, R=, Q=.')             # Showing the parameters
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.pause(0.001)
    # plt.pause(0.1)
    # plt.pause(0.5)

def main():
    print("---------------------------------------Start---------------------------------------")

    ### Read Data Files ###
    landmark_gt, gt, msmt, odo = read_data()

    ldmk_xcoords = landmark_gt['x [m]']
    ldmk_ycoords = landmark_gt['y [m]']

    run_ukf("v3", mu_tm1, Sigma_tm1, z_t, u_t, wm, wc, gamma, R, Q, dt, landmark_gt, M=50)


    # Read in the specified data
    all_data = pd.read_csv("action-update.csv")
                        #    names=["Time [s]",                                                  # time index
                        #           "Forward Velocity [m/s]","Angular Velocity [rad/s]",         # odometry
                        #           "Subject #","Range [m]","Bearing [rad]",                     # measurement
                        #           "x [m]","y [m]","Orientation [rad]"])                        # ground truth
    

    nx = 3  # State Vector [x y heading]
    
    
    # # init ground truth and dead reckoning and cur state to known ground truth
    # gt_state = dr_state = mu_tm1 = np.array([gt.iloc[0]["x [m]"], gt.iloc[0]["y [m]"], gt.iloc[0]["Orientation [rad]"]]).reshape(-1, 1)  

    # set init to gt of 1st action step
    gt_state = dr_state = mu_tm1 = np.array([all_data.iloc[1]["x [m]"], all_data.iloc[1]["y [m]"], all_data.iloc[1]["Orientation [rad]"]]).reshape(-1, 1)  
    
    
    # Init covariance matrix is Identity Matrix because (???)
    # if this is too large, then it can make filter trust measurements too much.
    # Sigma_tm1 = np.eye(nx)    
    Sigma_tm1 = np.zeros((nx, nx))
    # Sigma_tm1 = np.diag([0.00001] * nx)
    # Sigma_tm1 = np.diag([0.00000000000000001] * nx)

    # Covariance for UKF simulation
    R = np.diag([
        0.01,  # variance of location on x-axis
        0.01,  # variance of location on y-axis
        np.deg2rad(1.0),  # variance of yaw angle
    ]) ** 2  # predict state covariance

    # R = np.diag([1e-8, 1e-8, 0.001])

    # A large Q means you expect a lot of uncertainty in motion model.
    # Init diagonal 1 at start state where we know all landmarks
    # When not visible, the R for that landmark will be set to large value (1e8)
    # Q = np.eye(30) 
    # Q = np.diag([0.01] * 30)

    # Initial R, Q noise covariance factors
    # Q = np.eye(30) 

    MAX_UNCERTAINTY = 1e10
    OBSERVED_UNCERTAINTY = 1

    wm, wc, gamma = setup_ukf(nx)

    # history
    hist_gt = gt_state
    hist_dr = gt_state
    hist_mu = gt_state

    i = 0
    N = all_data.shape[0]

    # offset such that subj 6->0, and so on so that 15 landmarks are indexed 0-14
    subj_offset = 6

    # num actions taken ctr (expect total of 5121)
    actions_ctr = 1
    # i is always the action index
    while i < N:

        print(f"i: {i}, action {actions_ctr}, state x y theta: {mu_tm1[0]} {mu_tm1[1]} {mu_tm1[2]}")
        actions_ctr += 1

        # ground truth at this time step
        gt_state = np.array([all_data.iloc[i+1]["x [m]"], all_data.iloc[i+1]["y [m]"], all_data.iloc[i+1]["Orientation [rad]"]]).reshape(-1,1)

        # action at this time step
        u_t = np.array([[all_data.iloc[i]["Forward Velocity [m/s]"], all_data.iloc[i]["Angular Velocity [rad/s]"]]]).T

        # Assume the worst for measurement noise
        Q = np.eye(30) * MAX_UNCERTAINTY

        # get measurements for this action
        z_t = np.zeros((30, 1))
        j = i+2
        while j < N and not is_action_row(all_data.iloc[j]):
            # get measurement for each landmark
            jrow = all_data.iloc[j]
            subj = int(jrow["Subject #"])
            ldmk_r, ldmk_phi = jrow["Range [m]"], jrow["Bearing [rad]"]
            z_t[2*(subj-subj_offset)] = ldmk_r
            z_t[2*(subj-subj_offset)+1] = ldmk_phi
            # Trust this measurement by lowering noise in R
            Q[2*(subj-subj_offset), 2*(subj-subj_offset)] = OBSERVED_UNCERTAINTY
            j += 1
        
        # if reached end f experiment, break
        if j >= N:
            break

        # get the dt from next action
        dt = all_data.iloc[j]["Time [s]"] - all_data.iloc[i]["Time [s]"]
        # move to next action step
        i = j
        
        # get dead reckoning at this time step
        dr_state = motion_model(dr_state, u_t, dt)

        # run ukf for this action
        mu_tm1, Sigma_tm1 = ukf_estimation(mu_tm1, Sigma_tm1, z_t, u_t, wm, wc, gamma, R, Q, dt, landmark_gt)

        # store data history
        hist_gt = np.hstack((hist_gt, gt_state))
        hist_dr = np.hstack((hist_dr, dr_state))
        hist_mu = np.hstack((hist_mu, mu_tm1))

        # plot all history or last M steps
        M = 10

        if show_animation:
            draw_animation(hist_gt, hist_dr, hist_mu, ldmk_xcoords, ldmk_ycoords, M)

    if not show_animation:
        # plot the final state here
        draw_animation(hist_gt, hist_dr, hist_mu, ldmk_xcoords, ldmk_ycoords, M)
    
    print("Press escape or ctrl-c to close the plot/animation!")
    while True:
        plt.pause(0.001)
    print("---------------------------------------End---------------------------------------")


if __name__ == '__main__':
    main()

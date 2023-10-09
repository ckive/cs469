import pandas as pd
import numpy as np
import math

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



def normalize_angle(angle):
    """
    Normalize an angle to be between -pi and pi.
    """
    # https://samialperenakgun.com/blog/2022/05/scale_radian_range_python/
    return np.arctan2(np.sin(angle),np.cos(angle))


def residual_h(a, b):
    y = a - b
    # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y

def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


# f(x)
def motion_model(posn, dt, u):
    """
    Update the robot's position using the velocity motion model.
    
    posn = (x, y, theta)
    u = (v, omega)
    dt: Time duration.
    """

    # given posn = (x, y, theta)
    x, y, theta = posn
    
    # given vector u, it contains the velocity and angular velocity
    v, omega = u
    
    # If the robot is moving approximately straight
    if abs(omega) < 0.001:
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        # Orientation remains the same when omega is zero
    else:
        x += (v/omega) * (math.sin(theta + omega*dt) - math.sin(theta))
        y += (v/omega) * (math.cos(theta) - math.cos(theta + omega*dt))
        theta += omega * dt
        theta = normalize_angle(theta)

    # make sure it's a (1x3)vector
    return np.array([x, y, theta]).T


# helper function to get the measurement for a landmark with cur posn
def calc_rphi_1landmark(ldmk_x, ldmk_y, x, y, theta, 
                      sigma_range=None,
                      sigma_bearing=None):
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

    if sigma_range is None and sigma_range is None:
        r_expected = math.sqrt(delta_x**2 + delta_y**2)
        phi_expected = normalize_angle(math.atan2(delta_y, delta_x) - theta)
    else:
        # with noise?
        r_expected = math.sqrt(delta_x**2 + delta_y**2 +np.random.randn()*sigma_range)
        phi_expected = normalize_angle(math.atan2(delta_y, delta_x) - theta + np.random.randn()*sigma_bearing)
    
    return r_expected, phi_expected

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = math.atan2(sum_sin, sum_cos)
    return x


def is_action_row(row):
    return not pd.isna(row['Forward Velocity [m/s]']) and not pd.isna(row['Angular Velocity [rad/s]'])


# h(x)
# get measurement of all landmarks from a position(state) (K is sorted by subj number)
def get_landmark_measurements_from_state(state, landmark_df):
    """
    input: takes the df and a (3,) vector representing the robot's position
    for each landmark, compute the expected measurement from the robot's position
    final res is a 2K x 1 vector (30x1)
    """
    robo_x, robo_y, robo_theta = state
    msmnts = []

    for _, row in landmark_df.iterrows():
        ldmk_x, ldmk_y = row["x [m]"], row["y [m]"]
        r, phi = calc_rphi_1landmark(ldmk_x, ldmk_y, robo_x, robo_y, robo_theta)
        phi = np.arctan2(ldmk_y - robo_y, ldmk_x - robo_y)
        msmnts += [r, phi]
    
    # return a (30,) vector (or should it be 1x30?) (TODO: check this)
    return np.array(msmnts)
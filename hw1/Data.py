import pandas as pd
import matplotlib.pyplot as plt
import math


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

# -------------------------------------------------------------------------------

# Define a small time unit
EPSILON = 0.001

# for ignoring measurements of other robots
ROBOT_BARCODE_NUMS = [5, 14, 41, 32, 23]

# Combining all data and sorting
all_data = pd.concat([odo, msmt])
all_data = all_data.sort_values(by='Time [s]')

# New dataframe to hold the reordered data
restructured_data = []

# Use a list to keep track of the set of measurements after an action step
last_measurements = []

# Iterate through the combined data

def is_action_row(row):
    return not pd.isna(row['Forward Velocity [m/s]']) and not pd.isna(row['Angular Velocity [rad/s]'])

N = len(all_data)
i = 0
fivepct = N // 5
fivepctctr = 0
print(N)
while i < N:
    if i == 17689:
        print('hi')
    print(i, N)
    # if i % fivepct == 0:
    #     print(fivepctctr * 5)
    #     fivepctctr += 1

    # double check i is an action row
    row = all_data.iloc[i]

    # Ensure we start on an action row
    # Ensure we start on an action row (that has nonzero v,w)
    if is_action_row(row) and row["Angular Velocity [rad/s]"] != 0 and row["Forward Velocity [m/s]"] != 0:
        # Get the closest groundtruth
        closest_gt_idx = (gt['Time [s]'] - row['Time [s]']).abs().idxmin()
        closest_gt = gt.loc[closest_gt_idx].copy()
        # Adjust the timestamp to ensure order
        closest_gt['Time [s]'] = row['Time [s]'] + EPSILON

        restructured_data.append(row)
        restructured_data.append(closest_gt)

        # incr forward, gather measurements or drop actions

        # we have new measurements as last_measurements now
        row_had_msmnts = False
        j = i + 1
        if not is_action_row(all_data.iloc[j]):
            last_measurements = []
            row_had_msmnts = True
        while j < N:
            jrow = all_data.iloc[j]
            if is_action_row(jrow):
                if not row_had_msmnts:
                    # check if if has same value as our current row, if so skip it
                    if jrow['Forward Velocity [m/s]'] == row['Forward Velocity [m/s]'] and jrow['Angular Velocity [rad/s]'] == row['Angular Velocity [rad/s]']:
                        # incr j
                        j += 1
                    else: # we have a different action row, need to use the prev measurements for this action row
                        # incr i
                        i = j
                        break
                else:
                    # break (we had action row, had msmnts, now see new action row)
                    i = j
                    break
            else: # this is a measurement row we keep incr j until we dont find a measurement row
                if row["Subject #"] not in ROBOT_BARCODE_NUMS:
                    # map the barcode# in row to subject# and add it
                    jrow["Subject #"] = barcode2subject[jrow["Subject #"]]
                    last_measurements.append(jrow)
                    row_had_msmnts = True
                # otherwise ignore it
                j += 1
                
        
        # add the measurements for this action row to restructured data
        restructured_data += last_measurements

        if j >= N:
            # reached end of data in inner loop
            break
    else:
        # incr i
        i += 1

# completed
# Convert back to a DataFrame
restructured_df = pd.DataFrame(restructured_data)#.sort_values(by='Time [s]')

dd = restructured_df.reset_index(drop=True)
dd.to_csv('action-update.csv', index=False)
print('done')
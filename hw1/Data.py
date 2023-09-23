"""
Data Ingestion
"""
import pandas as pd

class RobotData:
    odometry_column_names = ["Time [s]", "Forward Velocity [m/s]", "Angular Velocity [rad/s]"]
    groundtruth_column_names = ["Time [s]", "x [m]", "y [m]", "Orientation [rad]"]
    msmt_column_names = ["Time [s]", "Subject ID", "Range [m]", "Bearing [rad]"]

    def __init__(self, dataset, ith):
        self.ith = ith
        self.odo = pd.read_csv(f"{dataset}/Robot{ith}_Odometry.dat", sep="\s+", skiprows=4, names=self.groundtruth_column_names)
        self.gt = pd.read_csv(f"{dataset}/Robot{ith}_Groundtruth.dat", sep="\s+", skiprows=4, names=self.groundtruth_column_names)
        self.msmt = pd.read_csv(f"{dataset}/Robot{ith}_Measurement.dat", sep="\s+", skiprows=4, names=self.groundtruth_column_names)

    def __repr__(self) -> str:
        return f"Robot {self.ith}"

class DataSet:
    """
    Data Ingestion
    """
    def __init__(self, dataset):
        # NOTE: 

        # set up subject 2 barcode mapping
        self.dataset = dataset
        self.subject2barcode = {}
        with open(f"{dataset}/Barcodes.dat") as f:
            for _ in range(4):
                next(f)
            
            for line in f:
                subject, barcode = map(int, line.strip().split())
                self.subject2barcode[subject] = barcode

        # set up landmark gts
        self.landmark_gt = pd.read_csv(f"{dataset}/Landmark_Groundtruth.dat", sep="\s+", skiprows=4, 
                                       names=["Subject #", "x [m]", "y [m]", "x std-dev [m]", "y std-dev [m]"])

        for i in range(1, 6):
            setattr(self, f"robot{i}", RobotData(dataset, i))

    def __repr__(self) -> str:
        return self.dataset
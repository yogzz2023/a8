import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('test.csv')

# Extract relevant columns
track_id_col = data.iloc[:, 0]
measurement_range_col = data.iloc[:, 11]
measurement_azimuth_col = data.iloc[:, 8]
measurement_elevation_col = data.iloc[:, 9]
measurement_time_col = data.iloc[:, 10]  # Assuming column index 10 for measurement time

# Group measurements by track ID
grouped_data = data.groupby(track_id_col)

# Kalman Filter parameters
dt = 1  # Time step
A = np.array([[1, dt, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, dt, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 0, 0, 1]])  # State transition matrix (constant velocity)

# Initialize the state vector x with zeros
x = np.zeros((8, 1))

# Initialize the state covariance matrix P with identity matrix
P = np.eye(8) * 1000  # Initial state covariance

# Measurement matrix
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0]])  # Measurement matrix

# Process noise covariance matrix
Q = np.eye(8) * 0.1  # Process noise covariance matrix

# Measurement noise covariance matrix (you need to define this)
R = np.eye(4) * 1  # Measurement noise covariance matrix

# Iterate through tracks
for track_id, group in grouped_data:
    track_measurements = group[['M_rng1', 'M_az', 'M_el', 'M_time']]

    # Kalman filtering and JPDA
    for measurement in track_measurements.values:
        # Print current state vector for debugging
        print("State vector x:", x.flatten())

        # Prediction step
        x = np.dot(A, x)
        P = np.dot(np.dot(A, P), A.T) + Q

        # Extract predicted values
        predicted_range = x[0, 0]
        predicted_azimuth = x[2, 0]
        predicted_elevation = x[4, 0]
        predicted_time = x[6, 0]

        # Print predicted values for debugging
        print("Predicted Range:", predicted_range)
        print("Predicted Azimuth:", predicted_azimuth)
        print("Predicted Elevation:", predicted_elevation)
        print("Predicted Time:", predicted_time)

        # Measurement update step
        if np.any(np.isnan(measurement)):  # Check if measurement is available
            continue  # Skip this iteration if measurement is not available

        # Reshape measurement to column vector
        z = measurement.reshape(-1, 1)

        # Compute innovation (measurement prediction error)
        y = z - np.dot(H, x)

        # Compute innovation covariance
        S = np.dot(np.dot(H, P), H.T) + R

        # Compute Kalman gain
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

        # Update state estimate
        x = x + np.dot(K, y)

        # Update state covariance
        P = P - np.dot(np.dot(K, H), P)

        # Store or visualize filtered track estimates

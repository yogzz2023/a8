import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('test.csv')

# Extract relevant columns
track_id_col = data.iloc[:, 0]
measurement_range_col = data.iloc[:, 11]
measurement_azimuth_col = data.iloc[:, 8]
measurement_elevation_col = data.iloc[:, 9]
measurement_time_col = data.iloc[:, 11]

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
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0]])  # Measurement matrix
Q = np.eye(8) * 0.1  # Process noise covariance matrix
R = np.eye(4) * 1  # Measurement noise covariance matrix

# Iterate through tracks
for track_id, group in grouped_data:
    track_measurements = group[['M_rng1', 'M_az', 'M_el', 'M_time']]
    
    # Initialize Kalman filter
    x = np.zeros((8, 1))  # Initial state estimate
    P = np.eye(8) * 1000  # Initial state covariance
    
    # Initialize dictionaries to store track estimates
    track_estimates = {}

    # Kalman filtering and JPDA
    for measurement in track_measurements.values:
        # Prediction step
        x = np.dot(A, x)
        P = np.dot(np.dot(A, P), A.T) + Q
        
        # Extract predicted values
        predicted_range = x[0, 0]
        predicted_azimuth = x[2, 0]
        predicted_elevation = x[4, 0]
        predicted_time = x[6, 0]
        
        # Print predicted values
        print("Predicted Range:", predicted_range)
        print("Predicted Azimuth:", predicted_azimuth)
        print("Predicted Elevation:", predicted_elevation)
        print("Predicted Time:", predicted_time)
        
        # Compute conditional probabilities
        conditional_probs = {}
        for track_id, track_estimate in track_estimates.items():
            x_pred = track_estimate['state_estimate']
            P_pred = track_estimate['state_covariance']
            y = measurement.reshape(-1, 1) - np.dot(H, x_pred)
            S = np.dot(np.dot(H, P_pred), H.T) + R
            mahalanobis_dist = np.dot(np.dot(y.T, np.linalg.inv(S)), y)
            conditional_prob = np.exp(-0.5 * mahalanobis_dist) / np.sqrt(np.linalg.det(2 * np.pi * S))
            conditional_probs[track_id] = conditional_prob

        # Compute marginal probabilities
        marginal_probs = {}
        for i, measurement in enumerate(track_measurements):
            marginal_probs[i] = sum(conditional_probs.values())

        # Perform association
        associations = {}
        for i, measurement in enumerate(track_measurements):
            best_track_id = max(conditional_probs, key=lambda track_id: conditional_probs[track_id])
            associations[i] = best_track_id

        # Update target states
        for i, measurement in enumerate(track_measurements):
            track_id = associations[i]
            if track_id is not None:
                x_pred = track_estimates[track_id]['state_estimate']
                P_pred = track_estimates[track_id]['state_covariance']
                z = measurement.reshape(-1, 1)
                y = z - np.dot(H, x_pred)
                S = np.dot(np.dot(H, P_pred), H.T) + R
                K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
                x_update = x_pred + np.dot(K, y)
                P_update = np.dot((np.eye(8) - np.dot(K, H)), P_pred)
                track_estimates[track_id]['state_estimate'] = x_update
                track_estimates[track_id]['state_covariance'] = P_update
        
        # Find the measurement associated with the target with the highest marginal probability
        most_likely_measurement_index = max(range(len(marginal_probs)), key=lambda i: marginal_probs[i])
        most_likely_measurement = track_measurements.values[most_likely_measurement_index]

        # Print the most likely measurement associated with the target
        print("Most Likely Measurement Associated with Track ID", track_id, ":", most_likely_measurement)
        
        # Store or visualize filtered track estimates

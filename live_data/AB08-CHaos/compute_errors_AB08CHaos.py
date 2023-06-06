import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os, sys
from scipy.signal import butter, lfilter

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')

from training_utils import calculate_gait_state_errors

filenames = [
            'circuit1/circuit1_seg1/exoboot_Vicon_AB08-CHaos_circuit1_seg1_processed.csv',
            'circuit1/circuit1_seg2/exoboot_Vicon_AB08-CHaos_circuit1_seg2_processed.csv',
            'circuit1/circuit1_seg3/exoboot_Vicon_AB08-CHaos_circuit1_seg3_processed.csv',
            'circuit1/circuit1_seg4/exoboot_Vicon_AB08-CHaos_circuit1_seg4_processed.csv',
            'circuit2/circuit2_seg1/exoboot_Vicon_AB08-CHaos_circuit2_seg1_processed.csv',
            'circuit2/circuit2_seg2/exoboot_Vicon_AB08-CHaos_circuit2_seg2_processed.csv',
            'circuit2/circuit2_seg3/exoboot_Vicon_AB08-CHaos_circuit2_seg3_processed.csv',
            'circuit2/circuit2_seg4/exoboot_Vicon_AB08-CHaos_circuit2_seg4_processed.csv',
            'circuit3/circuit3_seg1/exoboot_Vicon_AB08-CHaos_circuit3_seg1_processed.csv',
            'circuit3/circuit3_seg2/exoboot_Vicon_AB08-CHaos_circuit3_seg2_processed.csv',
            'circuit3/circuit3_seg3/exoboot_Vicon_AB08-CHaos_circuit3_seg3_processed.csv',
            'circuit3/circuit3_seg4/exoboot_Vicon_AB08-CHaos_circuit3_seg4_processed.csv'
            ]


phase_rmses = []
speed_rmses = []
incline_rmses = []
stair_height_accuracies = []

phase_rmses_ss = []
speed_rmses_ss = []
incline_rmses_ss = []
stair_height_accuracies_ss = []

phase_rmses_nss = []
speed_rmses_nss = []
incline_rmses_nss = []
stair_height_accuracies_nss = []


Ns = []
start_idx = 200
predictions_total = np.array([])
true_labels_total = np.array([])
is_steady_state_total = np.array([])

for i, filename in enumerate(filenames):

    df = pd.read_csv(filename)
    # print(df.head())

    dt = df['dt'].to_numpy()
    phase_ground_truth = df['phase_ground_truth'].to_numpy().reshape(-1,1)
    speed_ground_truth = df['speed_ground_truth'].to_numpy().reshape(-1,1)
    incline_ground_truth = df['incline_ground_truth'].to_numpy().reshape(-1,1)
    stairs_ground_truth = df['stairs_ground_truth'].to_numpy().reshape(-1,1)

    phase_hardware = df['phase_hardware'].to_numpy().reshape(-1,1)
    speed_hardware = df['speed_hardware'].to_numpy().reshape(-1,1)
    incline_hardware = df['incline_hardware'].to_numpy().reshape(-1,1)
    stairs_hardware = df['stairs_hardware'].to_numpy().reshape(-1,1)

    is_steady_state = df['is_steady_state_ground_truth'].to_numpy().reshape(-1,1)
    phase_events = df['phase_events_ground_truth'].to_numpy().reshape(-1,1)

    predictions = np.hstack((phase_hardware, speed_hardware, incline_hardware, stairs_hardware))
    true_labels = np.hstack((phase_ground_truth, speed_ground_truth, incline_ground_truth, stairs_ground_truth))

    #only consider the predictions starting from start_idx
    predictions = predictions[start_idx:,:]
    true_labels = true_labels[start_idx:,:]
    is_steady_state = is_steady_state[start_idx:,:]

    #reshape phase events to be 1D so np.nonzeros returns a 1D array
    phase_events = phase_events[start_idx:,:].reshape(-1)

    # if i == 0:
    #     predictions_total = predictions
    #     true_labels_total = true_labels
    #     is_steady_state_total = is_steady_state
    # else:
    #     predictions_total = np.vstack((predictions_total, predictions))
    #     true_labels_total = np.vstack((true_labels_total, true_labels))
    #     is_steady_state_total = np.vstack((is_steady_state_total, is_steady_state))


    phase_event_idxs = np.nonzero(phase_events == 1)
    #insert idxs for beginning and end of trial
    phase_event_idxs = np.insert(phase_event_idxs,0,0)
    phase_event_idxs = np.append(phase_event_idxs,len(phase_events)-1)

    #extract number of phase events
    num_phase_events = len(phase_event_idxs)
    # print(num_phase_events)
    # print(phase_event_idxs)
    # input()
    for i in range(num_phase_events-1):
        current_idx = phase_event_idxs[i]
        next_idx = phase_event_idxs[i+1]

        # print(f'current_idx: {current_idx}')
        # print(f'next_idx: {next_idx}')

        predictions_step = predictions[current_idx:next_idx,:]
        true_labels_step = true_labels[current_idx:next_idx,:]
        is_steady_state_step = is_steady_state[current_idx:next_idx,:]

        #calculate RMSE for each stride
        (phase_rmse_step, 
        speed_rmse_step, 
        incline_rmse_step, 
        stair_height_accuracy_step, 
        stair_height_accuracy_ascent_step, 
        stair_height_accuracy_descent_step) = calculate_gait_state_errors(predictions_step, true_labels_step, STAIRS_THRESHOLD_ROUND=0.5, DO_PRINT=False)

        #update RMSE overall vector
        phase_rmses.append(phase_rmse_step)
        speed_rmses.append(speed_rmse_step)
        incline_rmses.append(incline_rmse_step)
        stair_height_accuracies.append(stair_height_accuracy_step)

        IS_STEADY_STATE = np.mean(is_steady_state_step) >= 0.5
        # print(IS_STEADY_STATE)
        # input()

        if IS_STEADY_STATE:
            phase_rmses_ss.append(phase_rmse_step)
            speed_rmses_ss.append(speed_rmse_step)
            incline_rmses_ss.append(incline_rmse_step)
            stair_height_accuracies_ss.append(stair_height_accuracy_step)

        else:
            phase_rmses_nss.append(phase_rmse_step)
            speed_rmses_nss.append(speed_rmse_step)
            incline_rmses_nss.append(incline_rmse_step)
            stair_height_accuracies_nss.append(stair_height_accuracy_step)


#extract steady state predictions
# predictions_ss = predictions_total[is_steady_state_total.reshape(-1) == 1,:]
# true_labels_ss = true_labels_total[is_steady_state_total.reshape(-1) == 1,:]

# #extract non steady state predictions
# predictions_nss = predictions_total[is_steady_state_total.reshape(-1) != 1,:]
# true_labels_nss = true_labels_total[is_steady_state_total.reshape(-1) != 1,:]

# print(predictions_total.shape)
# print(predictions_ss.shape)
# print(predictions_nss.shape)

# (phase_rmse, 
#     speed_rmse, 
#     incline_rmse, 
#     stair_height_accuracy, 
#     stair_height_accuracy_ascent, 
#     stair_height_accuracy_descent) = calculate_gait_state_errors(predictions_total, true_labels_total, STAIRS_THRESHOLD_ROUND=0.2)

# (phase_rmse_ss, 
#     speed_rmse_ss, 
#     incline_rmse_ss, 
#     stair_height_accuracy_ss, 
#     stair_height_accuracy_ascent_ss, 
#     stair_height_accuracy_descent_ss) = calculate_gait_state_errors(predictions_ss, true_labels_ss, STAIRS_THRESHOLD_ROUND=0.2)

# (phase_rmse_nss, 
#     speed_rmse_nss, 
#     incline_rmse_nss, 
#     stair_height_accuracy_nss, 
#     stair_height_accuracy_ascent_nss, 
#     stair_height_accuracy_descent_nss) = calculate_gait_state_errors(predictions_nss, true_labels_nss, STAIRS_THRESHOLD_ROUND=0.2)


phase_rmses = np.array(phase_rmses)
speed_rmses = np.array(speed_rmses)
incline_rmses = np.array(incline_rmses)
stair_height_accuracies = np.array(stair_height_accuracies)

phase_rmses_ss = np.array(phase_rmses_ss)
speed_rmses_ss = np.array(speed_rmses_ss)
incline_rmses_ss = np.array(incline_rmses_ss)
stair_height_accuracies_ss = np.array(stair_height_accuracies_ss)

phase_rmses_nss = np.array(phase_rmses_nss)
speed_rmses_nss = np.array(speed_rmses_nss)
incline_rmses_nss = np.array(incline_rmses_nss)
stair_height_accuracies_nss = np.array(stair_height_accuracies_nss)



print('Overall')
print(f'phase_loss_avg: {np.mean(phase_rmses)} +- {np.std(phase_rmses)}')
print(f'speed_loss_avg: {np.mean(speed_rmses)} +- {np.std(speed_rmses)}')
print(f'incline_loss_avg: {np.mean(incline_rmses)} +- {np.std(incline_rmses)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies)} +- {np.std(stair_height_accuracies)}')
print()

print('Steady State')
print(f'phase_loss_avg: {np.mean(phase_rmses_ss)} +- {np.std(phase_rmses_ss)}')
print(f'speed_loss_avg: {np.mean(speed_rmses_ss)} +- {np.std(speed_rmses_ss)}')
print(f'incline_loss_avg: {np.mean(incline_rmses_ss)} +- {np.std(incline_rmses_ss)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies_ss)} +- {np.std(stair_height_accuracies_ss)}')
print()

print('Transitory')
print(f'phase_loss_avg: {np.mean(phase_rmses_nss)} +- {np.std(phase_rmses_nss)}')
print(f'speed_loss_avg: {np.mean(speed_rmses_nss)} +- {np.std(speed_rmses_nss)}')
print(f'incline_loss_avg: {np.mean(incline_rmses_nss)} +- {np.std(incline_rmses_nss)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies_nss)} +- {np.std(stair_height_accuracies_nss)}')
print()




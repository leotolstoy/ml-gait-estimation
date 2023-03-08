import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os, sys

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')

from training_utils import calculate_gait_state_errors, phase_dist
from sim_other_models import sim_other_models

filenames = [
            'circuit1/circuit1_seg1/20230221-18_AB09-Brand-circuit1_seg1.csv',
            'circuit1/circuit1_seg2/20230221-18_AB09-Brand-circuit1_seg2.csv',
            'circuit1/circuit1_seg3/20230221-18_AB09-Brand-circuit1_seg3.csv',
            'circuit1/circuit1_seg4/20230221-18_AB09-Brand-circuit1_seg4.csv',
            'circuit2/circuit2_seg1/20230221-18_AB09-Brand-circuit2_seg1.csv',
            'circuit2/circuit2_seg2/20230221-18_AB09-Brand-circuit2_seg2.csv',
            'circuit2/circuit2_seg3/20230221-18_AB09-Brand-circuit2_seg3.csv',
            'circuit2/circuit2_seg4/20230221-18_AB09-Brand-circuit2_seg4.csv',
            'circuit3/circuit3_seg1/20230221-18_AB09-Brand-circuit3_seg1.csv',
            'circuit3/circuit3_seg2/20230221-18_AB09-Brand-circuit3_seg2.csv',
            'circuit3/circuit3_seg3/20230221-18_AB09-Brand-circuit3_seg3.csv',
            'circuit3/circuit3_seg4/20230221-18_AB09-Brand-circuit3_seg4.csv',
            ]

vicon_filenames = [
            'circuit1/circuit1_seg1/exoboot_Vicon_AB09-Brand_circuit1_seg1_processed.csv',
            'circuit1/circuit1_seg2/exoboot_Vicon_AB09-Brand_circuit1_seg2_processed.csv',
            'circuit1/circuit1_seg3/exoboot_Vicon_AB09-Brand_circuit1_seg3_processed.csv',
            'circuit1/circuit1_seg4/exoboot_Vicon_AB09-Brand_circuit1_seg4_processed.csv',
            'circuit2/circuit2_seg1/exoboot_Vicon_AB09-Brand_circuit2_seg1_processed.csv',
            'circuit2/circuit2_seg2/exoboot_Vicon_AB09-Brand_circuit2_seg2_processed.csv',
            'circuit2/circuit2_seg3/exoboot_Vicon_AB09-Brand_circuit2_seg3_processed.csv',
            'circuit2/circuit2_seg4/exoboot_Vicon_AB09-Brand_circuit2_seg4_processed.csv',
            'circuit3/circuit3_seg1/exoboot_Vicon_AB09-Brand_circuit3_seg1_processed.csv',
            'circuit3/circuit3_seg2/exoboot_Vicon_AB09-Brand_circuit3_seg2_processed.csv',
            'circuit3/circuit3_seg3/exoboot_Vicon_AB09-Brand_circuit3_seg3_processed.csv',
            'circuit3/circuit3_seg4/exoboot_Vicon_AB09-Brand_circuit3_seg4_processed.csv'
            ]

phase_rmses_ekf = []
speed_rmses_ekf = []
incline_rmses_ekf = []

phase_rmses_tbe = []
start_idx = 200


for i, vicon_filename in enumerate(vicon_filenames):

    df = pd.read_csv(vicon_filename)
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

    
    #extract raw full data from exoboot
    data = np.loadtxt(filenames[i], delimiter=',')
    data = data[start_idx:,:]
    time_exo = data[:,0]
    dts_exo = np.diff(time_exo)
    dts_exo = np.insert(dts_exo,0,dts_exo[0])

    #SET UP SUBJECT LEG LENGTH
    SUBJECT_LEG_LENGTH = 0.935 #AB09
    torque_profile_path = '../../torque_profile/torque_profile_coeffs.csv'
    gait_model_covar_path = '../../old_ekf_funcs/covar_fourier_normalizedsL_linearsL.csv'
    gait_model_path = '../../old_ekf_funcs/gaitModel_fourier_normalizedsL_linearsL.csv'



    
    phase_event_idxs = np.nonzero(phase_events == 1)
    #insert idxs for beginning and end of trial
    phase_event_idxs = np.insert(phase_event_idxs,0,0)
    phase_event_idxs = np.append(phase_event_idxs,len(phase_events)-1)

    #extract number of phase events
    num_phase_events = len(phase_event_idxs)
    # print(num_phase_events)
    # print(phase_event_idxs)
    # input()

    #extract only the data that is moving for TBE
    tbe_mask = speed_ground_truth[start_idx:,:].reshape(-1) >= 0.05
    tbe_idxs = np.argwhere(tbe_mask)
    # print(tbe_idxs)
    # print(len(tbe_idxs))
    # print(tbe_idxs[0], tbe_idxs[-1])
    data_tbe = data[tbe_mask,:]
    #overwrite the time vector of the data at position zero
    dts_exo_tbe = dts_exo[tbe_mask]
    time_exo_tbe = np.cumsum(dts_exo_tbe)
    data_tbe[:,0] = time_exo_tbe
    true_labels_tbe = true_labels[tbe_mask,:]
    is_steady_state_tbe = is_steady_state[tbe_mask,:]

    #remove the phase events that aren't in the range of the TBE data
    phase_events_tbe = phase_events[tbe_mask]
    phase_event_idxs_tbe = np.nonzero(phase_events_tbe == 1)[0]

    #extract number of phase events for tbe
    num_phase_events_tbe = len(phase_event_idxs_tbe)

    # print(phase_event_idxs_tbe)

    #extract data that isn't stairs and is moving for old EKF
    ekf_mask = np.logical_and( (np.abs(stairs_ground_truth[start_idx:,:].reshape(-1)) <= 0.5), (speed_ground_truth[start_idx:,:].reshape(-1) >= 0.05)) 
    ekf_idxs = np.argwhere(ekf_mask)
    # print(ekf_idxs)
    # print(len(ekf_idxs))
    # print(ekf_idxs[0], ekf_idxs[-1])
    data_ekf = data[ekf_mask,:]
    #overwrite the time vector of the data at position zero
    dts_exo_ekf = dts_exo[ekf_mask]
    time_exo_ekf = np.cumsum(dts_exo_ekf)
    data_ekf[:,0] = time_exo_ekf
    true_labels_ekf = true_labels[ekf_mask,:]
    is_steady_state_ekf = is_steady_state[ekf_mask,:]

    #remove the phase events that aren't in the range of the EKF data
    phase_events_ekf = phase_events[ekf_mask]
    phase_event_idxs_ekf = np.nonzero(phase_events_ekf == 1)[0]
    # print(phase_event_idxs_ekf)
    #extract number of phase events for tbe
    num_phase_events_ekf = len(phase_event_idxs_ekf)

    # input()

    phase_sim_ekf, speed_sim_ekf, incline_sim_ekf, _ = sim_other_models(data_ekf, SUBJECT_LEG_LENGTH, torque_profile_path, gait_model_covar_path, gait_model_path, DO_PLOTS=not True)
    _, _, _, phase_sim_tbe = sim_other_models(data_tbe, SUBJECT_LEG_LENGTH, torque_profile_path, gait_model_covar_path, gait_model_path, DO_PLOTS=not True)

    if not True:
        fig, axs = plt.subplots(3,1)
        axs[0].plot(phase_sim_ekf,'r',label='EKF')
        axs[0].plot(true_labels_ekf[:,0],'b',label='ground truth')
        axs[0].plot(phase_events_ekf,'k',label='phase_events_ekf')
        axs[0].legend()
        axs[1].plot(speed_sim_ekf,'r')
        axs[1].plot(true_labels_ekf[:,1],'b')

        axs[2].plot(incline_sim_ekf,'r')
        axs[2].plot(true_labels_ekf[:,2],'b')

        fig, axs = plt.subplots()
        axs.plot(phase_sim_tbe,'r',label='TBE')
        axs.plot(true_labels_tbe[:,0],'b',label='ground truth')
        axs.plot(phase_events_tbe,'k',label='phase_events_tbe')
        axs.legend()

        plt.show()

    # print(len(speed_sim_ekf))
    #run through ekf data
    for i in range(num_phase_events_ekf-1):
        current_idx = phase_event_idxs_ekf[i]
        next_idx = phase_event_idxs_ekf[i+1]

        # print(f'current_idx: {current_idx}')
        # print(f'next_idx: {next_idx}')

        phase_sim_ekf_step = phase_sim_ekf[current_idx:next_idx,:].reshape(-1)
        speed_sim_ekf_step = speed_sim_ekf[current_idx:next_idx,:].reshape(-1)
        incline_sim_ekf_step = incline_sim_ekf[current_idx:next_idx,:].reshape(-1)

        true_labels_step = true_labels_ekf[current_idx:next_idx,:]

        if not True:
            fig, axs = plt.subplots(3,1)
            axs[0].plot(phase_sim_ekf_step,'r',label='EKF')
            axs[0].plot(true_labels_step[:,0],'b',label='ground truth')
            axs[0].legend()
            axs[1].plot(speed_sim_ekf_step,'r')
            axs[1].plot(true_labels_step[:,1],'b')

            axs[2].plot(incline_sim_ekf_step,'r')
            axs[2].plot(true_labels_step[:,2],'b')
            
            # plt.show()

        phase_rmse_step = np.sqrt(np.mean(phase_dist(phase_sim_ekf_step, true_labels_step[:,0])**2))
        speed_rmse_step = np.sqrt(np.mean((speed_sim_ekf_step - true_labels_step[:,1])**2))
        incline_rmse_step = np.sqrt(np.mean((incline_sim_ekf_step - true_labels_step[:,2])**2))


        #update RMSE overall vector
        phase_rmses_ekf.append(phase_rmse_step)
        speed_rmses_ekf.append(speed_rmse_step)
        incline_rmses_ekf.append(incline_rmse_step)
        # input()

    #run through tbe data
    for i in range(num_phase_events_tbe-1):
        current_idx = phase_event_idxs_tbe[i]
        next_idx = phase_event_idxs_tbe[i+1]

        # print(f'current_idx: {current_idx}')
        # print(f'next_idx: {next_idx}')

        phase_sim_tbe_step = phase_sim_tbe[current_idx:next_idx,:].reshape(-1)
        true_labels_step = true_labels_tbe[current_idx:next_idx,:]

        # print(speed_sim_tbe_step)
        phase_rmse_step = np.sqrt(np.mean(phase_dist(phase_sim_tbe_step, true_labels_step[:,0])**2))

        #update RMSE overall vector
        phase_rmses_tbe.append(phase_rmse_step)
        # input()


phase_rmses_ekf = np.array(phase_rmses_ekf)
speed_rmses_ekf = np.array(speed_rmses_ekf)
incline_rmses_ekf = np.array(incline_rmses_ekf)

phase_rmses_tbe = np.array(phase_rmses_tbe)


print('EKF')
print(phase_rmses_ekf)
print(f'phase_loss_avg: {np.mean(phase_rmses_ekf)} +- {np.std(phase_rmses_ekf)}')
print(f'speed_loss_avg: {np.mean(speed_rmses_ekf)} +- {np.std(speed_rmses_ekf)}')
print(f'incline_loss_avg: {np.mean(incline_rmses_ekf)} +- {np.std(incline_rmses_ekf)}')
print()

print('TBE')
print(f'phase_loss_avg: {np.mean(phase_rmses_tbe)} +- {np.std(phase_rmses_tbe)}')
print()






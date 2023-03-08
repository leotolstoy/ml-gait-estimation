"""This file contains numerous utility functions used when processing the live Vicon mocap data
"""
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

from training_utils import phase_dist, calculate_gait_state_errors


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Generate a lowpass filter to filter the speed

    Args:
        data (np array): unfiltered data
        cutoff (float): the cutoff frequency
        fs (float): The sampling frequency
        order (int, optional): The order of the lowpass filter. Defaults to 5.

    Returns:
        np array: filtered data
    """    
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def returnViconProcessed(filename, AB_SUBJECT, nrows_events,n_skiprows_marker_1,nrows,DO_PLOTS=False,DO_EXPORT=True):
    """This function takes in Vicon file directly and computes gait state quantities. This function
        is highly dependent on the structure of Vicon output files

    Args:
        filename (string): the unprocessed Vicon filename
        AB_SUBJECT (string): the name of the subject in the vicon file
        nrows_events (int): #row number of the last stride event row
        n_skiprows_marker_1 (int): # row num that contains AB0's:
        nrows (int): the number of total rows in the vicon file
        DO_PLOTS (bool, optional): whether to plot results. Defaults to False.
        DO_EXPORT (bool, optional): whether to export results. Defaults to True.

    Returns:
        _type_: _description_
    """    

    # Load in vicon data as a dataframe
    df_events = pd.read_csv(filename,skiprows=2,nrows=nrows_events-3)
    
    # print(df_events.tail())
    df_events = df_events.sort_values(by = 'Time (s)')
    # print(df_events.head())

    #extract events that have to do with phase
    phase_events = df_events.loc[(df_events['Name'] == 'Foot Strike') | 
        (df_events['Name'] == 'Foot Off') | 
        (df_events['Name'] == 'FlatFoot')]

    #extract the events that have to do with changing inclines
    ramp_events = df_events.loc[(df_events['Name'] == 'OnRamp') | 
        (df_events['Name'] == 'OffRamp')]
    
    #extract the events that have to do with changing stairs
    stair_events = df_events.loc[(df_events['Name'] == 'OnStairs') | 
        (df_events['Name'] == 'OffStairs')]

    print(phase_events.head())
    print(ramp_events.head())
    print(stair_events.head())

    right_leg_events = df_events.loc[df_events['Context'] == 'Right']
    HSs = right_leg_events.loc[right_leg_events['Name'] == 'Foot Strike']
    HSs = HSs['Time (s)'].tolist()
    print(HSs)
    # input()

    # MARKER
    n_skiprows_marker_2 = n_skiprows_marker_1+1#contains Frame, subframe
    n_skiprows_marker_3 = n_skiprows_marker_1-1#row before marker names

    # Load in vicon data for markers
    skiprows_vicon = [i for i in range(n_skiprows_marker_1)]
    skiprows_vicon.append(n_skiprows_marker_2)
    df_vicon = pd.read_csv(filename,skiprows=skiprows_vicon,nrows=nrows)

    #read in the Marker names from the raw file
    markernames = pd.read_csv(filename,skiprows=n_skiprows_marker_3,nrows=1)
    markernames = markernames.loc[:,~markernames.columns.str.contains('^Unnamed')]
    markernames = markernames.columns.tolist()
    # print(markernames)
    # print(len(markernames))
    col_rename_vicon = {}

    #extract separate data for the XYZ axes for each marker
    # for later use in a renamed dataframe
    for i in range(len(markernames)*3):

        markername = markernames[i//3]
        # print(markername)
        num = (i+1)//3
        oldstr = ''
        if num == 0:
            pass

        else:
            oldstr = '.'+str(num)

        # print(oldstr)
        # input()
        col_rename_vicon['X' + oldstr] = markername+'.X'
        col_rename_vicon['Y' + oldstr] = markername+'.Y'
        col_rename_vicon['Z' + oldstr] = markername+'.Z'

    df_vicon.rename(columns=col_rename_vicon,inplace=True)

    print(df_vicon.head())
    # print(df_vicon.tail())

    
    frames_vicon = df_vicon['Frame'].to_numpy()
    print('Frames vicon')
    print(len(frames_vicon))

    #compute total time
    time_vicon = np.linspace(frames_vicon[0] - frames_vicon[0],frames_vicon[-1] - frames_vicon[0],len(frames_vicon))/100

    #extract relevant markers in the forward and upward dimension, Y forward, Z upward
    LASI_Y = df_vicon[f'{AB_SUBJECT}:LASI.Y'].to_numpy().reshape(-1,1)/1000
    RASI_Y = df_vicon[f'{AB_SUBJECT}:RASI.Y'].to_numpy().reshape(-1,1)/1000
    LPSI_Y = df_vicon[f'{AB_SUBJECT}:LPSI.Y'].to_numpy().reshape(-1,1)/1000
    RPSI_Y = df_vicon[f'{AB_SUBJECT}:RPSI.Y'].to_numpy().reshape(-1,1)/1000

    RHEE_Y = df_vicon[f'{AB_SUBJECT}:RHEE.Y'].to_numpy().reshape(-1,1)/1000
    RHEE_Z = df_vicon[f'{AB_SUBJECT}:RHEE.Z'].to_numpy().reshape(-1,1)/1000

    RTOE_Y = df_vicon[f'{AB_SUBJECT}:RTOE.Y'].to_numpy().reshape(-1,1)/1000
    RTOE_Z = df_vicon[f'{AB_SUBJECT}:RTOE.Z'].to_numpy().reshape(-1,1)/1000

    #estimate the global foot angle using the arctangent of the heel position
    y = RTOE_Z - RHEE_Z
    x = RTOE_Y - RHEE_Y
    foot_angle_vicon_calcd = np.arctan2(y,x)*180/np.pi


    #calculate speeds of the markersby numerical differentiation
    #0.01 is the time step from vicon
    LASI_Y_VEL = np.diff(LASI_Y,axis=0).reshape(-1,1)/0.01
    RASI_Y_VEL = np.diff(RASI_Y,axis=0).reshape(-1,1)/0.01
    LPSI_Y_VEL = np.diff(LPSI_Y,axis=0).reshape(-1,1)/0.01
    RPSI_Y_VEL = np.diff(RPSI_Y,axis=0).reshape(-1,1)/0.01

    #compute speed by averaging individual marker speeds
    speed_vicon = np.nanmean(np.hstack((LASI_Y_VEL, RASI_Y_VEL, LPSI_Y_VEL, RPSI_Y_VEL)), axis=1)

    #account for any nans left by interpolating between the surrounding nonnan values
    nans_idxs, x = nan_helper(speed_vicon)
    speed_vicon[nans_idxs]= np.interp(x(nans_idxs), x(~nans_idxs), speed_vicon[~nans_idxs])
    speed_vicon = speed_vicon.reshape(-1,1)
    speed_vicon = np.vstack((speed_vicon[0,0],speed_vicon))

    # Filter speeds
    order = 6
    fs = 100       # sample rate, Hz
    cutoff = 10  # desired cutoff frequency of the filter, Hz
    speed_vicon = butter_lowpass_filter(speed_vicon.reshape(-1), cutoff, fs, order)
    speed_vicon = speed_vicon.reshape(-1,1)


    # Plot vicon stuff
    if DO_PLOTS:
        fig1, axs1 = plt.subplots(2,1,sharex=True,figsize=(10,6))
        axs1[0].plot(time_vicon, LASI_Y, label='RANK_X')
        axs1[0].plot(time_vicon, RASI_Y, label='RANK_Y')
        axs1[0].plot(time_vicon, LPSI_Y, label='RANK_Z')
        axs1[0].plot(time_vicon, RPSI_Y, label='RANK_Z')
        axs1[0].set_xlabel("time (sec)")

        axs1[1].plot(time_vicon, speed_vicon, label='speed_vicon')
        axs1[1].set_xlabel("time (sec)")


    #preallocate vectors to hold relevant quantities
    is_steady_state_vicon = np.ones(time_vicon.size) #vector that encodes if frame is on a steady state condition, 1 if steady state, 0 otherwise
    phase_events_vicon = np.zeros(time_vicon.size) #vector that holds booleans, 1 if a phase event happened, 0 otherwise
    phase_vicon = np.zeros(time_vicon.size)#vector that holds values of phase
    

    numSteps = 0
    prev_event_idx = 0
    prevHS_time = 0
    firstIdx = 0

    phase_event_times = phase_events['Time (s)'].tolist()

    #generate phase events into phase_events_vicon
    #this calculates the phases for the strides
    print(len(phase_events))
    for i in range(len(phase_events)-1):
        current_event = phase_events.iloc[i]
        next_event = phase_events.iloc[i+1]
        # print(current_event)
        current_event_time = current_event['Time (s)']
        next_event_time = next_event['Time (s)']

        current_event_idx = np.argmin(np.abs(current_event_time - time_vicon))
        phase_events_vicon[current_event_idx] = 1
        next_event_idx = np.argmin(np.abs(next_event_time - time_vicon))

        is_steady_state = 1
        start_phase = 0
        end_phase = 1
        if current_event['Name'] == 'Foot Off':
            start_phase = 0.65
            is_steady_state = 0

        if next_event['Name'] == 'FlatFoot':
            end_phase = 0.25
            is_steady_state = 0
        elif next_event['Name'] == 'Foot Strike':
            end_phase = 1
        elif next_event['Name'] == 'Foot Off':
            end_phase = 0

        
        phase_vicon[current_event_idx:next_event_idx] = np.linspace(start_phase, end_phase, len(phase_vicon[current_event_idx:next_event_idx]))
        is_steady_state_vicon[current_event_idx:next_event_idx] = is_steady_state

        
    #ensure the end of the trial is steady state
    #final event is a flat-foot contact, so from there to the end is steady state
    current_event = phase_events.iloc[len(phase_events)-1]
    current_event_time = current_event['Time (s)']
    current_event_idx = np.argmin(np.abs(current_event_time - time_vicon))
    phase_events_vicon[current_event_idx] = 1
    is_steady_state_vicon[current_event_idx:] = 1
    
    #generate ramp events, held in incline_vicon
    print(len(ramp_events))
    incline_vicon = np.zeros(time_vicon.size)
    for i in range(len(ramp_events)):
        current_event = ramp_events.iloc[i]
        # print(current_event)
        current_event_time = current_event['Time (s)']
        current_event_idx = np.argmin(np.abs(current_event_time - time_vicon))

        incline = 0
        is_steady_state = 1

        if i < len(ramp_events)-1:
            next_event = ramp_events.iloc[i+1]
            next_event_time = next_event['Time (s)']

            
            next_event_idx = np.argmin(np.abs(next_event_time - time_vicon))

            if current_event['Name'] == 'OnRamp' and next_event['Name'] == 'OffRamp':
                incline = 12.78

            #if we're returning back to the origin, negative speed, decline
            mean_speed = np.mean(speed_vicon[current_event_idx:next_event_idx])
            # print(mean_speed)
            if mean_speed < 0:
                print('GOING DOWNSLOPE')
                incline = -incline
        incline_vicon[current_event_idx:next_event_idx] = incline

        #handle steady state transitions
        
        #find the idx in the list of times of the closest phase event to the ramp event
        closest_phase_event_idx = np.argmin(np.abs(current_event_time - phase_event_times)) 
        

        #find the indices (in the list of times) of the closest phase events where the ramp changes
        prev_phase_event_idx = closest_phase_event_idx - 1
        prev_phase_event_idx = np.max((prev_phase_event_idx, 0))
        prev_phase_event_time = phase_event_times[prev_phase_event_idx]

        next_phase_event_idx = closest_phase_event_idx + 1
        next_phase_event_idx = np.min((next_phase_event_idx, len(phase_event_times)-1))
        next_phase_event_time = phase_event_times[next_phase_event_idx]

        #find indices in total vicon array of times to these two events
        prev_phase_event_idx = np.argmin(np.abs(prev_phase_event_time - time_vicon)) 
        next_phase_event_idx = np.argmin(np.abs(next_phase_event_time - time_vicon)) 

        #handle transitions to and from the task
        if current_event['Name'] == 'OnRamp':
            is_steady_state = 0
            is_steady_state_vicon[prev_phase_event_idx:current_event_idx] = is_steady_state

        elif current_event['Name'] == 'OffRamp':
            is_steady_state = 0
            is_steady_state_vicon[prev_phase_event_idx:current_event_idx] = is_steady_state

    #generate stair events held in stairs_vicon
    print(len(stair_events))
    stairs_vicon = np.zeros(time_vicon.size)
    for i in range(len(stair_events)):
        current_event = stair_events.iloc[i]
        current_event_time = current_event['Time (s)']
        # print(current_event)
        current_event_idx = np.argmin(np.abs(current_event_time - time_vicon))
        stair_height = 0
        is_steady_state = 1

        if i < len(stair_events)-1:
            next_event = stair_events.iloc[i+1]
        
            next_event_time = next_event['Time (s)']
            next_event_idx = np.argmin(np.abs(next_event_time - time_vicon))

            if current_event['Name'] == 'OnStairs' and next_event['Name'] == 'OffStairs':
                stair_height = 1


            #if we're going away from the origin, positive speed, decline
            mean_speed = np.mean(speed_vicon[current_event_idx:next_event_idx])
            # print(mean_speed)
            if mean_speed > 0:
                print('GOING DOWNSTAIRS')
                stair_height = -stair_height

        stairs_vicon[current_event_idx:next_event_idx] = stair_height

        #handle steady state transitions
        #find the idx of the closest phase event to the ramp event
        closest_phase_event_idx = np.argmin(np.abs(current_event_time - phase_event_times)) 

        #find the indices (in the list of times) of the closest phase events where the ramp changes
        prev_phase_event_idx = closest_phase_event_idx - 1
        prev_phase_event_idx = np.max((prev_phase_event_idx, 0))
        prev_phase_event_time = phase_event_times[prev_phase_event_idx]

        next_phase_event_idx = closest_phase_event_idx + 1
        next_phase_event_idx = np.min((next_phase_event_idx, len(phase_event_times)-1))
        next_phase_event_time = phase_event_times[next_phase_event_idx]

        #find indices in total vicon array of times to these two events
        prev_phase_event_idx = np.argmin(np.abs(prev_phase_event_time - time_vicon)) 
        next_phase_event_idx = np.argmin(np.abs(next_phase_event_time - time_vicon)) 
        
        #handle transitions to and from the task
        if current_event['Name'] == 'OnStairs':
            is_steady_state = 0
            is_steady_state_vicon[prev_phase_event_idx:current_event_idx] = is_steady_state

        elif current_event['Name'] == 'OffStairs':
            is_steady_state = 0
            is_steady_state_vicon[prev_phase_event_idx:current_event_idx] = is_steady_state


    
    #take absolute value of speed
    speed_vicon = np.abs(speed_vicon)

    #Plot Calculated Quantities
    if DO_PLOTS:
        fig3, axs3 = plt.subplots(5,1,sharex=True,figsize=(10,6))


        axs3[0].plot(time_vicon, phase_vicon,'r', label='Phase')
        axs3[0].plot(time_vicon, phase_events_vicon,'k', label='Phase Events')
        
        axs3[0].plot(time_vicon, RHEE_Z,label='RHEE_Z')
        axs3[0].plot(time_vicon, foot_angle_vicon_calcd/100,label='foot_angle_vicon_calcd')
        
        
        axs3[1].plot(time_vicon, speed_vicon,'r', label='Speed')
        axs3[2].plot(time_vicon, incline_vicon,'r', label='Incline')
        axs3[3].plot(time_vicon, stairs_vicon,'r', label='Stair')
        axs3[4].plot(time_vicon, is_steady_state_vicon,'k', label='is steady state')

        axs3[0].legend()
        axs3[1].legend()
        axs3[2].legend()
        axs3[3].legend()
        axs3[4].legend()
        

        axs3[-1].set_xlabel("time (sec)")

    # input()
    df = pd.DataFrame({'time': time_vicon.reshape(-1),
                    'phase_vicon': phase_vicon.reshape(-1),
                    'speed_vicon':speed_vicon.reshape(-1),
                    'incline_vicon':incline_vicon.reshape(-1),
                    'stairs_vicon':stairs_vicon.reshape(-1),
                    'foot_angle_vicon_calcd':foot_angle_vicon_calcd.reshape(-1),
                    'is_steady_state_vicon':is_steady_state_vicon.reshape(-1),
                    'phase_events_vicon':phase_events_vicon.reshape(-1)
                    })
    if DO_EXPORT:
        df.to_csv(f'Vicon_{AB_SUBJECT}_{filename[:-4]}_processed.csv',index=False)

    if DO_PLOTS:
        plt.show()
    return df


#
def combineExobootViconData(exoboot_filename, vicon_filename, time_vicon_offset, export_filename=None, DO_PLOTS=False):
    """This function computes the ground truth/vicon quantities interpolated to the 
        exoboot data

    Args:
        exoboot_filename (string): the filename from the exoboot
        vicon_filename (string): the filename containing vicon data
        time_vicon_offset (float): the time offset between the exoboot and vicon data files
        export_filename (string, optional): the filename to export the combined exoboot/vicon data Defaults to None.
        DO_PLOTS (bool, optional): whether to plot the combined data. Defaults to False.

    """    

    #READ IN EXO DATA
    data = np.loadtxt(exoboot_filename, delimiter=',')

    timeSec_vec=data[:,0]
    accelVec_corrected=data[:,1:4]

    gyroVec_corrected=data[:,4:7]

    ankleAngle = data[:,25]
    roll = data[:,28]
    pitch = data[:,29]
    yaw = data[:,30]

    phase_hardware = data[:,40]
    speed_hardware = data[:,41]
    incline_hardware = data[:,42]
    stairs_hardware = data[:,43]

    actTorque = data[:,28]
    desTorque = data[:,29]

    foot_angles = data[:,22]
    foot_vel_angles = data[:,23]
    shank_angles = data[:,24]
    shank_vel_angles = data[:,25]
    heelAccForward_meas = data[:,33] #92
    heelAccSide_meas = data[:,34]#71
    heelAccUp_meas = data[:,35]#71
    heelAccForward_meas_filt = data[:,47]
    heelAccUp_meas_filt = data[:,48]

    heel_acc_forward = heelAccForward_meas_filt
    heel_acc_up = heelAccUp_meas_filt

    heelAccForward_meas_fromDeltaVelocity = data[:,36] #92
    heelAccSide_meas_fromDeltaVelocity = data[:,37]#71
    heelAccUp_meas_fromDeltaVelocity = data[:,38]#71

    phase_nn = data[:,49]
    speed_nn = data[:,50]
    incline_nn = data[:,51]
    is_stairs_nn = data[:,52]
    is_moving_nn = data[:,53]

    footAngle_predicted_ekf = data[:,54]
    footAngleVel_predicted_ekf = data[:,55]
    shankAngle_predicted_ekf = data[:,56]
    shankAngleVel_predicted_ekf = data[:,57]
    heelAccForward_predicted_ekf = data[:,58]
    heelAccUp_predicted_ekf = data[:,59]

    count = data[:,60]
    m_distance_nn = data[:,61]
    m_distance_ekf = data[:,62]

    dts = data[:,39]
    freqData = 1/dts
    
    # LOAD IN VICON
    df_vicon = pd.read_csv(vicon_filename)

    time_vicon = df_vicon['time'].to_numpy()
    phase_vicon = df_vicon['phase_vicon'].to_numpy()
    speed_vicon = df_vicon['speed_vicon'].to_numpy()
    incline_vicon = df_vicon['incline_vicon'].to_numpy()
    stairs_vicon = df_vicon['stairs_vicon'].to_numpy()
    foot_angle_vicon = df_vicon['foot_angle_vicon_calcd'].to_numpy()
    is_steady_state_vicon = df_vicon['is_steady_state_vicon'].to_numpy()
    phase_events_vicon = df_vicon['phase_events_vicon'].to_numpy()


    # From vicon, extract ground truth at the exoboot time vector
    phase_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, phase_vicon)
    speed_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, speed_vicon)
    incline_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, incline_vicon)
    stairs_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, stairs_vicon)
    is_steady_state_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, is_steady_state_vicon)
    phase_events_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, phase_events_vicon)
    phase_events_ground_truth = np.round(phase_events_ground_truth)

    if DO_PLOTS:
        fig, axs = plt.subplots(4,1)
        axs[0].plot(timeSec_vec, phase_ground_truth,label='Ground Truth')
        axs[0].plot(timeSec_vec, phase_hardware,label='Hardware')

        axs[0].plot(timeSec_vec, is_steady_state_ground_truth,'k',label='Is steady state')
        axs[0].plot(timeSec_vec, phase_events_ground_truth,'r',label='phase_events_ground_truth')


        axs[0].legend()
        axs[0].set_ylabel('Phase')

        axs[1].plot(timeSec_vec, speed_ground_truth,label='speed_ground_truth')
        axs[1].plot(timeSec_vec, speed_hardware,label='speed_hardware')
        axs[1].set_ylabel('Speed (m/s)')

        axs[2].plot(timeSec_vec, incline_ground_truth,label='incline_ground_truth')
        axs[2].plot(timeSec_vec, incline_hardware,label='incline_hardware')
        axs[2].set_ylabel('Incline (deg)')

        axs[3].plot(timeSec_vec, stairs_ground_truth,label='stairs_ground_truth')
        axs[3].plot(timeSec_vec, stairs_hardware,label='stairs_hardware')
        axs[3].set_ylabel('Is Stairs')

        axs[-1].set_xlabel('Time (s)')



        fig1, axs1 = plt.subplots(4,1)
        axs1[0].plot(timeSec_vec, foot_angles,label='foot_angles')
        axs1[0].plot(time_vicon + time_vicon_offset, foot_angle_vicon,label='foot_angle_vicon')

        
        axs1[0].plot(timeSec_vec, phase_ground_truth*30,'k',label='phase_ground_truth')
        axs1[0].legend()

        axs1[1].plot(timeSec_vec, foot_vel_angles)
        axs1[2].plot(timeSec_vec, shank_angles)
        axs1[3].plot(timeSec_vec, shank_vel_angles)

        fig2, axs2 = plt.subplots()
        axs2.hist(dts)
    
    if DO_PLOTS:
        plt.show()
        

    data = np.hstack((
                foot_angles.reshape(-1,1), 
                foot_vel_angles.reshape(-1,1), 
                shank_angles.reshape(-1,1), 
                shank_vel_angles.reshape(-1,1),
                heel_acc_forward.reshape(-1,1), 
                heel_acc_up.reshape(-1,1), 
                dts.reshape(-1,1),
                phase_ground_truth.reshape(-1,1), 
                speed_ground_truth.reshape(-1,1), 
                incline_ground_truth.reshape(-1,1), 
                stairs_ground_truth.reshape(-1,1),
                actTorque.reshape(-1,1),
                desTorque.reshape(-1,1),
                phase_hardware.reshape(-1,1),
                speed_hardware.reshape(-1,1),
                incline_hardware.reshape(-1,1),
                stairs_hardware.reshape(-1,1),
                count.reshape(-1,1),
                is_steady_state_ground_truth.reshape(-1,1),
                phase_events_ground_truth.reshape(-1,1),
                ))

    print(data.shape)
    columns = ['foot_angles', 'foot_vel_angles', 'shank_angles', 'shank_vel_angles', \
        'heel_acc_forward', 'heel_acc_up', 'dt',\
        'phase_ground_truth', 'speed_ground_truth', 'incline_ground_truth', 'stairs_ground_truth',
        'actTorque','desTorque',
        'phase_hardware',
        'speed_hardware',
        'incline_hardware',
        'stairs_hardware',
        'synch_count',
        'is_steady_state_ground_truth',
        'phase_events_ground_truth'
        ]

    df = pd.DataFrame(data, columns=columns)
    print(df.head())

    df.to_csv(export_filename, index=None)

    return timeSec_vec, phase_ground_truth, speed_ground_truth, incline_ground_truth, stairs_ground_truth,\
            foot_angles, foot_vel_angles, shank_angles, shank_vel_angles,\
            heel_acc_forward, heel_acc_up,\
            dts

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        #>>> # linear interpolation of NaNs
        #>>> nans, x= nan_helper(y)
        #>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def process_file_for_finetuning(filename,export_filename, DO_PLOTS=True):
    """This function takes in a combined exoboot/vicon file that contains stairs data
    and processes it for finetuning. In practice, this selects the data on the stairs, + the two seconds 
    before and after the stair ascent/descent

    Args:
        filename (string): the combined exoboot/vicon file
        export_filename (string): the processed combined exoboot/vicon file for finetuning
        DO_PLOTS (bool, optional): whether to plot finetuned data. Defaults to True.
    """    
    df = pd.read_csv(filename)

    dt = df['dt'].to_numpy()
    phase_ground_truth = df['phase_ground_truth'].to_numpy()
    speed_ground_truth = df['speed_ground_truth'].to_numpy()
    incline_ground_truth = df['incline_ground_truth'].to_numpy()
    stairs_ground_truth = df['stairs_ground_truth'].to_numpy()

    foot_angles = df['foot_angles'].to_numpy()
    foot_vel_angles = df['foot_vel_angles'].to_numpy()
    shank_angles = df['shank_angles'].to_numpy()
    shank_vel_angles = df['shank_vel_angles'].to_numpy()
    heel_acc_forward = df['heel_acc_forward'].to_numpy()
    heel_acc_up = df['heel_acc_up'].to_numpy()

    idxs_total = np.arange(0,len(phase_ground_truth))

    #strip out only where the stairs are there
    is_stairs_bool = np.abs(stairs_ground_truth) > 0.001
    is_stairs_idxs = idxs_total[np.where(is_stairs_bool)]
    print(is_stairs_idxs)

    idxs_keep = []

    #loop through all the indexes and build a new vector of indexes to keep
    for i in range(len(idxs_total.tolist())):

        #keep the indexes where stairs occured
        if np.abs(stairs_ground_truth[i]) > 0.001:
            idxs_keep.append(i)
            
        #keep the indexes that are 200 iterations or less away to a stair segment
        closestStairIdx = is_stairs_idxs[np.argmin(np.abs(is_stairs_idxs - i))]
        if np.abs(closestStairIdx - i) <= 200:
            idxs_keep.append(i)
        

    idxs_keep.sort()
    idxs_keep = np.array(idxs_keep)
    print(idxs_keep)

    #extract data during the selected indexes
    dt = dt[idxs_keep]
    phase_ground_truth = phase_ground_truth[idxs_keep]
    speed_ground_truth = speed_ground_truth[idxs_keep]
    incline_ground_truth = incline_ground_truth[idxs_keep]
    stairs_ground_truth = stairs_ground_truth[idxs_keep]
    foot_angles = foot_angles[idxs_keep]
    foot_vel_angles = foot_vel_angles[idxs_keep]
    shank_angles = shank_angles[idxs_keep]
    shank_vel_angles = shank_vel_angles[idxs_keep]
    heel_acc_forward = heel_acc_forward[idxs_keep]
    heel_acc_up = heel_acc_up[idxs_keep]

    if DO_PLOTS:
        fig, axs = plt.subplots(4,1,sharex=True)
        axs[0].plot(phase_ground_truth,label='phase_ground_truth')
        axs[0].legend()

        axs[1].plot(speed_ground_truth,label='speed_ground_truth')
        axs[2].plot(incline_ground_truth,label='incline_ground_truth')
        axs[3].plot(stairs_ground_truth,label='stairs_ground_truth')



        fig1, axs1 = plt.subplots(4,1)
        axs1[0].plot(foot_angles,label='foot_angles')

        axs1[0].plot(phase_ground_truth*30,'r',label='phase_ground_truth')
        axs1[0].legend()

        axs1[1].plot(foot_vel_angles)
        axs1[2].plot(shank_angles)
        axs1[3].plot(shank_vel_angles)

        fig2, axs2 = plt.subplots()
        axs2.hist(dt)

    data = np.hstack((
                foot_angles.reshape(-1,1), 
                foot_vel_angles.reshape(-1,1), 
                shank_angles.reshape(-1,1), 
                shank_vel_angles.reshape(-1,1),
                heel_acc_forward.reshape(-1,1), 
                heel_acc_up.reshape(-1,1), 
                dt.reshape(-1,1),
                phase_ground_truth.reshape(-1,1), 
                speed_ground_truth.reshape(-1,1), 
                incline_ground_truth.reshape(-1,1), 
                stairs_ground_truth.reshape(-1,1),
                ))

    print(data.shape)
    columns = ['foot_angles', 'foot_vel_angles', 'shank_angles', 'shank_vel_angles', \
        'heel_acc_forward', 'heel_acc_up', 'dt',\
        'phase_ground_truth', 'speed_ground_truth', 'incline_ground_truth', 'stairs_ground_truth']

    df = pd.DataFrame(data, columns=columns)
    print(df.head())

    df.to_csv(export_filename, index=None)
    
    if DO_PLOTS:
        plt.show()

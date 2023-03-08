""" Simulates the gait transformer estimator using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
import time
import gc
import os, sys

import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
# import torch

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)
sys.path.append(thisdir + '/utils')

from filter_classes import FirstOrderLowPassLinearFilter, FirstOrderHighPassLinearFilter, SecondOrderHighPassFilter
from training_utils import phase_dist
sideMultiplier = -1


gc.disable()



def main():
    # filename = 'live_data/20230208-22_AE0934_gait_transformer.csv'
    # filename = "pilot_data/AB12-Brake/20230206-21_AB12-Brake-stairspilot1.csv"
    filename = "pilot_data/AB12-Brake/20230206-21_AB12-Brake-walkingcircuitpilot_seg1.csv"

    # filename = "pilot_data/AB05-Eagle/20230206-19_AB05-Eagle-pilotwalkingcircuit2_seg1.csv"
    # filename = "pilot_data/AB05-Eagle/pilotwalkingcircuitseg2/20230206-19_AB05-Eagle-pilotwalkingcircuit2_seg2.csv"

    # filename = "pilot_data/AB05-Eagle/20230206-19_AB05_Eagle_stairs1.csv"

    

    # data = np.loadtxt(filename, delimiter=',') 
    data = np.loadtxt(filename, delimiter=',',skiprows=1) 

    N_data = data.shape[0]
    print(N_data)

   
    speed_scale = (0,2)
    incline_scale = (-10,10)
    stair_height_scale = (-0.2,0.2)
    
    #set up filters for the heel acc's
    heel_forward_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    heel_up_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    
    #set up filters for the heel pos's
    heel_forward_vel_filter = FirstOrderHighPassLinearFilter(fc=0.001, dt=0.01)
    heel_up_vel_filter = FirstOrderHighPassLinearFilter(fc=0.001, dt=0.01)
    
    
    heel_forward_pos_filter = FirstOrderHighPassLinearFilter(fc=0.0001, dt=0.01)
    heel_up_pos_filter = FirstOrderHighPassLinearFilter(fc=0.0001, dt=0.01)
    
    #SET UP second order Filters
    omega = 0.3 * np.pi*2
    xi = 0.9

#     A = np.array([
#         [0,             1        ]      ,
#         [-ω**2,         -2*ω*ζ   ]])

#     # C = np.array([[-ω**2,         -2*ω*ζ   ]])
#     C = np.array([[1,         0  ]])
#     B = np.array([[0, 1]]).T
#     # D = np.array([[1.0]])
#     D = np.array([[0]])
    HPF_X0 = 0.0*np.ones((2,1))

    # heel_forward_pos_filter = SecondOrderHighPassFilter(omega, xi, 0.01, HPF_X0)
    # heel_up_pos_filter = SecondOrderHighPassFilter(omega, xi, 0.01, HPF_X0)

    #set up plotting options
    SHOW_FULL_STATE = True
    PLOT_MEASURED = True
    PLOT_MEASURED_NORM = not True

    plot_data = []
    plot_data_measured = []
    plot_data_measured_norm = []
    plot_data_measured_prediction = []
    plot_states = []
    plot_mahalanobis_dist = np.zeros((N_data,))
    
    prev=0

    tic = time.time()

    # EXTRACT ACT VARIABLES
    timeSec_vec=data[:,0]
    
    dt_vec = 0.01 * np.ones(timeSec_vec.shape)
    dt_vec[1:] = np.diff(timeSec_vec,axis=0)
    
    accelVec_corrected_vec=data[:,1:4]
    gyroVec_corrected_vec=data[:,4:7]
    ankleAngle_vec = data[:,25]
    roll_vec = data[:,28]
    pitch_vec = data[:,29]
    yaw_vec = data[:,30]
    phase_vec = data[:,40]
    speed_vec = data[:,41]
    incline_vec = data[:,42]
    is_stairs_vec = data[:,43]
    is_moving_vec = data[:,44]
    actTorque_vec = data[:,28]
    desTorque_vec = data[:,29]
    footAngle_meas_vec = data[:,22]
    footAngleVel_meas_vec = data[:,23]
    shankAngle_meas_vec = data[:,24]
    shankAngleVel_meas_vec = data[:,25]
    heelAccForward_meas_vec = data[:,33]
    heelAccUp_meas_vec = data[:,35]
    heelAccForward_meas_fromDeltaVelocity_vec = data[:,36]
    heelAccUp_meas_fromDeltaVelocity_vec = data[:,38]
    
    # #EXOBOOT
    # footAngle_meas_vec = data[:,0]
    # footAngleVel_meas_vec = data[:,1]
    # shankAngle_meas_vec = data[:,2]
    # shankAngleVel_meas_vec = data[:,3]
    # heelAccForward_meas_fromDeltaVelocity_vec = data[:,4] #92
    # heelAccUp_meas_fromDeltaVelocity_vec = data[:,5]#71
    # dt_vec = data[:,6]
    # timeSec_vec = np.cumsum(dt_vec,axis=0)
    # phase_vec = data[:,7]
    # speed_vec = data[:,8]
    # incline_vec = data[:,9]
    # is_stairs_vec = data[:,10]
    # is_moving_vec = data[:,11]

    is_moving_buffer = np.zeros((100,))
    
    heelVelForward_numInt = 0
    heelVelUp_numInt = 0
    
    heelPosForward_numInt = 0
    heelPosUp_numInt = 0
    for i,x in enumerate(data[:]):

        timeSec=timeSec_vec[i]
        dt = dt_vec[i]

        prev=timeSec
        phase = phase_vec[i]
        speed = speed_vec[i]
        footAngle_meas = footAngle_meas_vec[i]
        shankAngle_meas = shankAngle_meas_vec[i]
        footAngleVel_meas = footAngleVel_meas_vec[i]
        shankAngleVel_meas = shankAngleVel_meas_vec[i]

        heelAccForward_meas = heelAccForward_meas_vec[i]
        heelAccUp_meas = heelAccUp_meas_vec[i]
                                                       
        #filter
        heelAccForward_meas = heel_forward_acc_filter.step(i, heelAccForward_meas)
        heelAccUp_meas = heel_up_acc_filter.step(i, heelAccUp_meas)
        
        #ZUPT
        DO_ZUPT = np.abs(footAngleVel_meas) <= 15 and (phase <= 0.3)
        if DO_ZUPT:
            heelVelForward_numInt = 0
            heelVelUp_numInt = 0
        else:
            heelVelForward_numInt += heelAccForward_meas * dt
            heelVelUp_numInt += heelAccUp_meas * dt
        
        
        
        is_moving = np.abs(speed) > 0.05
        is_moving_buffer[:-1] = is_moving_buffer[1:]
        is_moving_buffer[-1] = int(is_moving)

        # print(speed)
        # print(is_moving_buffer)
        # input()
        IS_MOVING = np.sum(is_moving_buffer) >= 10
        # print(IS_MOVING)
        if not IS_MOVING:
            heelPosForward_numInt = 0
            heelPosUp_numInt = 0
            # print('RESETTING')
        else:
            heelPosForward_numInt += heelVelForward_numInt * dt
            heelPosUp_numInt += heelVelUp_numInt * dt
        # input()
        heelVelForward_numInt = heel_forward_vel_filter.step(i, heelVelForward_numInt)
        heelVelUp_numInt = heel_up_vel_filter.step(i, heelVelUp_numInt)

        heelPosForward = heel_forward_pos_filter.step(i, heelPosForward_numInt)
        heelPosUp = heel_up_pos_filter.step(i, heelPosUp_numInt)

#         states_heelPosForward, heelPosForward = heel_forward_pos_filter.step(i, dt, heelAccForward_meas)
#         heelPosForward = heelPosForward[0,0]
        
#         states_heelPosUp, heelPosUp = heel_up_pos_filter.step(i, dt, heelAccUp_meas)
#         heelPosUp = heelPosUp[0,0]
                             
        
        plot_data.append([
            timeSec, #0
            heelPosForward_numInt, #1
            heelPosUp_numInt, #2
            heelPosForward,
            heelPosUp,
            heelVelForward_numInt,
            heelVelUp_numInt
            ])
        
        plot_data_measured.append([
            timeSec,
            heelAccForward_meas,
            heelAccUp_meas
            
        ])


    toc = time.time()
    print(f"Ran simulation loop in {toc - tic:0.4f} seconds")

    plot_data = np.array(plot_data)
    plot_data_measured = np.array(plot_data_measured)


    print('Sampling Rate')
    sample_rate = 1/np.mean(np.diff(data[:,0]))
    print(sample_rate)

    
    fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
    # axs[0].plot(plot_data[:,0], plot_data[:,1], label=r"$pos forward num int$")
    axs[0].plot(plot_data[:,0], plot_data[:,3], label="pos forward")
    axs[0].plot(timeSec_vec, phase_vec,'k', label="phase")
    # axs[0].plot(timeSec_vec, incline_vec,'r', label="incline")
    axs[0].plot(timeSec_vec, speed_vec,'r', label="speed")

    speed

    axs[0].legend()

    # axs[1].plot(plot_data[:,0], plot_data[:,2], label=r"$pos up num int$")
    axs[1].plot(plot_data[:,0], plot_data[:,4], label="pos up")
    axs[1].plot(timeSec_vec, is_stairs_vec,'r', label="is_stairs")
    
    axs[1].legend()

    axs[-1].set_xlabel("time (sec)")
    # axs[-1].set_xlim([10,30])
    # axs[-1].set_xlim([40,50])

    print("this is done (show state)")
    
    if True:
        fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))

        axs[0].plot(plot_data[:,0], plot_data[:,5], label="vel forward hardware sim")
        axs[0].plot(timeSec_vec, phase_vec,'k', label="phase")

        axs[0].legend()

        axs[1].plot(plot_data[:,0], plot_data[:,6], label="vel up hardware sim")
        axs[1].plot(timeSec_vec, phase_vec,'k', label="phase")

        axs[1].legend()
        # axs[-1].set_xlim([10,30])
        axs[-1].set_xlabel("time (sec)")
    
    
    fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
    
    axs[0].plot(timeSec_vec, heelAccForward_meas_vec, label="acc forward hardware")
    axs[0].plot(plot_data_measured[:,0], plot_data_measured[:,1], label="acc forward hardware sim")

    axs[0].legend()

    axs[1].plot(timeSec_vec, heelAccUp_meas_vec, label="acc up hardware")
    axs[1].plot(plot_data_measured[:,0], plot_data_measured[:,2], label="acc up hardware sim")
  
    axs[1].legend()
    # axs[-1].set_xlim([10,30])
    axs[-1].set_xlabel("time (sec)")
    
    
    if PLOT_MEASURED:
        fig, axs = plt.subplots(4,1,sharex=True,figsize=(10,6))
        axs[0].plot(timeSec_vec, footAngle_meas_vec, label=r"$foot angle, meas_{hardware}$")

        axs[0].legend()
        axs[0].set_ylim([-70,50])

        axs[1].plot(timeSec_vec, footAngleVel_meas_vec, label=r"$foot angle vel, meas_{hardware}$")
        axs[1].set_ylim([-50,50])
        axs[1].legend()
        # axs[1].set_ylim([-500,500])

        axs[2].plot(timeSec_vec, shankAngle_meas_vec, label=r"$shank angle, meas_{hardware}$")

        axs[2].legend()
        axs[2].set_ylim([-70,50])

        axs[3].plot(timeSec_vec, shankAngleVel_meas_vec, label=r"$shank angle vel, meas_{hardware}$")
        axs[3].legend()
        # axs[-1].set_xlim([10,30])
        # axs[-1].set_xlim([20,23])
        
    
    

    plt.show()
    

if __name__ == '__main__':
    main()
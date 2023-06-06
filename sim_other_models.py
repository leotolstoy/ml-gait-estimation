""" Simulates the phase estimator EKF and TBE using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
import time
import gc
import sys

import matplotlib.pyplot as plt
import pandas as pd
from training_utils import phase_dist


sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/old_ekf_funcs')


from old_phase_ekf import PhaseEKF
from arctanMapFuncs import *
from heelphase_ekf import HeelPhaseEKF
from gait_model import GaitModel_Fourier
from ekf_torque_profile import TorqueProfile
from measurement_noise_model import MeasurementNoiseModel
from timing_based_estimator import TimingPhaseEstimator
from filter_classes import FirstOrderLowPassLinearFilter, FirstOrderHighPassLinearFilter, GenericLinearFilter
from attitudeEstimatorEKFFuncs import extractEulerAngles_new
from heel_strike_detector import HeelStrikeDetector

DO_KIDNAPPING =False
DO_OVERRIDES = True
UPDATE_OLS=True
DO_TBE = True
DO_NEW_HS_DETECTOR = True
sideMultiplier = -1

PLOT_EXO_IMU = False
PLOT_TIMING_INFORMATION = False
PLOT_MEASURED = True
PLOT_AHRS = False
SHOW_FULL_STATE = True
PLOT_SIM_FOOT_POS = True
USE_C = not True
SPOOF_SENSORS = False

DO_GUARDRAILS = not True


gc.disable()


def get_std(cov):
    """Get square root of diagonals (standard deviations) from covariances 
    
    Args:
        cov (np matrix): a stack (in the third dimension) of positive definite covariances
    
    Returns:
        stdevs (np matrix): an array of standard deviations
    """
    
    d, d, N = cov.shape
    std_devs = np.zeros((N, d), dtype=float)
    for ii in range(N):
        std_devs[ii, :] = np.sqrt(np.diag(cov[:, :, ii]))
    return std_devs


def sim_other_models(data, SUBJECT_LEG_LENGTH, torque_profile_path, gait_model_covar_path, gait_model_path, DO_PLOTS=True):
    """This function runs the TBE and EKF simulations over a set of exoboot data

    Args:
        data (np array): a np matrix containing the exoboot data
        SUBJECT_LEG_LENGTH (float): the participant leg length
        torque_profile_path (str): the path to the torque profile coefficients
        gait_model_covar_path (str): the path to the heteroscedastic model file
        gait_model_path (str): the path to the gait model coefficients
        DO_PLOTS (bool, optional): whether to plot results. Defaults to True.

    Returns:
        np arrays: the RMSEs for the gait states for the TBE and EKF simulations
    """        
    N_data = data.shape[0]
    # print(N_data)


    attitude_ekf_args = {'sigma_gyro':0.0023,
                        'sigma_accel': 0.0032,
                        'sigma_q_AE':1e2,
                        'Q_pos_scale':1e-10}

    #HETEROSCEDASTIC VELOCITY ON
    #stable
    # sigma_foot = 1
    # sigma_shank = 7

    # sigma_foot_vel = 10
    # sigma_shank_vel = 20

    # sigma_heel_pos_forward = 0.01 #m
    # sigma_heel_pos_up = 0.08 #m

    sigma_foot = 1
    sigma_shank = 7

    sigma_foot_vel = 10
    sigma_shank_vel = 20

    sigma_heel_pos_forward = 0.01 #m
    sigma_heel_pos_up = 0.08 #m

    meas_config = 'full'

    # #FULL
    R_meas = np.diag([sigma_foot**2,
        sigma_foot_vel**2,\
        sigma_shank**2,
        sigma_shank_vel**2,\
        sigma_heel_pos_forward**2, 
        sigma_heel_pos_up**2,
        ])

    #STABLE
    # sigma_q_phase=0
    # sigma_q_phase_dot=6e-4
    # sigma_q_sL=9e-4
    # sigma_q_incline=6e-3

    sigma_q_phase=0
    sigma_q_phase_dot=1e-3
    sigma_q_sL=2e-3
    sigma_q_incline=5e-2
    
    torque_profile = TorqueProfile(torque_profile_path)
    gait_model = GaitModel_Fourier(gait_model_path,phase_order=20, stride_length_order=1, incline_order=1)


    measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, meas_config=meas_config,DO_XSUB_R=True)
    phase_ekf_args = {'gait_model':gait_model,
            'torque_profile':torque_profile,
            'measurement_noise_model':measurement_noise_model,
            'CANCEL_RAMP':False,
            'BOOST_BANDWIDTH':False,
            'sigma_q_phase':sigma_q_phase,
            'sigma_q_phase_dot':sigma_q_phase_dot,
            'sigma_q_sL':sigma_q_sL,
            'sigma_q_incline':sigma_q_incline,
            'DO_GUARDRAILS':DO_GUARDRAILS
            }

    phase_ekf = PhaseEKF(**phase_ekf_args)

    if DO_TBE:
        timing_based_estimator = TimingPhaseEstimator()
    else:
        timing_based_estimator = None


    #INITIALIZE BACKUP EKF
    #velocity het off
    # Q_HP = np.diag([5e-5,1e-3])
    # R_HP = phase_ekf.R_mean

    #Velocity het on
    Q_HP = np.diag([1e-4,5e-3])
    R_HP = phase_ekf.R_mean
    
    heelphase_ekf=HeelPhaseEKF(phase_ekf, Q_HP, R_HP, timing_based_estimator=timing_based_estimator)

    #set up the filters for the gyro and accel signals
    fc_gyro = 5
    gyroYLowPassFilter = FirstOrderLowPassLinearFilter(fc=fc_gyro,dt=1/100)

    fc_accel = 20
    accelZHighPassFilter = FirstOrderHighPassLinearFilter(fc=fc_accel,dt=1/100)

    
    #SET UP Filters
    ω = 0.5 * np.pi*2
    ζ= 0.9

    A = np.array([
        [0,             1        ]      ,
        [-ω**2,         -2*ω*ζ   ]])

    # C = np.array([[-ω**2,         -2*ω*ζ   ]])
    C = np.array([[1,         0  ]])
    B = np.array([[0, 1]]).T
    # D = np.array([[1.0]])
    D = np.array([[0]])
    HPF_X0 = 0.0*np.ones((A.shape[0],1))

    heelPosForwardFilter = GenericLinearFilter(A, B, C, D, HPF_X0)


    ω = 0.5 * np.pi*2
    ζ= 0.9

    A = np.array([
        [0,             1        ]      ,
        [-ω**2,         -2*ω*ζ   ]])

    # C = np.array([[-ω**2,         -2*ω*ζ   ]])
    C = np.array([[1,         0  ]])
    B = np.array([[0, 1]]).T
    # D = np.array([[1.0]])
    D = np.array([[0]])

    HPF_X0 = 0.0*np.ones((A.shape[0],1))
    heelPosUpFilter = GenericLinearFilter(A, B, C, D, HPF_X0)
    MAX_TIME_STEP_INTEGRATE = 0.06

    #set up HSDetector
    HS_analysis_window = 10
    HSDetector = HeelStrikeDetector(HS_analysis_window)

    plot_data = []
    plot_data_measured = []

    plot_data_timing_phase_ekf = []
    plot_data_TBE = []
    plot_data_HPEKF = []

    plot_states = []

    state_std_devs = []

    prev=0


    #INITIALIZE STORAGE FOR COVARS
    P_covars = np.zeros((4, 4, N_data))

    tic = time.time()

    timeSec_vec_hardware = data[:,0]
    heelAccForward_meas_fromDeltaVelocity_vec_hardware = data[:,33]
    heelAccSide_meas_fromDeltaVelocity_vec_hardware = data[:,34]
    heelAccUp_meas_fromDeltaVelocity_vec_hardware = data[:,35]

    heelAccNorm_meas_fromDeltaVelocity_vec_hardware = np.sqrt(heelAccForward_meas_fromDeltaVelocity_vec_hardware**2 +
                                                            heelAccSide_meas_fromDeltaVelocity_vec_hardware**2 +
                                                            (heelAccUp_meas_fromDeltaVelocity_vec_hardware)**2)

    for i,x in enumerate(data[:]):

        timeSec=x[0]
        dt = timeSec-prev

        prev=timeSec
        accelVec_corrected=x[1:4]
        gyroVec_corrected=x[4:7]
        shankAngle_meas = x[24]
        footAngle_meas = x[22]

        HSDetected_hardware = x[24]
        accelZ = accelVec_corrected[2]
        gyroY = gyroVec_corrected[1]
        shankAngleVel_meas = x[25]
        footAngleVel_meas = x[23]


        heelAccForward_meas_fromDeltaVelocity = x[33] #92
        heelAccSide_meas_fromDeltaVelocity = x[34] #70
        heelAccUp_meas_fromDeltaVelocity = x[35]#71

        heelAccForward_meas_norm = np.sqrt(heelAccForward_meas_fromDeltaVelocity**2 +
                                                            heelAccSide_meas_fromDeltaVelocity**2 +
                                                            (heelAccUp_meas_fromDeltaVelocity)**2)

        if DO_NEW_HS_DETECTOR:
            gyroY_filter = gyroYLowPassFilter.step(i, gyroY)
            accelZ_filter = accelZHighPassFilter.step(i, accelZ)
            HSDetected_sim = HSDetector.detectHS(timeSec, footAngle_meas, footAngleVel_meas,  heelAccForward_meas_norm)

        else:
            HSDetected_sim = HSDetected_hardware

        

        phase_ekf.step(i,dt)

        if DO_KIDNAPPING and i % 1000 == 0 and i != 0:
            print(timeSec)
            temp_estimate = np.zeros(4)
            temp_estimate[0] = np.random.uniform(0,1)
            temp_estimate[1] = np.random.uniform(-1,1)
            temp_estimate[2] = np.random.uniform(-5,5)
            temp_estimate[3] = np.random.uniform(-10,10)
            phase_ekf.set_x_state_estimate(temp_estimate)

        
        dt_int = np.min([dt, MAX_TIME_STEP_INTEGRATE])
        states_heelPosForward, heelPosForward_meas_filt = heelPosForwardFilter.step(i, dt_int, heelAccForward_meas_fromDeltaVelocity)
        heelPosForward_meas_filt = heelPosForward_meas_filt[0,0]

        # heelPosForward_meas_filt = np.max([np.min([heelPosForward_meas_filt, 0.4]), -0.25])

        states_heelPosUp, heelPosUp_meas_filt = heelPosUpFilter.step(i, dt_int, heelAccUp_meas_fromDeltaVelocity)
        heelPosUp_meas_filt = heelPosUp_meas_filt[0,0]
        # heelPosUp_meas_filt = np.max([np.min([heelPosUp_meas_filt, 0.2]), -0.16])



        z_measured_sim = np.array([footAngle_meas, footAngleVel_meas, shankAngle_meas, shankAngleVel_meas, heelPosForward_meas_filt, heelPosUp_meas_filt])
        phase_ekf.update(i, dt, z_measured_sim)

        heelphase_ekf.step(i, timeSec, dt, z_measured_sim, HSDetected_sim, DO_OVERRIDES=DO_OVERRIDES)
        
        strideLength_update_descaled_sim = arctanMap(phase_ekf.x_state_update[2].item(0))

        #scale strideLength by subject height
        strideLength_update_descaled_sim = SUBJECT_LEG_LENGTH * strideLength_update_descaled_sim

        P_covars[:, :, i] = phase_ekf.P_covar_update
        arctanFromPos = np.arctan2(heelPosUp_meas_filt, heelPosForward_meas_filt) * 180/np.pi

        plot_data.append([
            timeSec, #0
            phase_ekf.x_state_update[0].item(0), #1
            phase_ekf.x_state_update[1].item(0), #2
            strideLength_update_descaled_sim, #3
            phase_ekf.x_state_update[3].item(0), #4
            phase_ekf.SSE, #5
            phase_ekf.get_torque(), #6
            HSDetected_sim, #7
            ])

        plot_data_timing_phase_ekf.append([
            timeSec,
            phase_ekf.timing_step,
            phase_ekf.timing_measure,
            phase_ekf.timing_update,
            phase_ekf.timing_gain_schedule_R])

        plot_data_measured.append([
            timeSec, #0
            phase_ekf.z_model[0].item(0),#1
            phase_ekf.z_model[1].item(0),#2
            phase_ekf.z_model[2].item(0),#3
            phase_ekf.z_model[3].item(0),#4
            phase_ekf.z_model[4].item(0),#5
            phase_ekf.z_model[5].item(0),#6
            z_measured_sim[0], #7
            z_measured_sim[1], #8
            z_measured_sim[2], #9
            z_measured_sim[3],#10
            z_measured_sim[4],#11
            z_measured_sim[5],#12
            phase_ekf.z_model[0].item(0) - 2*np.sqrt(phase_ekf.R[0,0]),#13
            phase_ekf.z_model[0].item(0) + 2*np.sqrt(phase_ekf.R[0,0]), #14
            phase_ekf.z_model[1].item(0) - 2*np.sqrt(phase_ekf.R[1,1]),#15
            phase_ekf.z_model[1].item(0) + 2*np.sqrt(phase_ekf.R[1,1]), #16
            phase_ekf.z_model[2].item(0) - 2*np.sqrt(phase_ekf.R[2,2]),#17
            phase_ekf.z_model[2].item(0) + 2*np.sqrt(phase_ekf.R[2,2]), #18
            phase_ekf.z_model[3].item(0) - 2*np.sqrt(phase_ekf.R[3,3]),#19
            phase_ekf.z_model[3].item(0) + 2*np.sqrt(phase_ekf.R[3,3]), #20
            phase_ekf.z_model[4].item(0) - 2*np.sqrt(phase_ekf.R[4,4]), #21
            phase_ekf.z_model[4].item(0) + 2*np.sqrt(phase_ekf.R[4,4]), #22
            phase_ekf.z_model[5].item(0) - 2*np.sqrt(phase_ekf.R[5,5]), #23
            phase_ekf.z_model[5].item(0) + 2*np.sqrt(phase_ekf.R[5,5]), #24
            arctanFromPos])

        plot_data_HPEKF.append([
            timeSec, 
            heelphase_ekf.isOverriding,\
            heelphase_ekf.phase_HP,
            heelphase_ekf.phase_rate_HP,
            SUBJECT_LEG_LENGTH * arctanMap(heelphase_ekf.x_state[0,0]),
            heelphase_ekf.x_state[1,0],
            heelphase_ekf.SSE
            ])

        state_std_devs.append([
            phase_ekf.x_state_update[0].item(0) - 2*np.sqrt(phase_ekf.P_covar_update[0,0]),#0
            phase_ekf.x_state_update[0].item(0) + 2*np.sqrt(phase_ekf.P_covar_update[0,0]),#1
            phase_ekf.x_state_update[1].item(0) - 2*np.sqrt(phase_ekf.P_covar_update[1,1]),#2
            phase_ekf.x_state_update[1].item(0) + 2*np.sqrt(phase_ekf.P_covar_update[1,1]),#3
            strideLength_update_descaled_sim - 2*np.sqrt(phase_ekf.P_covar_update[2,2]),#4
            strideLength_update_descaled_sim + 2*np.sqrt(phase_ekf.P_covar_update[2,2]),#5
            phase_ekf.x_state_update[3].item(0) - 2*np.sqrt(phase_ekf.P_covar_update[3,3]),#6
            phase_ekf.x_state_update[3].item(0) + 2*np.sqrt(phase_ekf.P_covar_update[3,3]),#7
            ])


        if DO_TBE:
            phase_estimate_TBE = timing_based_estimator.phase_estimate_TBE
            plot_data_TBE.append([
                timeSec, 
                phase_estimate_TBE,
                timing_based_estimator.stepDuration,
                timing_based_estimator.timeStrideMean])

    toc = time.time()
    print(f"Ran simulation loop in {toc - tic:0.4f} seconds")

    plot_data = np.array(plot_data)

    plot_data_measured = np.array(plot_data_measured)
    plot_data_TBE = np.array(plot_data_TBE)
    plot_states = np.array(plot_states)
    plot_data_HPEKF = np.array(plot_data_HPEKF)

    state_std_devs = np.array(state_std_devs)

    # state_std_devs = get_std(P_covars)

    # print(plot_data_timing_heelphase)

    # print sampling rate

    print('Sampling Rate')
    sample_rate = 1/np.mean(np.diff(data[:,0]))
    print(sample_rate)


    
    if SHOW_FULL_STATE and DO_PLOTS:
        fig, axs = plt.subplots(5,1,sharex=True,figsize=(10,6))

        axs[0].plot(plot_data[:,0], plot_data[:,1],'b', label=r"$phase_{sim}$")
        # axs[0].fill_between(plot_data[:,0], state_std_devs[:,0],  state_std_devs[:,1], color='blue', alpha=0.3)
        axs[0].plot(plot_data[:,0], state_std_devs[:,0],'b--')
        axs[0].plot(plot_data[:,0], state_std_devs[:,1],'b--')
        axs[0].plot(plot_data_HPEKF[:,0], plot_data_HPEKF[:,2], label=r"$phase_{hpekf}$")

        if DO_TBE:
            axs[0].plot(plot_data[:,0], plot_data_TBE[:,1], label=r"$phase_{TBE}$") 

        # axs[0].plot(timeSec_vec_hardware, HSDetected_vec_hardware, label=r"$HSDetected, hardware$")
        axs[0].plot(plot_data[:,0], plot_data[:,7],'k', label=r"$HSDetected Sim$")
        # axs[0].plot(plot_data[:,0], plot_data_HPEKF[:,1],'k', label=r"$isOverriding sim$")
        axs[0].legend()
        

        axs[1].plot(plot_data[:,0], plot_data[:,2],'b', label=r"$phasedot_{sim}$")
        # axs[0].fill_between(plot_data[:,0], state_std_devs[:,2],  state_std_devs[:,3], color='blue', alpha=0.3)
        axs[1].plot(plot_data[:,0], state_std_devs[:,2],'b--')
        axs[1].plot(plot_data[:,0], state_std_devs[:,3],'b--')
        # axs[1].fill_between(plot_data[:,0], plot_data[:,6] - 1.96 * state_std_devs[:, 1],  plot_data[:,6] + 1.96*state_std_devs[:, 1], color='blue', alpha=0.3)
        axs[1].plot(plot_data_HPEKF[:,0], plot_data_HPEKF[:,3], label=r"$phase rate_{hpekf}$")
        axs[1].legend()
        axs[1].set_ylim([0,1.3])

        axs[2].plot(plot_data[:,0], plot_data[:,3],'b', label=r"$Stride Length_{sim}$")
        # axs[0].fill_between(plot_data[:,0], state_std_devs[:,4],  state_std_devs[:,5], color='blue', alpha=0.3)
        # axs[2].plot(plot_data[:,0], state_std_devs[:,4],'b--')
        # axs[2].plot(plot_data[:,0], state_std_devs[:,5],'b--')
        # axs[2].fill_between(plot_data[:,0], plot_data[:,7] - 1.96 * state_std_devs[:, 2],  plot_data[:,7] + 1.96*state_std_devs[:, 2], color='blue', alpha=0.3)
        axs[2].plot(plot_data_HPEKF[:,0], plot_data_HPEKF[:,4], label=r"$stride length_{hpekf}$")

        # axs[2].plot(plot_data[:,0], HSDetected_vec_hardware, label=r"$HSDetected, hardware$"))
        # axs[2].plot(plot_data[:,0], plot_data[:,7],'k', label=r"$isOverriding sim$")
        # axs[2].plot(plot_data[:,0], plot_data[:,19], label=r"$isOverriding old$")
        axs[2].legend()

        axs[3].plot(plot_data[:,0], plot_data[:,4],'b', label=r"$Ramp_{sim}$")
        # axs[0].fill_between(plot_data[:,0], state_std_devs[:,6],  state_std_devs[:,7], color='blue', alpha=0.3)
        axs[3].plot(plot_data[:,0], state_std_devs[:,6],'b--')
        axs[3].plot(plot_data[:,0], state_std_devs[:,7],'b--')
        # axs[3].fill_between(plot_data[:,0], plot_data[:,8] - 1.96 * state_std_devs[:, 3],  plot_data[:,8] + 1.96*state_std_devs[:, 3], color='blue', alpha=0.3)

        axs[3].plot(plot_data_HPEKF[:,0], plot_data_HPEKF[:,5], label=r"$ramp_{hpekf}$")

        # axs[3].plot(plot_data[:,0], HSDetected_vec_hardware*10, label=r"$HSDetected$")
        # axs[3].plot(plot_data[:,0], plot_data[:,7]*10,'k', label=r"$isOverriding sim$")
        # axs[3].plot(plot_data[:,0], plot_data[:,19], label=r"$isOverriding old$")
        axs[3].legend()
        axs[3].set_ylim([-14,14])

        axs[4].plot(plot_data_HPEKF[:,0], plot_data_HPEKF[:,6], label=r"$SSE_{hpekf}$")
        axs[4].plot(plot_data[:,0], plot_data[:,5], label=r"$SSE_{sim}$")
        # axs[4].plot(plot_data[:,0], HSDetected_vec_hardware*1e3, label=r"$HSDetected$")
        # axs[4].plot(plot_data[:,0], plot_data[:,7]*1e3,'k', label=r"$isOverriding sim$")
        # axs[4].plot(plot_data[:,0], plot_data[:,19], label=r"$isOverriding old$")

        axs[4].legend()

        axs[-1].set_xlabel("time (sec)")
        print("this is done (show state)")
        # plt.show()

    if PLOT_MEASURED and DO_PLOTS:

        fig, axs = plt.subplots(5,1,sharex=True,figsize=(10,6))

        # axs[0].plot(plot_data[:,0], plot_data[:,1], label=r"$phase_{hardware}$")
        axs[0].plot(plot_data_measured[:,0], plot_data_measured[:,1], label=r"$foot angle, model_{sim}$")
        axs[0].plot(plot_data_measured[:,0], plot_data_measured[:,7], label=r"$foot angle, meas_{sim}$")
        axs[0].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        # axs[0].plot(plot_data[:,0], HSDetected_vec_hardware*1e1, label=r"$HSDetected hardware$")
        axs[0].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e1,'k', label=r"$isOverriding sim$")
        # axs[0].fill_between(plot_data_measured[:,0], plot_data_measured[:,16],plot_data_measured[:,17], alpha=.5)
        
        axs[0].legend()
        axs[0].set_ylim([-70,50])

        # axs[0].plot(plot_data[:,0], plot_data[:,1], label=r"$phase_{hardware}$")
        axs[1].plot(plot_data_measured[:,0], plot_data_measured[:,2], label=r"$foot angle vel, model_{sim}$")
        axs[1].plot(plot_data_measured[:,0], plot_data_measured[:,8], label=r"$foot angle vel, meas_{sim}$")
        axs[1].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[1].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e1,'k', label=r"$isOverriding sim$")
        axs[1].legend()
        # axs[1].set_ylim([-500,500])

        # axs[1].plot(plot_data[:,0], plot_data[:,2], label=r"$phasedot_{hardware}$")
        axs[2].plot(plot_data_measured[:,0], plot_data_measured[:,3], label=r"$shank angle, model_{sim}$")
        axs[2].plot(plot_data_measured[:,0], plot_data_measured[:,9], label=r"$shank angle, meas_{sim}$")
        axs[2].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[2].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e1,'k', label=r"$isOverriding sim$")
        # axs[2].fill_between(plot_data_measured[:,0], plot_data_measured[:,18],plot_data_measured[:,19], alpha=.5)
        axs[2].legend()
        axs[2].set_ylim([-70,50])

        axs[3].plot(plot_data_measured[:,0], plot_data_measured[:,4], label=r"$shank angle vel, model_{sim}$")
        axs[3].plot(plot_data_measured[:,0], plot_data_measured[:,10], label=r"$shank angle vel, meas_{sim}$")
        axs[3].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[3].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e1,'k', label=r"$isOverriding sim$")
        axs[3].legend()
        # axs[3].set_ylim([-500,500])

        axs[4].plot(plot_data_measured[:,0], plot_data_measured[:,5], label=r"$foot position, model_{sim}$")
        axs[4].plot(plot_data_measured[:,0], plot_data_measured[:,11], label=r"$foot position, meas_{sim}$")
        axs[4].fill_between(plot_data_measured[:,0], plot_data_measured[:,21],plot_data_measured[:,22], alpha=.5)
        axs[4].plot(plot_data[:,0], plot_data[:,7]*1e-1, 'r', label=r"$HSDetected Sim$")
        # axs[4].plot(plot_data[:,0], HSDetected_vec_hardware*1e-1, label=r"$HSDetected Hardware$")
        axs[4].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")
        axs[4].legend()
        # axs[4].set_ylim([-0.5,0.5])

        print("this is done (plot measured)")

    if PLOT_SIM_FOOT_POS and DO_PLOTS:
        fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
        axs[0].plot(plot_data_measured[:,0], plot_data_measured[:,5], label=r"$foot position forward, model_{sim}$")
        axs[0].fill_between(plot_data_measured[:,0], plot_data_measured[:,21],plot_data_measured[:,22], alpha=.5)
        axs[0].plot(plot_data_measured[:,0], plot_data_measured[:,11], '-o', label=r"$foot position forward, meas_{sim}$")
        # axs[0].plot(plot_data[:,0], HSDetected_vec_hardware*1e-1, label=r"$HSDetected Hardware$")
        axs[0].plot(plot_data[:,0], plot_data[:,7]*1e-1, 'r', label=r"$HSDetected Sim$")
        axs[0].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")

        axs[0].legend()

        axs[1].plot(plot_data_measured[:,0], plot_data_measured[:,6], label=r"$foot position up, model_{sim}$")
        axs[1].fill_between(plot_data_measured[:,0], plot_data_measured[:,23],plot_data_measured[:,24], alpha=.5)
        axs[1].plot(plot_data_measured[:,0], plot_data_measured[:,12], '-o', label=r"$foot position up, meas_{sim}$")
        # axs[1].plot(plot_data[:,0], HSDetected_vec_hardware*1e-1, label=r"$HSDetected hardware$")
        axs[1].plot(plot_data[:,0], plot_data[:,7]*1e-1, 'r', label=r"$HSDetected Sim$")
        axs[1].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")
        axs[1].legend()

        print("this is done (plot foot pos)")

    if not True and DO_PLOTS:
        fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
        axs[0].plot(plot_data_measured[:,0], plot_states[:,0],'-o', label=r"$state 1$")
        axs[0].plot(plot_data_measured[:,0], plot_states[:,1],'-o', label=r"$state 2$")
        axs[0].set_ylabel('Foot Forward')
        axs[0].legend()


        axs[1].plot(plot_data_measured[:,0], plot_states[:,2],'-o', label=r"$state 1$")
        axs[1].plot(plot_data_measured[:,0], plot_states[:,3],'-o', label=r"$state 2$")
        axs[1].set_ylabel('Foot Up')
        axs[1].legend()



    if not True and DO_PLOTS:

        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(plot_data_measured[:,15], plot_data_measured[:,27])
        axs.set_xlabel('Foot pos forward')
        axs.set_ylabel('Foot pos Up')

        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(plot_data_measured[:,0], plot_data_measured[:,28],label='arctan angle')
        axs.set_xlabel('Time')


    #PLOT FOOT IMU ACCEL
    if True and DO_PLOTS:
        fig, axs = plt.subplots(4,1,sharex=True,figsize=(10,6))

        axs[0].plot(timeSec_vec_hardware, heelAccForward_meas_fromDeltaVelocity_vec_hardware, label=r"$heel acc from delta vel$")

        axs[0].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[0].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e1,'k', label=r"$isOverriding sim$")

        axs[1].plot(timeSec_vec_hardware, heelAccSide_meas_fromDeltaVelocity_vec_hardware, label=r"$heel side from delta vel$")

        axs[1].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[1].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")

        axs[2].plot(timeSec_vec_hardware, heelAccUp_meas_fromDeltaVelocity_vec_hardware, label=r"$heel up from delta vel$")

        axs[2].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[2].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")

        axs[3].plot(timeSec_vec_hardware, heelAccNorm_meas_fromDeltaVelocity_vec_hardware, label=r"$heel accel norm$")
        axs[3].plot(plot_data[:,0], plot_data[:,7]*1e1, 'r', label=r"$HSDetected Sim$")
        axs[3].plot(plot_data[:,0], plot_data_HPEKF[:,1]*1e-1,'k', label=r"$isOverriding sim$")


    if not True and DO_PLOTS:
        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(timeSec_vec_hardware[1:], dt_vec_hardware, label=r"$dt$")

    phase_sim_ekf = plot_data[:,1].reshape(-1,1)
    speed_sim_ekf = plot_data[:,2].reshape(-1,1) * plot_data[:,3].reshape(-1,1)
    incline_sim_ekf = plot_data[:,4].reshape(-1,1)

    phase_sim_tbe = plot_data_TBE[:,1].reshape(-1,1)

    return phase_sim_ekf, speed_sim_ekf, incline_sim_ekf, phase_sim_tbe


def sim_other_models_over_multiple_files(filenames, vicon_filenames):
    """This function runs the simulation of the TBE and EKF over multiple paired exoboot and vicon filenames
        that have been exported using the processing scripts for each participant
        e.g. filenames[i] corresponds to the same trial as vicon_filenames[i]

    Args:
        filenames (list): an ordered list of the exoboot filenames
        vicon_filenames (list): an ordered list of the corresponding vicon filenames
    """    
    phase_rmses_ekf = []
    speed_rmses_ekf = []
    incline_rmses_ekf = []
    phase_rmses_tbe = []

    phase_rmses_ekf_ss = []
    speed_rmses_ekf_ss = []
    incline_rmses_ekf_ss = []
    phase_rmses_tbe_ss = []

    phase_rmses_ekf_nss = []
    speed_rmses_ekf_nss = []
    incline_rmses_ekf_nss = []
    phase_rmses_tbe_nss = []

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
        SUBJECT_LEG_LENGTH = 0.935 #AB01
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

            # determine if we're steady state
            is_steady_state_ekf_step = is_steady_state_ekf[current_idx:next_idx].flatten()
            is_steady_state_ekf_step_bool = np.average(is_steady_state_ekf_step) >= 0.5
            # print(is_steady_state_ekf_step.shape)
            # print(is_steady_state_ekf_step)
            # print(is_steady_state_ekf_step_bool)


            # input()

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

            # handle steady state/transitory specific errors
            if is_steady_state_ekf_step_bool:
                phase_rmses_ekf_ss.append(phase_rmse_step)
                speed_rmses_ekf_ss.append(speed_rmse_step)
                incline_rmses_ekf_ss.append(incline_rmse_step)

            else:
                phase_rmses_ekf_nss.append(phase_rmse_step)
                speed_rmses_ekf_nss.append(speed_rmse_step)
                incline_rmses_ekf_nss.append(incline_rmse_step)

            # input()

        #run through tbe data
        for i in range(num_phase_events_tbe-1):
            current_idx = phase_event_idxs_tbe[i]
            next_idx = phase_event_idxs_tbe[i+1]

            # print(f'current_idx: {current_idx}')
            # print(f'next_idx: {next_idx}')

            phase_sim_tbe_step = phase_sim_tbe[current_idx:next_idx,:].reshape(-1)
            true_labels_step = true_labels_tbe[current_idx:next_idx,:]

            # determine if we're steady state
            is_steady_state_tbe_step = is_steady_state_tbe[current_idx:next_idx].flatten()
            is_steady_state_tbe_step_bool = np.average(is_steady_state_tbe_step) >= 0.5

            # print(speed_sim_tbe_step)
            phase_rmse_step = np.sqrt(np.mean(phase_dist(phase_sim_tbe_step, true_labels_step[:,0])**2))

            #update RMSE overall vector
            phase_rmses_tbe.append(phase_rmse_step)

            # handle steady state/transitory specific errors
            if is_steady_state_tbe_step_bool:
                phase_rmses_tbe_ss.append(phase_rmse_step)

            else:
                phase_rmses_tbe_nss.append(phase_rmse_step)


            # input()


    phase_rmses_ekf = np.array(phase_rmses_ekf)
    speed_rmses_ekf = np.array(speed_rmses_ekf)
    incline_rmses_ekf = np.array(incline_rmses_ekf)
    phase_rmses_tbe = np.array(phase_rmses_tbe)

    phase_rmses_ekf_ss = np.array(phase_rmses_ekf_ss)
    speed_rmses_ekf_ss = np.array(speed_rmses_ekf_ss)
    incline_rmses_ekf_ss = np.array(incline_rmses_ekf_ss)
    phase_rmses_tbe_ss = np.array(phase_rmses_tbe_ss)

    phase_rmses_ekf_nss = np.array(phase_rmses_ekf_nss)
    speed_rmses_ekf_nss = np.array(speed_rmses_ekf_nss)
    incline_rmses_ekf_nss = np.array(incline_rmses_ekf_nss)
    phase_rmses_tbe_nss = np.array(phase_rmses_tbe_nss)

    print('EKF')
    print(phase_rmses_ekf)
    print('Overall')
    print(f'phase_loss_avg: {np.mean(phase_rmses_ekf)} +- {np.std(phase_rmses_ekf)}')
    print(f'speed_loss_avg: {np.mean(speed_rmses_ekf)} +- {np.std(speed_rmses_ekf)}')
    print(f'incline_loss_avg: {np.mean(incline_rmses_ekf)} +- {np.std(incline_rmses_ekf)}')

    print('Steady State')
    print(f'phase_loss_avg: {np.mean(phase_rmses_ekf_ss)} +- {np.std(phase_rmses_ekf_ss)}')
    print(f'speed_loss_avg: {np.mean(speed_rmses_ekf_ss)} +- {np.std(speed_rmses_ekf_ss)}')
    print(f'incline_loss_avg: {np.mean(incline_rmses_ekf_ss)} +- {np.std(incline_rmses_ekf_ss)}')

    print('Transitory')
    print(f'phase_loss_avg: {np.mean(phase_rmses_ekf_nss)} +- {np.std(phase_rmses_ekf_nss)}')
    print(f'speed_loss_avg: {np.mean(speed_rmses_ekf_nss)} +- {np.std(speed_rmses_ekf_nss)}')
    print(f'incline_loss_avg: {np.mean(incline_rmses_ekf_nss)} +- {np.std(incline_rmses_ekf_nss)}')
    print()

    print('TBE')
    print('Overall')
    print(f'phase_loss_avg: {np.mean(phase_rmses_tbe)} +- {np.std(phase_rmses_tbe)}')
    print('Steady State')
    print(f'phase_loss_avg: {np.mean(phase_rmses_tbe_ss)} +- {np.std(phase_rmses_tbe_ss)}')
    print('Transitory')
    print(f'phase_loss_avg: {np.mean(phase_rmses_tbe_nss)} +- {np.std(phase_rmses_tbe_nss)}')


    print()
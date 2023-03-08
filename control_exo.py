'''
Live script to run on the RasPi to control the ExoBoots
Handles reading from the IMU and processing the sensor
readings before sending them to an offboard script. 
Receives gait states from the offboard script.
Uses live gait states to command exo torque
Logs live data
'''

import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append(thisdir + '/utils')

sys.path.append('../')
sys.path.append('../utils/')

sys.path.append('/home/pi/code/ml-gait-estimation/')
sys.path.append('/home/pi/code/ml-gait-estimation/utils')


from time import sleep, time, strftime, perf_counter
import traceback
import csv
import numpy as np
from scipy import linalg
import scipy.linalg
from StatProfiler import StatProfiler
from SoftRealtimeLoop import SoftRealtimeLoop
from ActPackMan import ActPackMan, FlexSEA
from attitude_ekf import AttitudeEKF
from torque_profile import TorqueProfile
from AhrsManager import AhrsManager
from UdpBinarySynch import UdpBinarySynchB
from training_utils import convert_unique_cov_vector_to_mat

from filter_classes import FirstOrderLowPassLinearFilter
from phase_ekf import PhaseEKF

import callibrate_angles_nls as ca
from utils.attitudeEstimatorEKFFuncs import extractEulerAngles_new

np.set_printoptions(precision=4)

FADE_IN_TIME = 1.0

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

#set up conversion factors that scale sensor data to human interpretable units
accelScaleFactor = 8192 #LSB/g
gyroScaleFactor = 32.8 #LSB/ deg/s
degToCount = 45.5111
countToDeg = 1/degToCount
accelNormCutoff = 1.15

#account for small biases in the gyro measurment, in the IMU frame in rad/s
gyroX_IMU_bias = 0.021065806164927196
gyroY_IMU_bias = 0.01037782833021424
gyroZ_IMU_bias = 0.007913779656035359

eye3 = np.eye(3)
eye6 = np.eye(6)
eye4= np.eye(4)


# ===== CORRECT THE EXOSKELETON IMU ====
#good for the right leg
theta_correction = 39.1090 * np.pi/180
#correct to non tilted axes
Rot_unskew = np.array(  [[np.cos(theta_correction), -np.sin(theta_correction),0],[np.sin(theta_correction), np.cos(theta_correction),0],[0, 0, 1]])

# correct to z up, x forward, y left
Rot1 = np.array( [[1, 0, 0],[0,np.cos(-np.pi/2), -np.sin(-np.pi/2)],[0,np.sin(-np.pi/2), np.cos(-np.pi/2)]] )
Rot2 = np.array( [[np.cos(-np.pi/2), 0 ,np.sin(-np.pi/2)],[0,1, 0],[-np.sin(-np.pi/2),0, np.cos(-np.pi/2)]]  )
Rot_correct = Rot2 @ Rot1 @ Rot_unskew
Rot3 = np.array( [[np.cos(np.pi/4), 0 ,np.sin(np.pi/4)],[0,1, 0],[-np.sin(np.pi/4),0, np.cos(np.pi/4)]]  )
Rot4 = np.array( [[1, 0, 0],[0,np.cos(np.pi), -np.sin(np.pi)],[0,np.sin(np.pi), np.cos(np.pi)]] )
Rot5 = np.array( [[np.cos(np.pi), 0 ,np.sin(np.pi)],[0,1, 0],[-np.sin(np.pi),0, np.cos(np.pi)]]  )
Rot_correct = Rot5 @ Rot4 @ Rot3 @ Rot_correct

Kt = 0.14 * 0.537/np.sqrt(2) #exo motor constant
N_avg = 15

#set up which exoskeleton side is used
side = 'right'

sideMultiplier = 1
if (side == "left" or side == "l"):
    sideMultiplier = -1
elif (side == "right" or side == "r"):
    sideMultiplier = 1


def runPB_EKF(exo_right, writer, fd_l, am, run_time = 60*10):
    """This function is called live to run the exo

    Args:
        exo_right: exoboot object for the right exo
        writer: writer containing file to write to
        fd_l: file handle
        am: AHRS manager instance
        run_time (float, optional): total run time in seconds for the exo. Defaults to 60*10.
    """    
    #set up Attitude EKF
    attitude_ekf_args = {'sigma_gyro':0.0023,
                            'sigma_accel': 0.0032*5*1/5,
                            'sigma_q_AE':1e2,
                            'Q_pos_scale':1e-10}
    
    attitude_ekf=AttitudeEKF(**attitude_ekf_args)

    #set up the torque profile
    model_dict = {'model_filepath': 'torque_profile/torque_profile_coeffs.csv',
				'phase_order': 20,
				'speed_order': 1,
				'incline_order': 1}

    model_dict_stairs = {'model_filepath': 'torque_profile/torque_profile_stairs_coeffs.csv',
                    'phase_order': 20,
                    'speed_order': 1,
                    'stair_height_order': 1}

    torque_profile = TorqueProfile(model_dict=model_dict,model_dict_stairs=model_dict_stairs)
    
    #set up scaling factors for gait states
    speed_scale = (0,2)
    incline_scale = (-10,10)
    stair_height_scale = (-1,1)

    #set up kinematics normalization factor
    meas_scale = np.array([
                        [-69.35951035,  27.62815047],
                        [-456.18013759,  401.13782617],
                        [-63.71649984,  22.06632622],
                        [-213.4786175,   396.93801619],
                        [-35.26603985,  20.78473636],
                        [-20.95456523,  14.63961137],
                        [0,1]])


    # set up confidence model, which is the average covariance between the gait model and the measurements
    confidence_covariance = np.array([[ 6.22071536e+01,  4.13354403e+00,  4.03713665e+01,  2.72375660e+01,
   1.17944186e+00,  4.18763741e+00],
 [ 4.13354403e+00,  5.03677224e+03, -2.05252589e+01,  2.15252953e+03,
   1.06062729e+01,  1.22145608e+00],
 [ 4.03713665e+01, -2.05252589e+01,  4.98263772e+01,  2.72531664e+00,
  -2.05628825e+00,  1.93535187e+00],
 [ 2.72375660e+01,  2.15252953e+03,  2.72531664e+00,  2.18753110e+03,
   1.96397551e+01,  1.75659746e+01],
 [ 1.17944186e+00,  1.06062729e+01, -2.05628825e+00,  1.96397551e+01,
   2.07395652e+01, -1.28210406e+00],
 [ 4.18763741e+00,  1.22145608e+00,  1.93535187e+00,  1.75659746e+01,
  -1.28210406e+00,  1.30252823e+01]])
    
    confidence_covariance_inv = np.linalg.inv(confidence_covariance)
    
    #set up EKF parameters
    #set up the importance of the mahalanobis distance used to scale trust in the 
    # gait state estimates from the Transformer. The higher the importance, the 
    # faster the dropoff in trust in the Transformer estimates
    m_dist_importance = 10 
    
#     sigma_q_phase=3e-2
#     sigma_q_speed=4e-2
#     sigma_q_incline=3e-1
#     sigma_q_is_stairs=7e-2
    
#     R_meas_gait_state = np.diag([
#         0.04**2,
#         0.09**2,
#         1.0**2,
#         0.1**2,
#         ])
    
    #set up process noises for EKF state estimates
    sigma_q_phase=7e-2
    sigma_q_speed=4e-2
    sigma_q_incline=7e-1
    sigma_q_is_stairs=7e-2
    
    #set up the measurement noises for the gait states from the Transformer
    R_meas_gait_state = np.diag([
        0.02**2,
        0.09**2,
        1.0**2,
        0.1**2,
        ])
    
    #set up measurement standard deviation noises for the kienematics
    sigma_foot = 3
    sigma_foot_vel = 30
    sigma_shank = 20
    sigma_shank_vel = 100
    sigma_heel_acc_forward = 3
    sigma_heel_acc_up = 2.5
    
    R_meas_kinematics = np.diag([sigma_foot**2,
    sigma_foot_vel**2,\
    sigma_shank**2,
    sigma_shank_vel**2,\
    sigma_heel_acc_forward**2, 
    sigma_heel_acc_up**2
    ])
    
    #set up other EKF factors
    DO_GAIT_MODEL_IN_EKF = True
    DO_HETEROSCEDASTIC = True and DO_GAIT_MODEL_IN_EKF
    if DO_GAIT_MODEL_IN_EKF:
        R_meas = np.block([
                    [R_meas_gait_state,               np.zeros((4, 6))],
                    [np.zeros((6, 4)), R_meas_kinematics]
                ])
    else:
        R_meas = R_meas_gait_state
        
    CANCEL_STAIRS = not True
    
    phase_ekf_args = {'R': R_meas,
                'sigma_q_phase':sigma_q_phase,
                'sigma_q_speed':sigma_q_speed,
                'sigma_q_incline':sigma_q_incline,
                'sigma_q_is_stairs':sigma_q_is_stairs,
                'speed_scale':speed_scale,
                'incline_scale':incline_scale,
                'stair_height_scale':stair_height_scale,
                'meas_scale':meas_scale,
                'm_dist_importance':m_dist_importance,
                'DO_GAIT_MODEL_IN_EKF':DO_GAIT_MODEL_IN_EKF,
                'DO_HETEROSCEDASTIC':DO_HETEROSCEDASTIC,
                'CANCEL_STAIRS':CANCEL_STAIRS
                }
    
    phase_ekf = PhaseEKF(**phase_ekf_args)   
    
    
    
    startTime = time()
    prevTime = 0
    # inProcedure = True
    i = 0
    dt = 1/180

    DRAW_CURRENT = True #whether to actuate torque
    CORRECT_VICON = True #whether to correct for the expected yaw rotation of the foot IMU
    HSDetected = False

    updateFHfreq = 20
    isUpdateTime = True
    isUpdateR = False

    SHANK_ANGLE_OFFSET_VICON = -10 #people tend to stand up at -10 degrees in the Vicon data, so account for that
    
    ML_WINDOW_SIZE = 150 #the number of kinematics samples buffered into the Transformer
    
    #set up filters for the heel acc's
    heel_forward_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    heel_up_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    
    exo_right.set_current_gains()

    #Set up UDP comms code
    synch = UdpBinarySynchB(
        recv_IP="192.168.1.104",
        recv_port=5558,
        send_IP="192.168.1.127", 
        send_port=5557)
    
    prev_synch_count = 0

    loop = SoftRealtimeLoop(dt=0.01, report=True, fade=0.1)

    #main loop
    for t in loop:# inProcedure:
        mainprof.tic()
        firsthalf.tic()

        unskewTime0 = time()
        currentTime = time()
        timeSec = currentTime - startTime

        isUpdateTime = (timeSec % 1/updateFHfreq  < 1e-2)
        dt = timeSec - prevTime

        #UPDATE THE EXO STATE
        read_exo_prof.tic()
        exo_right.update()
        exoState=exo_right.act_pack
        read_exo_prof.toc()
    
        prof_accel_gyro_math.tic()

        #EXTRACT SIGNALS FROM EXOSKELETON
        accelX = exoState.accelx/accelScaleFactor # in units of g
        accelY = exoState.accely/accelScaleFactor
        accelZ = exoState.accelz/accelScaleFactor

        gyroX = exoState.gyrox/gyroScaleFactor * np.pi/180 #in units of rad/s
        gyroY = exoState.gyroy/gyroScaleFactor * np.pi/180
        gyroZ = exoState.gyroz/gyroScaleFactor * np.pi/180

        #express imu exo measurements in the correct frame
        accelVec = np.array([accelX,accelY,accelZ])
        accelVec_corrected = Rot_correct @ (accelVec)
        accelNorm = np.linalg.norm(accelVec_corrected)
        gyroVec = np.array([gyroX,gyroY,gyroZ])
        gyroVec_corrected = Rot_correct @ (gyroVec)
        gyroVec_corrected = gyroVec_corrected - Rot_correct @ np.array([gyroX_IMU_bias,gyroY_IMU_bias,gyroZ_IMU_bias])
    
        ankleAngle_buffer = exoState.ank_ang
        ankleAngleVel_buffer = exoState.ank_vel
        motorCurrent_buffer = exoState.mot_cur
    
        accelZ = accelVec_corrected[2]
        gyroY = gyroVec_corrected[1]

        prof_accel_gyro_math.toc()
        
        # read foot IMU data
        ahrs_prof.tic()
        am.update()
        ahrs_prof.toc()

        filters_prof.tic()

        unskewTime1 = time()
        unskewTime = unskewTime1 - unskewTime0
    
        #READ IN AHRS ACCEL
        acc_vec_ahrs = am.get_linear_acc(CORRECT_VICON=CORRECT_VICON)
        heelAccForward_meas = acc_vec_ahrs[0,0]
        heelAccSide_meas = acc_vec_ahrs[1,0]
        heelAccUp_meas = acc_vec_ahrs[2,0]# - 9.81
        
        #filter the noisy accelerometer signals
        heelAccForward_meas_filt = heel_forward_acc_filter.step(i, heelAccForward_meas)
        heelAccUp_meas_filt = heel_up_acc_filter.step(i, heelAccUp_meas)

        acc_vec_ahrs_fromDeltaVelocity = am.get_linear_acc_fromDeltaVelocity()
        heelAccForward_meas_fromDeltaVelocity = acc_vec_ahrs_fromDeltaVelocity[0,0]
        heelAccSide_meas_fromDeltaVelocity = acc_vec_ahrs_fromDeltaVelocity[1,0]
        heelAccUp_meas_fromDeltaVelocity = acc_vec_ahrs_fromDeltaVelocity[2,0]# - 9.81

        filters_prof.toc()
        firsthalf.toc()

        #CHECK THE SOFT LIMIT
        if abs(motorCurrent_buffer) > 35000: #if the motor current magnitude exceeds 40 Amps
        
            print("HIT CURRENT SOFT LIMIT\n")
            raise Exception("HIT CURRENT SOFT LIMIT\n")
            beforeExiting()

        #step through attitude EKF
        attStep_prof.tic()
        attitude_ekf.step(i, dt, isUpdateTime)
        attStep_prof.toc()

        # run the attitude EKF measurement step
        attUpdate_prof.tic()
        attitude_ekf.measure(i, gyroVec_corrected, accelVec_corrected, isUpdateTime, CORRECT_VICON)
        psi, theta, phi = attitude_ekf.get_euler_angles()
        attUpdate_prof.toc()

        prevTime = timeSec

        math_prof.tic()

        #Compute shank angle 
        ankleAngle = sideMultiplier * ((ankleAngle_buffer * countToDeg ) - ankleAngleOffset)
        shankAngle_meas = attitude_ekf.get_useful_angles(sideMultiplier)

        #offset shank angle and correct for Vicon offset
        shankAngle_meas = shankAngle_meas - shankAngleOffset + SHANK_ANGLE_OFFSET_VICON

        # print(shankAngle_meas)

        # compute foot angle using the rotation matrix from the foot IMU
        R_foot = am.get_R_foot(CORRECT_VICON=CORRECT_VICON)
        roll, pitch, yaw = extractEulerAngles_new(R_foot)
        footAngle_meas = -pitch*180/np.pi
        footAngle_meas = footAngle_meas - footAngleOffset

        #ensure foot angle is +- 180 degrees
        if footAngle_meas < -180:
            footAngle_meas += 360

        if footAngle_meas > 180:
            footAngle_meas -= 360

        #calculate shank velocity
        shankAngleVel_meas = sideMultiplier * -1 * gyroVec_corrected[1] * 180/np.pi

        #compute foot angle velocity
        ang_vel_vec_ahrs = -1 * am.get_rotational_vel(CORRECT_VICON=CORRECT_VICON)
        footAngleVel_meas = ang_vel_vec_ahrs[1,0]

        # print(footAngleVel_meas)

        math_prof.toc()
        getters_prof.tic()
        
        # run prediction step on EKF
        #change functionality st it doesnt iterate vefore ML window size
        phase_ekf.step(i, dt)
        
        
        z_measured = np.array([footAngle_meas, footAngleVel_meas, shankAngle_meas, shankAngleVel_meas, \
            heelAccForward_meas_filt, heelAccUp_meas_filt, dt]).reshape(-1,1)
        
        #send both the measurements from the exo and the current ekf state
        data_to_computer = np.vstack((z_measured, phase_ekf.x_state.reshape(-1,1)))
        
        #send data to pi and receive a numpy column vector of :
        # - predicted_gait_state_neural_net (4x1)
        # - predicted_kinematics_neural_net (6x1)
        # - z_model_kinematics (6x1)
        # - kinematics_gradient (30x1)
        # - heteroscedastic_vec (21x1)
        data_from_computer = synch.update(data_to_computer)

        #update synch_count
        synch_count = synch.my_count
        
        # print(data_from_computer.shape)
        # print(predicted_gait_state_neural_net)

        #the gait transformer will only output meaningful results after ML_WINDOW_SIZE samples have been sent
        #until that happens, overwrite the state estimates
        if i < ML_WINDOW_SIZE:
             data_from_computer = np.zeros((16+30+21,))
                
        predicted_gait_state_neural_net = data_from_computer[0:4].reshape(-1,1)
        predicted_kinematics_neural_net = data_from_computer[4:10].reshape(-1,1)
        predicted_kinematics_ekf = data_from_computer[10:16].reshape(-1,1)
        kinematics_gradient = data_from_computer[16:16+30]
        kinematics_gradient = kinematics_gradient.reshape(6,5)
        heteroscedastic_vec = data_from_computer[16+30:16+30+21]
        
                
        #compute mahalanobis distances for kinematics predicted using
        d = z_measured[:-1,:] - predicted_kinematics_neural_net
        m_distance_neural_net_predictions = np.sqrt(d.T @ confidence_covariance_inv @ d)[0,0]
        # print(m_distance_neural_net_predictions)
        
        #run update step on EKF
        z_measured_ekf = predicted_gait_state_neural_net
        
        #only update if we received new data from the neural network
        RECEIVED_NEW_DATA = not (synch_count == prev_synch_count)
        
        if RECEIVED_NEW_DATA:
            if DO_GAIT_MODEL_IN_EKF:
                #generate measurements for ekf by stacking the measured gait state and the measurements of the kinematics
                z_measured_ekf = np.vstack((predicted_gait_state_neural_net, z_measured[:-1,:]))
                
                R_heteroscedastic=None
                if DO_HETEROSCEDASTIC:
                    R_heteroscedastic = convert_unique_cov_vector_to_mat(heteroscedastic_vec,dim_mat=6)

                phase_ekf.update(i, dt, z_measured_ekf, z_model_kinematics=predicted_kinematics_ekf, kinematics_gradient=kinematics_gradient,m_dist=m_distance_neural_net_predictions, R_heteroscedastic=R_heteroscedastic)
                
            else:
                phase_ekf.update(i, dt, z_measured_ekf, m_dist=m_distance_neural_net_predictions)
            
        predicted_gait_state_ekf = phase_ekf.x_state.reshape(-1,1)
        
        #compute mahalanobis distance from predicted kinematics
        d = z_measured[:-1,:] - predicted_kinematics_ekf.reshape(-1,1)
        m_distance_ekf_predictions = np.sqrt(d.T @ confidence_covariance_inv @ d)[0,0]
        

        # print(data_from_computer)
        phase = predicted_gait_state_ekf[0,0]
        speed = predicted_gait_state_ekf[1,0]
        incline = predicted_gait_state_ekf[2,0]
        stair_height = predicted_gait_state_ekf[3,0]

        
        getters_prof.toc()

        # Calculate desired torque using the states as inputs to the torque profile
        #for simplicity, 
        desTorque = torque_profile.evalTorqueProfile(phase, speed, incline, stair_height)

        #calculate desired current
        desCurrent = desTorque / (Kt * N_avg * sideMultiplier)
        
        #if we don't have enough samples for accurate prediction, do zero torque
        if i < ML_WINDOW_SIZE:
            desCurrent = 0
        
        #add small buffer current to ensure belt tension
        if desCurrent < 0.7:
            desCurrent = 0.7

        if desCurrent >= 25:
            desCurrent = 25
            
        # print(desCurrent) 
        actTorque = motorCurrent_buffer/1000 * Kt * N_avg
        

        if DRAW_CURRENT:
            if t>FADE_IN_TIME:
                exo_right.i = desCurrent*sideMultiplier*loop.fade
            else:
                exo_right.i = (t/FADE_IN_TIME)*desCurrent*sideMultiplier*loop.fade

        log_prof.tic()
        i+=1
        
        prev_synch_count = synch_count


        data_frame_vec = [
            round(timeSec,4), #0
            accelVec_corrected[0], #1
            accelVec_corrected[1], #2
            accelVec_corrected[2], #3
            gyroVec_corrected[0], #4
            gyroVec_corrected[1], #5
            gyroVec_corrected[2], #6
            psi, #7 
            theta, #8
            phi, # 9
            attitude_ekf.z_measured[0,0], #10
            attitude_ekf.z_measured[1,0], #11
            attitude_ekf.z_measured[2,0], #12
            attitude_ekf.z_measured[3,0], #13
            attitude_ekf.z_measured[4,0], #14
            attitude_ekf.z_measured[5,0], #15
            attitude_ekf.z_model[0,0], #16
            attitude_ekf.z_model[1,0], #17
            attitude_ekf.z_model[2,0], #18
            attitude_ekf.z_model[3,0], #19
            attitude_ekf.z_model[4,0], #20
            attitude_ekf.z_model[5,0], #21
            footAngle_meas, #22
            footAngleVel_meas, #23
            shankAngle_meas, #24
            shankAngleVel_meas, #25
            int(HSDetected), #26
            ankleAngle, #27
            actTorque, #28
            desTorque, #29
            roll*180/3.1415, #30
            pitch*180/3.1415, #31
            yaw*180/3.1415, #32
            heelAccForward_meas, #33
            heelAccSide_meas, #34
            heelAccUp_meas, #35
            heelAccForward_meas_fromDeltaVelocity, #36
            heelAccSide_meas_fromDeltaVelocity, #37
            heelAccUp_meas_fromDeltaVelocity, #38
            dt, #39
            phase, #40 
            speed, #41
            incline, #42
            stair_height, #43
            1, #44
            motorCurrent_buffer, #45
            desCurrent, #46
            heelAccForward_meas_filt, #47
            heelAccUp_meas_filt, #48
            predicted_gait_state_neural_net[0,0], #49
            predicted_gait_state_neural_net[1,0], #50
            predicted_gait_state_neural_net[2,0], #51
            predicted_gait_state_neural_net[3,0], #52
            0, #53
            predicted_kinematics_ekf[0,0], #54
            predicted_kinematics_ekf[1,0], #55
            predicted_kinematics_ekf[2,0], #56
            predicted_kinematics_ekf[3,0], #57
            predicted_kinematics_ekf[4,0], #58
            predicted_kinematics_ekf[5,0], #59
            synch.my_count, #60
            m_distance_neural_net_predictions, #61
            m_distance_ekf_predictions#62
        ]
        # print(data_frame_vec)

        writer.writerow(data_frame_vec)
    
        i += 1
        log_prof.toc()

        if t > run_time:
            break
        # if timeSec >= run_time:
        #     inProcedure = False
        #     fxClose(devId)

        mainprof.toc()

        
    return True

if __name__ == '__main__':

    # port_cfg_path = '/home/pi/code/Actuator-Package/Python/flexsea_demo/ports.yaml'
    # ports, baud_rate = fxu.load_ports_from_file(port_cfg_path)

    read_exo_prof = StatProfiler(name="reading_exo")
    prof_accel_gyro_math = StatProfiler(name="accel and gyro math")
    ahrs_prof = StatProfiler(name="Ahrs.update")
    filters_prof = StatProfiler(name="filters")
    HSdetect_prof = StatProfiler(name="detecting HS")
    attStep_prof = StatProfiler(name="attitude step")
    phaseStep_prof = StatProfiler(name="phase step")
    math_prof = StatProfiler(name="Math and getting important euler angles")
    getters_prof = StatProfiler(name="Getters")
    attUpdate_prof = StatProfiler(name="attituse update and get euler")
    log_prof = StatProfiler(name="Logging")

    mainprof = StatProfiler(name="Main Loop")
    firsthalf = StatProfiler(name="first_quarter_loop")

    filename_offsets = 'angleOffsets_right.csv'
    print(filename_offsets)

    attitude_ekf_calibrate=AttitudeEKF()#Attitude_EKF()
    imu_port = "/dev/ttyACM0"

    with ActPackMan("/dev/ttyACM1") as exo_right:
        print("in")
        footAngleOffset, shankAngleOffset, ankleAngleOffset = ca.main(exo_right, filename_offsets, attitude_ekf_calibrate)

        filename = '{0}_gait_transformer.csv'.format(strftime("%Y%m%d-%H_AE%M%S"))
        print(filename)

        input('HIT ENTER TO START')
        with open(filename, "w", newline="\n") as fd_l:
            writer_l = csv.writer(fd_l)
            print(writer_l)
            # with AhrsManager(csv_file_name='{0}_R_matrix_log.csv'.format(strftime("%Y%m%d-%H_AE%M%S"))) as am:
            with AhrsManager() as am:
                runPB_EKF(exo_right,  writer_l, fd_l, am)
    

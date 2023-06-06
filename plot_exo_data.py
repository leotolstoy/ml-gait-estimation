""" Plots the results of an exoskeleton trial """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt


import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt



def main():

    # SUBJECT_LEG_LENGTH = 0.935 #AB01
    # # SUBJECT_LEG_LENGTH = 0.975 # AB02
    # # SUBJECT_LEG_LENGTH = 0.965 #AB03
    # # SUBJECT_LEG_LENGTH = 0.96 #AB04
    # # SUBJECT_LEG_LENGTH = 0.96 #AB05
    # # SUBJECT_LEG_LENGTH = 0.845 #AB06
    # data = np.loadtxt("live_data/20230210-22_AE2538_gait_transformer.csv", delimiter=',')

    # data = np.loadtxt("live_data/AB02-Parallel/circuit1/circuit1_seg1/20230227-19_AB02-Parallel_circuit1_seg1.csv", delimiter=',')
    # data = np.loadtxt("live_data/AB14-Belleau/circuit2/circuit2_seg2/20230320-19_AB14-Belleau_circuit2_seg2.csv", delimiter=',')
    data = np.loadtxt("live_data/misc/20230530-20_AE4223_gait_transformer.csv", delimiter=',')



    timeSec=data[:,0]
    accelVec_corrected=data[:,1:4]

    gyroVec_corrected=data[:,4:7]

    ankleAngle = data[:,25]
    roll = data[:,28]
    pitch = data[:,29]
    yaw = data[:,30]

    phase = data[:,40]
    speed = data[:,41]
    incline = data[:,42]
    is_stairs = data[:,43]
    is_moving = data[:,44]

    actTorque = data[:,28]
    desTorque = data[:,29]

    footAngle_meas = data[:,22]
    footAngleVel_meas = data[:,23]
    shankAngle_meas = data[:,24]
    shankAngleVel_meas = data[:,25]
    heelAccForward_meas = data[:,33] #92
    heelAccSide_meas = data[:,34]#71
    heelAccUp_meas = data[:,35]#71
    heelAccForward_meas_filt = data[:,47]
    heelAccUp_meas_filt = data[:,48]


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



    dt = data[:,39]
    freqData = 1/dt


    plt.figure()
    plt.hist(freqData)
    plt.title('freqData')
    plt.savefig("time_")


    #PLOT STATES
    fig, axs = plt.subplots(4,1,sharex=True,figsize=(10,6))

    axs[0].plot(timeSec, phase,'.', label=r"$phase_{ekf, hardware}$")
    axs[0].plot(timeSec, phase_nn, label=r"$phase_{nn, hardware}$")

    axs[0].legend()

    axs[1].plot(timeSec, speed, label=r"$Speed_{ekf, hardware}$")
    axs[1].plot(timeSec, speed_nn, label=r"$Speed_{nn, hardware}$")
    axs[1].legend()

    axs[2].plot(timeSec, incline, label=r"$Ramp_{ekf, hardware}$")
    axs[2].plot(timeSec, incline_nn, label=r"$Ramp_{nn, hardware}$")
    axs[2].legend()

    axs[3].plot(timeSec, is_stairs, label=r"$Stair Height_{ekf, hardware}$")
    axs[3].plot(timeSec, is_stairs_nn, label=r"$Stair Height_{nn, hardware}$")
    axs[3].legend()


    axs[-1].set_xlabel("time (sec)")
    print("this is done")

    plt.savefig("states_")

    # plt.show()


    #PLOT EXO IMU DATA
    if not True:

        fig, axs = plt.subplots(3,2,sharex=True,figsize=(10,6))

        axs[0,0].plot(timeSec, accelVec_corrected[:,0], label=r"$Accel X$")
        axs[0,0].legend()
        axs[1,0].plot(timeSec, accelVec_corrected[:,1], label=r"$Accel Y$")
        axs[1,0].legend()
        axs[2,0].plot(timeSec, accelVec_corrected[:,2], label=r"$Accel Z$")
        axs[2,0].legend()
        axs[0,1].plot(timeSec, gyroVec_corrected[:,0], label=r"$Gyro X$")
        axs[0,1].legend()
        axs[1,1].plot(timeSec, gyroVec_corrected[:,1], label=r"$Gyro Y$")
        axs[1,1].legend()
        axs[2,1].plot(timeSec, gyroVec_corrected[:,2], label=r"$Gyro Z$")
        axs[2,1].legend()
        axs[-1,0].set_xlabel("time (sec)")
        axs[-1,1].set_xlabel("time (sec)")
        print("this is done")
        # plt.show()

        # plt.savefig(filename)

    #PLOT KINEMATICS
    if True:

        fig, axs = plt.subplots(6,1,sharex=True,figsize=(10,6))
        axs[0].set_title("Kinematics")

        # axs[0].plot(timeSec, plot_data[:,1], label=r"$phase_{hardware}$")
        axs[0].plot(timeSec, footAngle_meas, label=r"$foot angle, measured$")
        axs[0].plot(timeSec, footAngle_predicted_ekf, label=r"$foot angle, predicted, ekf$")
        axs[0].set_ylim([-70,50])
        axs[0].legend()


        axs[1].plot(timeSec, footAngleVel_meas, label=r"$foot angle vel, measured$")
        axs[1].plot(timeSec, footAngleVel_predicted_ekf, label=r"$foot angle vel, predicted, ekf$")

        axs[1].legend()



        axs[2].plot(timeSec, shankAngle_meas, label=r"$shank angle, measured$")
        axs[2].plot(timeSec, shankAngle_predicted_ekf, label=r"$shank angle, predicted, ekf$")

        axs[2].set_ylim([-70,50])
        axs[2].legend()

        axs[3].plot(timeSec, shankAngleVel_meas, label=r"$shank angle vel, measured$")
        axs[3].plot(timeSec, shankAngleVel_predicted_ekf, label=r"$shank angle vel, predicted, ekf$")

        axs[3].legend()


        axs[4].plot(timeSec, heelAccForward_meas, label=r"$heel acc forward measured$")
        axs[4].plot(timeSec, heelAccForward_meas_filt, label=r"$heel acc forward filt measured$")
        axs[4].plot(timeSec, heelAccForward_predicted_ekf, label=r"$heel acc forward, predicted, ekf$")

        axs[4].legend()


        axs[5].plot(timeSec, heelAccUp_meas, label=r"$heel acc up measured$")
        axs[5].plot(timeSec, heelAccUp_meas_filt, label=r"$heel acc up filt measured$")
        axs[5].plot(timeSec, heelAccUp_predicted_ekf, label=r"$heel acc up, predicted, ekf$")

        axs[5].legend()


    if True:
        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(timeSec, actTorque, label=r"$actTorque$")
        axs.plot(timeSec, desTorque, label=r"$desTorque$")
        axs.legend()
        
    if True:
        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(timeSec, dt, label=r"$dt$")

    if True:
        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(timeSec, count, label="count")
        axs.legend()
        axs.set_xlabel("time (sec)")

        # dcountds = np.diff(count)/np.diff(timeSec)
        # axs[1].plot(timeSec[1:], dcountds,'-o', label="rate of count increment")
        # axs[1].legend()
        # axs[1].set_ylim([0,110])

    if True:
        fig, axs = plt.subplots(sharex=True,figsize=(10,6))
        axs.plot(timeSec, m_distance_nn, label="m distance nn")
        axs.plot(timeSec, phase,'k', label="phase")

    plt.show()




if __name__ == '__main__':
    main()
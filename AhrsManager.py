"""Summary
"""
from SoftRealtimeLoop import SoftRealtimeLoop
import time
import csv
import sys, time
import numpy as np
sys.path.append(r'/usr/share/python3-mscl/')    # Path of the MSCL
import traceback
import mscl
from utils.attitudeEstimatorEKFFuncs import extractEulerAngles_new,extractEulerAngles_new_ZYX



class AhrsManager():

    """This class handles the Lord Microstrain AHRS used to
    measure the kinematics of the foot
    
    Attributes:
        accVecCorrect (np matrix): 3-vector of linear accelerations in the world frame
        accX (float): IMU frame linear X acceleration
        accY (float): IMU frame linear Y acceleration
        accZ (float): IMU frame linear Z acceleration
        connection (TYPE): Description
        csv_file (TYPE): logging file handle
        csv_file_name (TYPE): logging filename
        csv_writer (TYPE): Description
        node (TYPE): Description
        port (TYPE): USB port
        prevTime (float): previous time step
        R (TYPE): rotation matrix describing the AHRS' position in 3D space
        R_foot (TYPE): rotation matrix describing the foot position in 3D space
        R_sensor_to_foot (TYPE): Coordinate transform that goes from sensor frame to foot frame
        R_vicon_correct (TYPE): Coordinate transform that accounts for foot (yaw) rotation as read by Vicon
        R_y_correct (TYPE): rotation matrix that ensures plantarflexion rotation is negative 
        save_csv (bool): Whether to write internal logging to a csv or not
    """
    
    def __init__(self, csv_file_name=None, port="/dev/ttyACM0"):
        """A class (by Dr. Gray Thomas) that reads data from a Lord Microstrain AHRS
        
        Args:
            csv_file_name (None, optional): filename to write internal logging data
            port (str, optional): USB port for the AHRS
        """
        self.port = port
        self.save_csv = not (csv_file_name is None)
        self.csv_file_name = csv_file_name
        self.csv_file = None
        self.csv_writer = None
        self.prevTime = 0.0
        co = np.cos(np.pi/180 * 82)
        so = np.sin(np.pi/180 * 82)
        self.R_sensor_to_foot = np.array([[co, 0 , -so],[0, 1, 0],[so, 0, co]]) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        self.R_y_correct = np.array([[-1,0,0],[0,1,0],[0,0,-1]])

        self.accX = 0
        self.accY = 0
        self.accZ = 0
        self.angVelX = 0
        self.angVelY = 0
        self.angVelZ = 0
        self.accVecCorrect = np.zeros((3,1))
        self.angVelVecCorrect = np.zeros((3,1))
        self.deltaVelocity = np.zeros((3,1))

        self.accVecCorrect_fromDeltaVelocity = np.zeros((3,1))

        # VICON SCALE
        vicon_angle = -25
        co_vicon = np.cos(np.pi/180 * (vicon_angle))
        so_vicon = np.sin(np.pi/180 * (vicon_angle)) 
        self.R_vicon_correct = np.array([[co_vicon, -so_vicon, 0],[so_vicon, co_vicon, 0],[0, 0, 1]])
        self.R_foot = np.eye(3)#initialize R_foot to identity for data storage purposes
        

    def __enter__(self):
        if self.save_csv:
            with open(self.csv_file_name,'w') as fd:
                writer = csv.writer(fd)
                writer.writerow(["pi_time",
                    "r00", "r01", "r02",
                    "r10", "r11", "r12",
                    "r20", "r21", "r22",
                    "r00_foot", "r01_foot", "r02_foot",
                    "r10_foot", "r11_foot", "r12_foot",
                    "r20_foot", "r21_foot", "r22_foot",
                    'accX','accY','accZ'])
            self.csv_file = open(self.csv_file_name,'a').__enter__()
            self.csv_writer = csv.writer(self.csv_file)


        self.connection = mscl.Connection.Serial(self.port, 921600)
        self.node = mscl.InertialNode(self.connection)
        # self.node.setToIdle()


        # Clean the internal circular buffer. Select timeout to be 500ms
        # self.packets = self.node.getDataPackets(500)

        # self.deltaTime = 0
        # self.sampleRate = mscl.SampleRate(1,500)
        #Resume node for streaming
        # self.node.resume()
        #if the node supports AHRS/IMU
        if self.node.features().supportsCategory(mscl.MipTypes.CLASS_AHRS_IMU):
            self.node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)

        #if the self.node supports Estimation Filter
        if self.node.features().supportsCategory(mscl.MipTypes.CLASS_ESTFILTER):
            self.node.enableDataStream(mscl.MipTypes.CLASS_ESTFILTER)

        #if the self.node supports GNSS
        if self.node.features().supportsCategory(mscl.MipTypes.CLASS_GNSS):
            self.node.enableDataStream(mscl.MipTypes.CLASS_GNSS)

        return self


    def __exit__(self, etype, value, tb):
        """Closes the file properly 

        """
        if self.save_csv:
            self.csv_file.__exit__(etype, value, tb)
            print('Exiting CSV')
        self.node.setToIdle()
        if not (etype is None):
            traceback.print_exception(etype, value, tb)


    def update(self):
        """Updates/reads from the AHRS
        
        """
        t0=time.time()

        microstrainData = self.readIMUnode(timeout=5)
        # print([microstrainDatum.keys() for microstrainDatum in microstrainData ])
        for datum in microstrainData:
            # print(datum.keys())
            if 'orientMatrix' in datum.keys():
                self.R = datum['orientMatrix']
            if 'estAngularRateX' in datum.keys():
                self.angVelX = datum['estAngularRateX']*180/np.pi
            if 'estAngularRateY' in datum.keys():
                self.angVelY = datum['estAngularRateY']*180/np.pi
            if 'estAngularRateZ' in datum.keys():
                self.angVelZ = datum['estAngularRateZ']*180/np.pi
            if 'estLinearAccelX' in datum.keys():
                self.accX = datum['estLinearAccelX']
            if 'estLinearAccelY' in datum.keys():
                self.accY = datum['estLinearAccelY']
            if 'estLinearAccelZ' in datum.keys():
                self.accZ = datum['estLinearAccelZ']
            if 'deltaVelX' in datum.keys():
                self.deltaVelX = datum['deltaVelX']
            if 'deltaVelY' in datum.keys():
                self.deltaVelY = datum['deltaVelY']
            if 'deltaVelZ' in datum.keys():
                self.deltaVelZ = datum['deltaVelZ']

        # self.R = self.readIMUnode()['orientMatrix']
        # self.R= np.eye(3)
        dur = time.time()-t0
        if self.save_csv:
            self.csv_writer.writerow([time.time()
                , self.R[0,0], self.R[0,1], self.R[0,2]
                , self.R[1,0], self.R[1,1], self.R[1,2]
                , self.R[2,0], self.R[2,1], self.R[2,2]
                , self.R_foot[0,0], self.R_foot[0,1], self.R_foot[0,2]
                , self.R_foot[1,0], self.R_foot[1,1], self.R_foot[1,2]
                , self.R_foot[2,0], self.R_foot[2,1], self.R_foot[2,2]
                , self.accVecCorrect[0,0], self.accVecCorrect[1,0], self.accVecCorrect[2,0]
                ])
            
        return 1

    def get_R_foot(self, CORRECT_VICON=True):
        """Updates and returns the foot rotation matrix (foot orientation in world frame)
        
        Args:
            CORRECT_VICON (bool, optional): Whether to account for Vicon rotation
        
        Returns:
            TYPE: the foot rotation matrix
        """
        #R_foot is world in foot
        if CORRECT_VICON:
            self.R_foot = self.R_vicon_correct @  self.R_sensor_to_foot @ self.R @ self.R_y_correct

        else:
            self.R_foot = self.R_sensor_to_foot @ self.R @ self.R_y_correct
        return self.R_foot

    def applyTransform(self, three_vector):
        yaw, pitch, roll = extractEulerAngles_new_ZYX(self.R_foot.T)
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)],[0, -np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0 , -np.sin(pitch)],[0, 1, 0],[np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), np.sin(yaw) , 0],[-np.sin(yaw), np.cos(yaw), 0],[0, 0, 1]])

        return (Ry @ Rx) @ three_vector



    def get_linear_acc(self,CORRECT_VICON=True):
        """sets and returns the linear accelerations of the heel in the "locomotion frame"
        Rotates in yaw with the person
        
        Returns:
            TYPE: the acceleration vector
        """
        accVec = np.array([self.accX,self.accY,self.accZ])[:,np.newaxis]

        if CORRECT_VICON:
            self.accVecCorrect = self.applyTransform(self.R_vicon_correct @ self.R_sensor_to_foot @ accVec)
        else:
            self.accVecCorrect = self.applyTransform(self.R_sensor_to_foot @ accVec)
        return self.accVecCorrect

    def get_rotational_vel(self,CORRECT_VICON=True):
        """sets and returns the linear accelerations of the heel in the "locomotion frame"
        Rotates in yaw with the person
        
        Returns:
            TYPE: the acceleration vector
        """
        angVelVec = np.array([self.angVelX,self.angVelY,self.angVelZ])[:,np.newaxis]


        if CORRECT_VICON:
            self.angVelVecCorrect = self.applyTransform(self.R_vicon_correct @ self.R_sensor_to_foot @ angVelVec)
        else:
            self.angVelVecCorrect = self.applyTransform(self.R_sensor_to_foot @ angVelVec)
        return self.angVelVecCorrect


    def get_linear_acc_fromDeltaVelocity(self,CORRECT_VICON=True):
        """sets and returns the linear accelerations of the heel in the "locomotion frame"
        Rotates in yaw with the person
        
        Returns:
            TYPE: the acceleration vector
        """
        accVec = np.array([self.deltaVelX,self.deltaVelY,self.deltaVelZ])[:,np.newaxis]/0.01 *9.81

        if CORRECT_VICON:
            self.accVecCorrect_fromDeltaVelocity = self.applyTransform(self.R_vicon_correct @ self.R_sensor_to_foot @ accVec)
        else:
            self.accVecCorrect_fromDeltaVelocity = self.applyTransform(self.R_sensor_to_foot @ accVec)
        return self.accVecCorrect_fromDeltaVelocity


        

    def readIMUnode(self, timeout = 5):
        """Reads the IMU node for data packets
        
        Args:
            timeout (int, optional): Description
        
        Returns:
            TYPE: Description
        """
        # print(f'totalPackets: {self.node.totalPackets()}')
        packets = self.node.getDataPackets(timeout, maxPackets=0)
        microstrainData = []
        for packet in packets[-2:]:
            microstrainDatum = dict()
            for dataPoint in packet.data():
                if dataPoint.storedAs() == 0:
                    microstrainDatum[dataPoint.channelName()] = dataPoint.as_float()
                
                elif dataPoint.storedAs() == 3:
                    # print(dir(dataPoint))
                    # ts = dataPoint.as_Timestamp()
                    microstrainDatum[dataPoint.channelName()] = None

                elif dataPoint.storedAs() == 1:
                    # print(dir(dataPoint))
                    ts = dataPoint.as_double()
                    microstrainDatum[dataPoint.channelName()] = ts
                    
                elif dataPoint.storedAs() == 9:
                    mat = dataPoint.as_Matrix()
                    npmat = np.array([[mat.as_floatAt(i,j) for j in range(3)] for i in range(3)])
                    microstrainDatum[dataPoint.channelName()] = npmat
                else:
                    print("no solution for datapoint stored as", dataPoint.storedAs(), dataPoint.channelName())
                    microstrainDatum[dataPoint.channelName()] = None
            microstrainData.append(microstrainDatum)
        return microstrainData


def main():
    """Summary
    """
    with AhrsManager(csv_file_name="test_ahrs.csv") as am:
        class looper():
            
            def __init__(self):
                """Summary
                """
                self.i=0
            def __call__(self):
                """Summary
                """
                am.update()
                R=am.R
                if self.i%100==5:
                    time.sleep(.1)

                if self.i%10==0:
                    R_foot = am.get_R_foot()
                    print('R_foot')
                    print(R_foot)

                    print('R_foot normal')
                    roll, pitch, yaw = extractEulerAngles_new(R_foot)


                    print(roll*180/np.pi)
                    print(pitch*180/np.pi)
                    print(yaw*180/np.pi)

                    Rx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)],[0, -np.sin(roll), np.cos(roll)]])
                    Ry = np.array([[np.cos(pitch), 0 , -np.sin(pitch)],[0, 1, 0],[np.sin(pitch), 0, np.cos(pitch)]])
                    Rz = np.array([[np.cos(yaw), np.sin(yaw) , 0],[-np.sin(yaw), np.cos(yaw), 0],[0, 0, 1]])
                    
                    print(Rx @ Ry @ Rz)
                    cr = np.cos(roll)
                    sr = np.sin(roll)
                    cp = np.cos(pitch)
                    sp = np.sin(pitch)
                    cy = np.cos(yaw)
                    sy = np.sin(yaw)

                    print('Accels')
                    acc_vec = am.get_linear_acc(CORRECT_VICON=False)

                    print(f'Acc X: {acc_vec[0]}')
                    print(f'Acc Y: {acc_vec[1]}')
                    print(f'Acc Z: {acc_vec[2]}')

                    acc_vec_fromDeltaVelocity = am.get_linear_acc_fromDeltaVelocity(CORRECT_VICON=False)

                    print(f'Acc X _fromDeltaVelocity: {acc_vec_fromDeltaVelocity[0]}')
                    print(f'Acc Y _fromDeltaVelocity: {acc_vec_fromDeltaVelocity[1]}')
                    print(f'Acc Z _fromDeltaVelocity: {acc_vec_fromDeltaVelocity[2]}')

                    # print(np.array([[cp*cy, cp*sy, -sp],[sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],[cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]]))
                    # print('R_foot transpose')
                    # roll, pitch, yaw = extractEulerAngles(R_foot.T)
                    # Rz = np.array([[np.cos(yaw), np.sin(yaw) , 0],[-np.sin(yaw), np.cos(yaw), 0],[0, 0, 1]])
                    # Ry = np.array([[np.cos(pitch), 0 , -np.sin(pitch)],[0, 1, 0],[np.sin(pitch), 0, np.cos(pitch)]])
                    # Rx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)],[0, -np.sin(roll), np.cos(roll)]])

                    # print(roll*180/np.pi)
                    # print(pitch*180/np.pi)
                    # print(yaw*180/np.pi)
                    # print(Rz @ Ry @ Rx)



                self.i+=1
        my_looper=looper()
        for t in SoftRealtimeLoop(1/100):
            my_looper() 

if __name__ == '__main__':
    main()

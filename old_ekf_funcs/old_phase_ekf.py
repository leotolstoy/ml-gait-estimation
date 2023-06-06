"""Made by Leo. Contains the full EKF for phase
"""
from time import time
import numpy as np
from arctanMapFuncs import *

class PhaseEKF():

    """Summary
    
    Attributes:
        F (TYPE): Description
        F0 (TYPE): Description
        gait_model (TYPE): Description
        meas_config (TYPE): Description
        measurement_noise_model (TYPE): Description
        P0 (TYPE): Description
        P_covar_estimate (TYPE): Description
        P_covar_update (TYPE): Description
        P_prev_covar_estimate (TYPE): Description
        Q_rate (TYPE): Description
        R (TYPE): Description
        R_mean (TYPE): Description
        SSE (int): Description
        timing_gain_schedule_R (int): Description
        timing_measure (int): Description
        timing_step (int): Description
        timing_update (int): Description
        torque_profile (TYPE): Description
        x0 (TYPE): Description
        x_prev_state_estimate (TYPE): Description
        x_state_estimate (TYPE): Description
        x_state_update (TYPE): Description
        y_residual (TYPE): Description
        z_measured (TYPE): Description
        z_model (TYPE): Description
    """
    
    def __init__(self, gait_model, torque_profile, measurement_noise_model,CANCEL_RAMP=False,BOOST_BANDWIDTH=False,
                    sigma_q_phase=0,sigma_q_phase_dot=5.1e-4,sigma_q_sL=5e-3,sigma_q_incline=7e-2,DO_GUARDRAILS=False):
        """The Extended Kalman Filter, which linearizes about the current estimated state
        to approximate the nonlinear systems as linear
        
            Args:
                gait_model (class): the model that maps the gait state to measured kinematics
                torque_profile (class): the torque profile the exoskeleton will apply, as a function of gait state
                measurement_noise_model (class): Contains the measurement covariances 
                CANCEL_RAMP (bool, optional): whether to use the ramp state or not
                BOOST_BANDWIDTH (bool, optional): whether to increase the bandwidth of the KF
                sigma_q_phase (float, optional): the process noise for phase (should be noiseless)
                sigma_q_phase_dot (float, optional): the process noise for phase dot
                sigma_q_sL (float, optional): the process noise for stride length
                sigma_q_incline (float, optional): the process noise for incline
        
        Args:
            gait_model (TYPE): Description
            torque_profile (TYPE): Description
            measurement_noise_model (TYPE): Description
            CANCEL_RAMP (bool, optional): Description
            BOOST_BANDWIDTH (bool, optional): Description
            sigma_q_phase (int, optional): Description
            sigma_q_phase_dot (float, optional): Description
            sigma_q_sL (float, optional): Description
            sigma_q_incline (float, optional): Description
        """
        # Initialize state vector and covariance
        # State vector contains, in order: phase, phase rate, stride length, incline
        self.x0 = np.array([[0],[1],[invArctanMap(1.3)],[0]])
        self.P0 = 1e-3 * np.eye(4) #empirically arrived

        if CANCEL_RAMP: 
            self.P0[2,2] = 1e-20
            self.P0[3,3] = 1e-20
        
        # print(self.x0)
        # Initialize state transition matrix
        self.F0 = np.array([[1, 1/180,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # print(self.F0)

        if CANCEL_RAMP: 
            sigma_q_sL = 1e-20
            sigma_q_incline = 1e-20

        # Q is initialized as a covariance rate, which is scaled by the time step to maintain consistent bandwidth behavior 
        self.Q_rate = np.diag([0,sigma_q_phase_dot**2,sigma_q_sL**2,sigma_q_incline**2])/(1e-2)
        # lambda_psL = -0.031
        # epsilon = 1e-20
        # self.Q_rate = np.zeros((4,4))
        # self.Q_rate[1:3,1:3] = sigma_q_phase_dot**2 * np.array([[1, lambda_psL],[lambda_psL, lambda_psL**2]])
        # self.Q_rate[2,2] += epsilon
        # self.Q_rate[3,3] = sigma_q_incline**2
        # print(self.Q_rate)

        if BOOST_BANDWIDTH: 
            self.Q_rate = self.Q_rate*100
        else:
            self.Q_rate = self.Q_rate

        self.measurement_noise_model = measurement_noise_model
        self.meas_config = self.measurement_noise_model.meas_config
        self.torque_profile = torque_profile
        self.gait_model = gait_model

        self.gain_schedule_R(0)
        self.R_mean = self.measurement_noise_model.calc_R_mean()

        self.x_state_estimate = None
        self.P_covar_estimate = None

        self.F = None

        self.SSE = 0

        #INITIALIZE VARS TO HANDLE LONG TIME STEPS
        self.MAX_TIME_STEP = 0.06
        self.DISTRUST_HEELPOS = False
        self.DISTRUST_HEELPOS_COUNTER = 0

        #INITIALIZE CLAMPING VARIABLES
        self.DO_GUARDRAILS = DO_GUARDRAILS
        if self.DO_GUARDRAILS:
            self.PHASE_RATE_LIMITS = [0, np.inf]
            self.INCLINE_LIMITS = [-20,20]

        #timing internal variables
        self.timing_step = 0
        self.timing_measure = 0
        self.timing_update = 0
        self.timing_gain_schedule_R = 0

    def step(self, i, dt):
        """Step function that encodes the prediction step of the EKF
        Follows standard EKF formulae
        
        Args:
            i: int, current iteration count
            dt: float, the time step
            
        """
        time0 = time()

        first=(i==0)

        if first:
            # print('1st step')
            F = np.array([[1, 1/100.0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            Q = self.Q_rate * 1/100.0
            self.x_state_estimate = F @ self.x0
            self.P_covar_estimate = (F @ self.P0 @ F.transpose()) + Q

           
        else:
            F = np.array([[1, dt,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            Q = self.Q_rate * dt
            self.x_state_estimate = F @ self.x_prev_state_estimate
            self.P_covar_estimate = (F @ self.P_prev_covar_estimate @ F.transpose()) + Q

        self.x_state_estimate[0] = self.x_state_estimate[0].item(0) % 1
        if self.DO_GUARDRAILS:
            self.x_state_estimate[1,0] = np.clip(self.x_state_estimate[1,0], self.PHASE_RATE_LIMITS[0], self.PHASE_RATE_LIMITS[1])
            self.x_state_estimate[3,0] = np.clip(self.x_state_estimate[3,0], self.INCLINE_LIMITS[0], self.INCLINE_LIMITS[1])
            # print('GUARDRAILS STEP')

        time1 = time()
        self.timing_step = time1 - time0

    # Measurement function that conducts the measurement step of the EKF

    def update(self, i, dt, data):
        """Summary
        
        Args:
            i (int): current iteration count
            data (np vector (N,)): the measured kinematics
        """
        time0 = time()
        self.z_measured = data.reshape(-1,1)

        # Extract state vector elements
        phase_estimate = self.x_state_estimate[0].item(0)
        phase_dot_estimate = self.x_state_estimate[1].item(0)
        pseudoStrideLength_estimate = self.x_state_estimate[2].item(0)
        strideLength_estimate = arctanMap(pseudoStrideLength_estimate)
        

        # strideLength_estimate = strideLength_estimate
        # if strideLength_estimate >= 1.6 and DO_SATURATE:
        #     strideLength_estimate = np.min([strideLength_estimate, 1.6])
        #     SAT_FLAG = 0

        incline_estimate = self.x_state_estimate[3].item(0)

        # Compute the modeled foot and shank angles based on the regressed gait_model
        footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate,strideLength_estimate,incline_estimate)
        shankAngle_estimate = self.gait_model.returnShankAngle(phase_estimate,strideLength_estimate,incline_estimate)

        #estimate modeled velocity by multiplying the phase rate by the partials wrt phase
        evalfootAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,strideLength_estimate,incline_estimate)
        evalshankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,strideLength_estimate,incline_estimate)

        footAngleVel_estimate = phase_dot_estimate * evalfootAngleDeriv_dphase
        shankAngleVel_estimate = phase_dot_estimate * evalshankAngleDeriv_dphase
        self.z_model = [footAngle_estimate, footAngleVel_estimate, shankAngle_estimate, shankAngleVel_estimate]

        if self.meas_config == 'full' or self.meas_config == 'heelForward':
            heelForwardPos_estimate = self.gait_model.returnHeelPosForward(phase_estimate,strideLength_estimate,incline_estimate)
            self.z_model.append(heelForwardPos_estimate)

        if self.meas_config == 'full' or self.meas_config == 'heelUp':
            heelUpPos_estimate = self.gait_model.returnHeelPosUp(phase_estimate,strideLength_estimate,incline_estimate)
            self.z_model.append(heelUpPos_estimate)

        self.z_model = np.array(self.z_model).reshape(-1,1)
        self.y_residual = self.z_measured - self.z_model

  
        self.SSE = (self.y_residual.T @ np.linalg.solve(self.R_mean, self.y_residual) ) + self.SSE
    



        dsLdPsL = dArctanMap(pseudoStrideLength_estimate)

        # calculate the H matrix for the phase estimator
        # H is the Jacobian of the measurements wrt the state vector

        dh11 = evalfootAngleDeriv_dphase
        dh12 = 0
        dh13 = dsLdPsL * self.gait_model.returnFootAngleDeriv_dsL(phase_estimate,strideLength_estimate,incline_estimate)
        dh14 = self.gait_model.returnFootAngleDeriv_dincline(phase_estimate,strideLength_estimate,incline_estimate)

        dh21 = phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphase2(phase_estimate,strideLength_estimate,incline_estimate)
        dh22 = evalfootAngleDeriv_dphase
        dh23 = dsLdPsL * phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphasedsL(phase_estimate,strideLength_estimate,incline_estimate)
        dh24 = phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphasedincline(phase_estimate,strideLength_estimate,incline_estimate)

        dh31 = evalshankAngleDeriv_dphase
        dh32 = 0
        dh33 = dsLdPsL * self.gait_model.returnShankAngleDeriv_dsL(phase_estimate,strideLength_estimate,incline_estimate)
        dh34 = self.gait_model.returnShankAngleDeriv_dincline(phase_estimate,strideLength_estimate,incline_estimate)

        dh41 = phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphase2(phase_estimate,strideLength_estimate,incline_estimate)
        dh42 = evalshankAngleDeriv_dphase
        dh43 = dsLdPsL * phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphasedsL(phase_estimate,strideLength_estimate,incline_estimate)
        dh44 = phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphasedincline(phase_estimate,strideLength_estimate,incline_estimate)

        H = np.array([[dh11, dh12,dh13,dh14],
            [dh21, dh22,dh23,dh24],
            [dh31, dh32,dh33,dh34],
            [dh41, dh42,dh43,dh44],
            ])

        if self.meas_config == 'full' or self.meas_config == 'heelForward':
            #heel accelrow
            evalHeelForwardPosDeriv_dphase = self.gait_model.returnHeelPosForwardDeriv_dphase(phase_estimate,strideLength_estimate,incline_estimate)

            dh51 = evalHeelForwardPosDeriv_dphase
            dh52 = 0
            dh53 = dsLdPsL * self.gait_model.returnHeelPosForwardDeriv_dsL(phase_estimate,strideLength_estimate,incline_estimate)
            dh54 = self.gait_model.returnHeelPosForwardDeriv_dincline(phase_estimate,strideLength_estimate,incline_estimate)

            H = np.vstack((H, np.array([[dh51, dh52,dh53,dh54]])))

        if self.meas_config == 'full' or self.meas_config == 'heelUp':
            #tibia accelrow
            evalTibiaForwardPosDeriv_dphase = self.gait_model.returnHeelPosUpDeriv_dphase(phase_estimate,strideLength_estimate,incline_estimate)

            dh61 = evalTibiaForwardPosDeriv_dphase
            dh62 = 0
            dh63 = dsLdPsL * self.gait_model.returnHeelPosUpDeriv_dsL(phase_estimate,strideLength_estimate,incline_estimate)
            dh64 = self.gait_model.returnHeelPosUpDeriv_dincline(phase_estimate,strideLength_estimate,incline_estimate)
            H = np.vstack((H, np.array([[dh61, dh62,dh63,dh64]])))


        #update the measurement covariance
        self.gain_schedule_R(phase_estimate)

        #HANDLE LONG TIME STEPS
        R_eff = self.R
        # print(f'dt: {dt}')
        if dt >= self.MAX_TIME_STEP and i > 0:
            print('EXCEEDED MAX TIME STEP')
            print(f'dt: {dt}')
            self.DISTRUST_HEELPOS = True
            self.P_covar_estimate = (1e-3 * np.eye(4))

        if self.DISTRUST_HEELPOS:
            R_eff[4,4] = 1
            R_eff[5,5] = 1
            self.DISTRUST_HEELPOS_COUNTER += 1

            if self.DISTRUST_HEELPOS_COUNTER > 70:
                self.DISTRUST_HEELPOS_COUNTER = 0
                self.DISTRUST_HEELPOS = False


        S_covariance = H @ self.P_covar_estimate @ H.transpose() + R_eff

        # Compute Kalman Gain
        K_gain = self.P_covar_estimate @ H.transpose() @ np.linalg.inv(S_covariance)

        self.x_state_update = self.x_state_estimate + K_gain @ self.y_residual

        # Modulo phase to be between 0 and 1
        self.x_state_update[0] = self.x_state_update[0].item(0) % 1
        if self.DO_GUARDRAILS:
            self.x_state_update[1,0] = np.clip(self.x_state_update[1,0], self.PHASE_RATE_LIMITS[0], self.PHASE_RATE_LIMITS[1])
            self.x_state_update[3,0] = np.clip(self.x_state_update[3,0], self.INCLINE_LIMITS[0], self.INCLINE_LIMITS[1])
            # print('GUARDRAILS UPDATE')

        # Update covariance
        self.P_covar_update = (np.eye(4) - K_gain @ H) @ self.P_covar_estimate

        self.x_prev_state_estimate = self.x_state_update
        self.P_prev_covar_estimate = self.P_covar_update
        time1 = time()
        self.timing_update = time1 - time0


    def get_torque(self):
        """Returns exoskeleton torque given the current state vector
        
        Returns:
            float: the desired torque
        """
        phase_estimate = self.x_state_update[0].item(0)
        phase_dot_estimate = self.x_state_update[1].item(0)
        pseudoStrideLength_estimate = self.x_state_update[2].item(0)
        strideLength_estimate = arctanMap(pseudoStrideLength_estimate)
        incline_estimate = self.x_state_update[3].item(0)
        
        return self.torque_profile.evalTorqueProfile(phase_estimate,strideLength_estimate,incline_estimate)


    def gain_schedule_R(self,phase_estimate):
        """Updates the measurement covariance matrix R
        
        Args:
            phase_estimate (float): the phase value at which to evaluate R
        """
        self.R = self.measurement_noise_model.gain_schedule_R(phase_estimate)
        return self.R



    def set_SSE(self,newSSE):
        """Setter method that sets SSE
        
        Args:
            newSSE (float): the new SSE to set
        """
        self.SSE = newSSE

    def getSSE(self,):
        """Getter method that returns SSE
        
        Returns:
            float: SSE
        """
        return self.SSE

    def get_z_measured(self,):
        """Getter method that returns z_measured
        
        Returns:
            np matrix: z_measured
        """
        return self.z_measured

    def set_y_residual(self,y_residual_new):
        """Setter method that sets y_residual
        
        Args:
            y_residual_new (np vector): the new y_residual
        """
        self.y_residual = y_residual_new

    def set_prev_x_state_estimate(self,x_prev_state_estimate_new):
        """Setter method that sets x_prev_state_estimate
        
        Args:
            x_prev_state_estimate_new (np vector): the new x_prev_state_estimate
        """
        self.x_prev_state_estimate[0, 0] = x_prev_state_estimate_new[0]
        self.x_prev_state_estimate[1, 0] = x_prev_state_estimate_new[1]
        self.x_prev_state_estimate[2, 0] = x_prev_state_estimate_new[2]
        self.x_prev_state_estimate[3, 0] = x_prev_state_estimate_new[3]

    def set_prev_P_covar_estimate(self,P_prev_covar_estimate_new):
        """Setter method that sets P_prev_covar_estimate
        
        Args:
            P_prev_covar_estimate_new (np matrix): The new P_prev_covar_estimate
        """
        self.P_prev_covar_estimate = P_prev_covar_estimate_new

    def set_x_state_estimate(self,x_state_estimate_new):
        """Setter method that sets x_state_estimate
        
        Args:
            x_state_estimate_new (np vector): the new x_state_estimate
        """
        self.x_state_estimate[0, 0] = x_state_estimate_new[0]
        self.x_state_estimate[1, 0] = x_state_estimate_new[1]
        self.x_state_estimate[2, 0] = x_state_estimate_new[2]
        self.x_state_estimate[3, 0] = x_state_estimate_new[3]

    def get_x_state_estimate(self,):
        """Getter method that returns x_state_estimate
        
        Returns:
            np matrix: x_state_estimate
        """
        return self.x_state_estimate

    def set_P_covar_estimate(self,P_covar_estimate_new):
        """Setter method that sets P_covar_estimate
        
        Args:
            P_covar_estimate_new (np matrix): The new P_covar_estimate
        """
        self.P_covar_estimate = P_covar_estimate_new
        


    def get_x_state_update(self,):
        """Getter method that returns x_state_update
        
        Returns:
            np matrix: x_state_update
        """
        return self.x_state_update

    def set_x_state_update(self,x_state_update_new):
        """Setter method that sets x_state_update
        
        Args:
            x_state_update_new (np vector): the new x_state_update
        """

        self.x_state_update[0, 0] = x_state_update_new[0]
        self.x_state_update[1, 0] = x_state_update_new[1]
        self.x_state_update[2, 0] = x_state_update_new[2]
        self.x_state_update[3, 0] = x_state_update_new[3]

    def set_P_covar_update(self,P_covar_update_new):
        """Setter method that sets P_covar_update
        
        Args:
            P_covar_update_new (np matrix): The new P_covar_update
        """
        self.P_covar_update = P_covar_update_new



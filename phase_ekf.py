"""Made by Leo. Contains the full EKF for phase
"""
from time import time
import numpy as np
from training_utils import phase_dist
from training_utils import unscale_gait_state, unscale_kinematics, normalize_kinematics, unscale_kinematics_gradient, scale_gait_state, phase_to_trig, trig_to_phase
from training_utils import generate_unit_vector


class PhaseEKF():
    """This class encodes an Extended Kalman Filter (EKF) that estimates the gait state 
    using measurements of kinematics.
    The gait state is encoded as: [phase, speed, incline, is_stairs]
    The kinematics expected are: [foot angle, 
        foot angle velocity, 
        shank angle, 
        shank angle velocity, 
        heel forward acceleration, 
        heel upward acceleration] 

    The neural network models/computations are passed in as arguments rather than being encapsulated 
    in the EKF. This is because the EKF is designed to run real-time on a Raspberry Pi, which cannot run 
    the neural networks fast enough. The networks run on a separate computer (e.g. Jetson Nano) and their
    computed quantities are sent to this class.
    """    
    
    def __init__(self,sigma_q_phase=0,
                 sigma_q_speed=5e-3,
                 sigma_q_incline=7e-2,
                 sigma_q_is_stairs=7e-2,
                 R=np.eye(5),
                 m_dist_importance=1,
                 speed_scale=None,
                 incline_scale=None,
                 stair_height_scale=None,
                 meas_scale=None,
                 DO_GAIT_MODEL_IN_EKF=False,
                 DO_HETEROSCEDASTIC=False,
                 CANCEL_STAIRS=False):
        """

        Args:
            sigma_q_phase (float, optional): the standard deviation of the phase process noise. Defaults to 0.
            sigma_q_speed (float, optional): the standard deviation of the speed process noise. Defaults to 5e-3.
            sigma_q_incline (float, optional): the standard deviation of the incline process noise. Defaults to 7e-2.
            sigma_q_is_stairs (float, optional): the standard deviation of the is_stairs process noise. Defaults to 7e-2.
            R (2D np array, optional): The measuremnt noise matrix. Defaults to np.eye(5).
            m_dist_importance (float, optional): the relative importance of the Mahalanobis distance, used to scale the trust in the neural network predictions. Defaults to 1.
            speed_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the speed
            incline_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the incline
            stair_height_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the stair heights
            meas_scale (2D np array, optional): an Nx2 array of the kinematics scaling factors, where N is the number of measurements
            DO_GAIT_MODEL_IN_EKF (bool, optional): whether to use the continuous gait model in the EKF. Defaults to False.
            DO_HETEROSCEDASTIC (bool, optional): whether to use the heteroscedastic model. Defaults to False.
            CANCEL_STAIRS (bool, optional): whether to estimate the stairs state. Defaults to False.
        """                 
        
        # Initialize state vector and covariance
        # State vector contains, in order: cos_phase, sin_phase, speed, incline, stair_height
        self.x0 = np.array([[-1],[0],[0.0],[0],[0]])
        self.P0 = 1e-2 * np.eye(5) #empirically arrived
        self.n_states = 5 #number of states (including phase trig)

        # print(self.x0)
        # Initialize state transition matrix
        self.F0 = np.eye(self.n_states)

        # Q is initialized as a covariance rate, which is scaled by the time step to maintain consistent bandwidth behavior 
        self.Q_rate = np.diag([sigma_q_phase**2,sigma_q_phase**2,sigma_q_speed**2,
                               sigma_q_incline**2,sigma_q_is_stairs**2])
    
        
            
        self.phase_prev = 0
        self.x_state_estimate = None
        self.P_covar_estimate = None
        
        #append additional noise term on R to account for phase trig
        self.R = np.block([
                    [R[0,0],               np.zeros((1,R.shape[0]))],
                    [np.zeros((R.shape[0], 1)), R]
                ])
        self.R_eff = self.R.copy()
        
        self.m_dist_importance = m_dist_importance
        self.DO_GAIT_MODEL_IN_EKF = DO_GAIT_MODEL_IN_EKF
        self.DO_HETEROSCEDASTIC = DO_HETEROSCEDASTIC
        self.CANCEL_STAIRS = CANCEL_STAIRS
        self.speed_scale=speed_scale
        self.incline_scale=incline_scale
        self.stair_height_scale=stair_height_scale
        self.meas_scale=meas_scale
        
        if self.CANCEL_STAIRS:
            self.P0[4,4] = 1e-30
            self.Q_rate[4,4] = 1e-30
          
        #timing internal variables
        self.timing_step = 0
        self.timing_measure = 0
        self.timing_update = 0
        self.timing_gain_schedule_R = 0
            
    def _convert_state_trig_to_phase(self,x_internal):
        """This function converts the gait state with trig-encoded phase to regular phase.
        Trig-encoded phase is used internally in the EKF, while the regular phase is designed to 
        be used and called externally outside the EKF
        """        
        x_ext = np.zeros((x_internal.shape[0]-1, 1))
        x_ext[0,0] = trig_to_phase(x_internal[0,0], x_internal[1,0])
        x_ext[1:,0] = x_internal[2:,0]
        return x_ext
    
    def _convert_state_phase_to_trig(self,x_ext):
        """This function converts the gait state with regular phase to trig encoded phasee
        """   
        x_int = np.zeros((x_ext.shape[0]+1, 1))
        phase = x_ext[0,0]
        cp, sp = phase_to_trig(phase)
        x_int[0,0] = cp
        x_int[1,0] = sp
        x_int[2:] = x_ext[1:]
        return x_int
    
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
            F = self.F0
            Q = self.Q_rate * 1/100.0
            self.x_state_trig = F @ self.x0            
            self.P_covar = (F @ self.P0 @ F.transpose()) + Q

           
        else:
            #estimate phase rate using a scaled version of speed
            pdot = self.x_state[1,0]*0.8
            
            #encode the state transition matrix, 
            # state is modified by integrating the phase rate by the time step,
            # which is encoded by a rotation matrix that rotates the trig-encoded phase by an amount 
            # equal to pdot x dt
            #the other states are assumed to stay constant
            F = np.array([
                [np.cos(2*np.pi*dt*pdot), -np.sin(2*np.pi*dt*pdot),0,0,0],
                [np.sin(2*np.pi*dt*pdot),np.cos(2*np.pi*dt*pdot),0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1]])
            
            Q = self.Q_rate * dt #generate the process noise matrix
            self.x_state_trig = F @ self.x_state_trig# 
            self.P_covar = (F @ self.P_covar @ F.transpose()) + Q
        
        
        
        #clamp speed
        self.x_state_trig[2,0] = np.clip(self.x_state_trig[2,0], 0, 2)
        
        #clamp incline
        self.x_state_trig[3,0] = np.clip(self.x_state_trig[3,0], -55, 55)
        
        #clamp stair height            
        self.x_state_trig[4,0] = np.clip(self.x_state_trig[4,0], -1, 1)
        
        
        #convert to non-trig state 
        self.x_state = self._convert_state_trig_to_phase(self.x_state_trig)
        
        # Modulo phase to be between 0 and 1
        self.x_state[0,0] = self.x_state[0] % 1
        
        #store the estimate from the prediction step for logging
        self.x_state_estimate = self.x_state.copy()
        
        time1 = time()
        self.timing_step = time1 - time0

    def update(self, i, dt, z_measured, z_model_kinematics=None, kinematics_gradient=None, m_dist=None, R_heteroscedastic=None):
        """Measurement function that conducts the measurement step of the EKF
        This function can take in as measurements the gait states as predicted by the gait transformer, and 
        the kinematic measurements from the exoskeleton

        Args:
            i (int): current iteration count
            dt (float): the delta time step from the previous iteration
            z_measured (np array): the measurements to the EKF (gait states + kinematics)
            z_model_kinematics (np array, optional): The predicted kinematics from the gait model. Defaults to None.
            kinematics_gradient (2D np array): the Jacobian of kinematics wrt the gait state. Defaults to None.
            m_dist (float, optional): the Mahalanobis distance of the predicted kinematics to the measured kinematics. Defaults to None.
            R_heteroscedastic (2D np array, optional): the covariance matrix from the heteroscedastic network. Defaults to None.
        """        
        time0 = time()

        self.z_measured = z_measured.reshape(-1,1)  

        #convert the first five elements of z_measured (which are the regular state measurments) to trig
        self.z_measured_trig = np.vstack((self._convert_state_phase_to_trig(self.z_measured[:4]),self.z_measured[4:]))

        #initialize the measurement matrix, which for the first five elements of z_measured, are the states 
        # themselves, so H is the identity matrix
        H = np.eye(self.n_states)
                
        
        if self.DO_GAIT_MODEL_IN_EKF:
            self.z_model_kinematics = z_model_kinematics.reshape(-1,1)
            H = np.vstack((H, kinematics_gradient)) #append kinematics to measured gait state
            # print(H.shape)
            
            #augment model vector
            self.z_model = np.vstack((self.x_state_trig, self.z_model_kinematics))
        else:
            self.z_model = self.x_state_trig
                    
        self.y_residual = self.z_measured_trig - self.z_model
        
        #compute effective R based on the passed mahalanobis distance
        #we also multiply by the m_distance value at the 84th percentile (1 SD away) to normalize
        
        #need to copy the matrix to prevent R_eff from shadowing self.R, which would otherwise
        # cause R_eff to increased exponential by the scaling_factors
        self.R_eff = self.R.copy() 

        if m_dist:
            #compute trust scaling factor using a power law based on the importance of the mahalnobis distance
            scaling_factor = np.power(self.m_dist_importance, ( (m_dist/3.15) - 1) ) + 1e-1

            #scale the elements of R corresponding to the gait state vector
            self.R_eff[0:self.n_states,0:self.n_states] = self.R_eff[0:self.n_states,0:self.n_states] * scaling_factor
            
        if self.DO_HETEROSCEDASTIC:
            # print(R_heteroscedastic)
            # input()
            self.R_eff[self.n_states:11,self.n_states:11] = self.R_eff[self.n_states:11,self.n_states:11] + R_heteroscedastic
            
        # print(R_eff.shape)
        S_covariance = H @ self.P_covar @ H.transpose() + self.R_eff

        # Compute Kalman Gain
        K_gain = self.P_covar @ H.transpose() @ np.linalg.inv(S_covariance)

        #update state
        self.x_state_trig = self.x_state_trig + K_gain @ self.y_residual

        
        #clamp speed
        self.x_state_trig[2,0] = np.clip(self.x_state_trig[2,0], 0, 2)
        
        #clamp incline
        self.x_state_trig[3,0] = np.clip(self.x_state_trig[3,0], -55, 55)
        
        #clamp stair height
        self.x_state_trig[4,0] = np.clip(self.x_state_trig[4,0], -1, 1)
                
        #convert to non-trig state 
        self.x_state = self._convert_state_trig_to_phase(self.x_state_trig)
        
        # Modulo phase to be between 0 and 1
        self.x_state[0,0] = self.x_state[0] % 1
        
        # Update covariance
        self.P_covar = (np.eye(5) - K_gain @ H) @ self.P_covar

        time1 = time()
        self.timing_update = time1 - time0


class StairsOnlyEKF():
    
    def __init__(self,
                 sigma_q_is_stairs=7e-2,
                 R=np.eye(5),
                 m_dist_importance=1,
                 speed_scale=None,
                 incline_scale=None,
                 stair_height_scale=None,
                 meas_scale=None,
                 DO_HETEROSCEDASTIC=None):
        """A modified EKF that only estimates stairs
        """
        # Initialize state vector and covariance
        # State vector contains, in order: phase, phase rate, stride length, incline
        self.x0 = np.array([[0]])
        self.P0 = 1e-3 * np.eye(1) #empirically arrived

        # print(self.x0)
        # Initialize state transition matrix
        self.F0 = np.array([[1]])
        self.n_states = 1

        # Q is initialized as a covariance rate, which is scaled by the time step to maintain consistent bandwidth behavior 
        self.Q_rate = np.diag([sigma_q_is_stairs**2])
        
        
        #append additional noise term on R to account for phase trig
        self.R = R
        self.R_eff = R
        self.m_dist_importance = m_dist_importance
        self.speed_scale=speed_scale
        self.incline_scale=incline_scale
        self.stair_height_scale=stair_height_scale
        self.meas_scale=meas_scale
        
        self.F = None
        self.DO_HETEROSCEDASTIC = DO_HETEROSCEDASTIC
        
        #timing internal variables
        self.timing_step = 0
        self.timing_measure = 0
        self.timing_update = 0
        self.timing_gain_schedule_R = 0

    def step(self, i, dt, x_state_estimate_from_full):
        """Step function that encodes the prediction step of the EKF
        Follows standard EKF formulae
        
        Args:
            i: int, current iteration count
            dt: float, the time step
            x_state_estimate_from_full: 
            
        """
        time0 = time()

        first=(i==0)
        
        if first:
            # print('1st step')
            F = np.array([[1]])
            Q = self.Q_rate * 1/100.0
            self.x_state = F @ self.x0
            self.P_covar = (F @ self.P0 @ F.transpose()) + Q

           
        else:
            F = np.array([[1]])
            Q = self.Q_rate * dt
            self.x_state = F @ self.x_state#
            self.P_covar = (F @ self.P_covar @ F.transpose()) + Q

        #clamp stair height
        self.x_state[0] = np.clip(self.x_state[0,0], -1, 1)
        
        phase_from_full = x_state_estimate_from_full[0,0]
        speed_from_full = x_state_estimate_from_full[1,0]
        stair_height = self.x_state[0,0]
        
        self.x_state_estimate_effective = np.array([[phase_from_full],[speed_from_full],[0],[stair_height]])
        
        time1 = time()
        self.timing_step = time1 - time0

    # Measurement function that conducts the measurement step of the EKF
    def update(self, i, dt, z_measured, z_model_kinematics=None, kinematics_gradient=None, R_heteroscedastic=None, m_dist=None):
        """Summary
        
        Args:
            i (int): current iteration count
            data (np vector (N,)): the measured kinematics
        """
        time0 = time()
        self.z_measured = z_measured.reshape(-1,1)        
        # print('self.z_measured')
        # print(self.z_measured)
        
                
        self.z_model_kinematics = z_model_kinematics
        stair_height = self.x_state[0,0]
        self.z_model = np.vstack((stair_height, self.z_model_kinematics))
                                
        #extract column of numerical gradient corresponding to stairs
        H = np.vstack(([[1]], kinematics_gradient))
        # print(H.shape)

        
        # print('self.z_model')
        # print(self.z_model)
            
        self.y_residual = self.z_measured - self.z_model
        
        # print('self.y_residual')
        # print(self.y_residual)
        # input()
        
        #compute effective R based on the passed mahalanobis distance
        #we also multiply by the m_distance value at the 84th (1 SD) percentile to normalize
        self.R_eff = self.R.copy()
        if m_dist:
            # R_eff = self.R * (self.m_dist_importance * (m_dist/6.84)**2 + 1)
            self.R_eff[0:self.n_states,0:self.n_states] = self.R_eff[0:self.n_states,0:self.n_states] * (self.m_dist_importance * (m_dist/3.15)**2 + 1)
        
        if self.DO_HETEROSCEDASTIC:
            # print(R_heteroscedastic)
            # input()
            self.R_eff[self.n_states:7,self.n_states:7] = self.R_eff[self.n_states:7,self.n_states:7] + R_heteroscedastic
            
        # print(R_eff.shape)
        
        S_covariance = H @ self.P_covar @ H.transpose() + self.R_eff

        # Compute Kalman Gain
        K_gain = self.P_covar @ H.transpose() @ np.linalg.inv(S_covariance)
        self.x_state = self.x_state + K_gain @ self.y_residual
        
        #clamp stair height
        self.x_state[0,0] = np.clip(self.x_state[0,0], -1, 0)
        # Update covariance
        self.P_covar = (np.eye(1) - K_gain @ H) @ self.P_covar

        time1 = time()
        self.timing_update = time1 - time0
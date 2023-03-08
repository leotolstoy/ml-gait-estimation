"""This file contains various utilities that don't depend on pytorch for training and evaluation

"""
from copy import deepcopy
import numpy as np

    
def phase_dist(phase_a, phase_b):
    """computes a distance that accounts for the modular arithmetic of phase
    and guarantees that the output is between 0 and .5
    
    Args:
        phase_a (float): a phase between 0 and 1
        phase_b (float): a phase between 0 and 1
    
    Returns:
        dist_prime: the difference between the phases, modulo'd between 0 and 0.5
    """
    if isinstance(phase_a, np.ndarray):
        dist_prime = (phase_a-phase_b)
        dist_prime[dist_prime > 0.5] = 1-dist_prime[dist_prime > 0.5]

        dist_prime[dist_prime < -0.5] = -1-dist_prime[dist_prime < -0.5]

    else:
        dist_prime = (phase_a-phase_b)
        if dist_prime > 0.5:
            dist_prime = 1-dist_prime

        elif dist_prime < -0.5:
            dist_prime = -1-dist_prime
    return dist_prime

def generate_unit_vector(dim=3, coordinate=0):
    """computes a unit vector
    
    Args:
        dim (int): the dimension of the unit vector
        coordinate (int): the direction of the unit vector
    
    Returns:
        e_vec: the unit vector
    """
    
    e_vec = np.zeros((dim,))
    e_vec[coordinate] = 1
    return e_vec
 
def scale_gait_state(gait_states_raw, speed_scale, incline_scale, stair_height_scale):
    """Scale a gait state input to the range (0,1) by the provided scales. Also convert the phase 
    gait variable to the trig encoding. Gait state must be in order: [phase, speed, incline, stairs]

    Args:
        gait_states_raw (np array, 4x2): the unscaled gait states
        speed_scale (tuple): the range of the speeds to scale
        incline_scale (tuple): the range of the inclines to scale
        stair_height_scale (tuple): the range of the stair heights to scale

    Returns:
        5x2 np array: scaled gait states
    """    
    rows, cols = gait_states_raw.shape
    gait_states_scaled = np.zeros((rows,cols+1)) #preallocate
    
    #scale speed
    speed_lb = speed_scale[0]
    speed_ub = speed_scale[1]
    gait_states_scaled[:,2] = ((1 - 0)/(speed_ub - speed_lb)) * (gait_states_raw[:,1] - speed_lb)
    
    #scale incline
    incline_lb = incline_scale[0]
    incline_ub = incline_scale[1]
    gait_states_scaled[:,3] = ((1 - 0)/(incline_ub - incline_lb)) * (gait_states_raw[:,2] - incline_lb)
    
    #scale stairs
    stair_height_lb = stair_height_scale[0]
    stair_height_ub = stair_height_scale[1]
    gait_states_scaled[:,4] = ((1 - 0)/(stair_height_ub - stair_height_lb)) * (gait_states_raw[:,3] - stair_height_lb)
    
        
    #project phase into sine and cosine encoding
    phase_as_angle = 2*np.pi*(gait_states_raw[:,0]-0.5)
    cp = np.cos(phase_as_angle)
    sp = np.sin(phase_as_angle)
    gait_states_scaled[:,0] = cp
    gait_states_scaled[:,1] = sp
    
    return gait_states_scaled

def unscale_gait_state(gait_state_vec, speed_scale, incline_scale, stair_height_scale):
    """This script descales a gait state vector from (0,1) range to a human interpretable/physically
    meaningful range. Additionally, it converts phase trig encoding to phase between 0 and 1. 
    Gait state must be in order: [cos_phase, sin_phase, speed, incline, stairs]

    Args:
        gait_state_vec (5x2): The scaled gait states
        speed_scale (tuple): the range of the speeds to scale
        incline_scale (tuple): the range of the inclines to scale
        stair_height_scale (tuple): the range of the stair heights to scale

    Returns:
        4x2 np array: _description_
    """    
    rows, cols = gait_state_vec.shape
    # print(gait_state_vec.shape)
    gait_state_unscaled = np.zeros((rows,cols-1))
    
    cp = gait_state_vec[:,0]
    sp = gait_state_vec[:,1]
    
    #undo the trig on phase
    x = np.arctan2(sp,cp)
    phase_p = ((x)/(2*np.pi)) + 0.5
    
    gait_state_unscaled[:,0] = phase_p
    
    #unscale speed
    speed_lb = speed_scale[0]
    speed_ub = speed_scale[1]
    speed_unscaled = (gait_state_vec[:,2] * (speed_ub - speed_lb)) + speed_lb
    gait_state_unscaled[:,1] = speed_unscaled
    
    #unscale incline
    incline_lb = incline_scale[0]
    incline_ub = incline_scale[1]
    incline_unscaled = (gait_state_vec[:,3] * (incline_ub - incline_lb)) + incline_lb
    gait_state_unscaled[:,2] = incline_unscaled
    
    #unscale stair height
    stair_height_lb = stair_height_scale[0]
    stair_height_ub = stair_height_scale[1]
    stair_height_unscaled = (gait_state_vec[:,4] * (stair_height_ub - stair_height_lb)) + stair_height_lb
    gait_state_unscaled[:,3] = stair_height_unscaled
    
    
    return gait_state_unscaled


def normalize_kinematics(kinematics_vector, meas_scale):
    """This convenience function scales measurements of kinematics to the range (0,1)

    Args:
        kinematics_vector (2D np.array, dim: (N,7)): the kinematics measurement vector
            contains, in order: 
                foot angle,
                foot angle velocity,
                shank angle, 
                shank angle velocity,
                forward heel acceleration
                upward heel acceleration,
                time step difference
        meas_scale (2D np.array, dim: (7,2)): the ordered bounds to normalize the kinematics by

    Returns:
        2-D np.array, dim: (N,7): the normalized kinematics
    """    
    kinematics_vector_norm = np.zeros(kinematics_vector.shape)
    N_samples, n_meas = kinematics_vector.shape
    for i in range(n_meas):
        lb = meas_scale[i,0]
        ub = meas_scale[i,1]
        kinematics_vector_norm[:,i] = ((1 - 0)/(ub - lb)) * (kinematics_vector[:,i] - lb)
    return kinematics_vector_norm


def unscale_kinematics(scaled_kinematics_vec, scales):
    """This function unscales a scaled kinematics array (0,1) to a human interpretable range. 
    The kinematics measurement vector
            contains, in order: 
                foot angle,
                foot angle velocity,
                shank angle, 
                shank angle velocity,
                forward heel acceleration
                upward heel acceleration,
                time step difference

    Args:
        scaled_kinematics_vec (np.array,Nx7): the normalized kinematics. Contains N samples of kinematics
        scales (2D np array, 7x2): the ordered bounds to denormalize the kinematics by

    Returns:
        2-D np.array, dim: (N,7): the denormalized kinematics
    """    
    rows, cols = scaled_kinematics_vec.shape
    unscaled_kinematics_vec = np.zeros((rows,cols))
    
    for i in range(cols):
        scale_lb = scales[i,0]
        scale_ub = scales[i,1]
        unscaled_kinematics_vec[:,i] = (scaled_kinematics_vec[:,i] * (scale_ub - scale_lb)) + scale_lb
    
    return unscaled_kinematics_vec


def unscale_kinematics_gradient(scaled_kinematics_grad, phase, meas_scale, speed_scale, incline_scale, stair_height_scale):
    """This function unscales a kinematics gradient/Jacobian that contains the partial derivatives
        of scaled kinematics and scaled gait states

    Args:
        scaled_kinematics_grad (np array): the scaled jacobian
        phase (float): the phase at the gradient evaluation point
        meas_scale (2D np.array, dim: (7,2)): the ordered bounds to normalize the kinematics by
        speed_scale (tuple): the range of the speeds to scale
        incline_scale (tuple): the range of the inclines to scale
        stair_height_scale (tuple): the range of the stair heights to scale

    Returns:
        np array: the unscaled Jacobian
    """    
    rows, cols = scaled_kinematics_grad.shape
    unscaled_kinematics_grad = np.zeros((rows,cols-1))

    #calculate gradient wrt the phase in trig encoding
    #requires combining the gradients wrt cp and sp (which are output by the model)
    dcpdp = -2*np.pi*np.sin(2*np.pi*(phase-0.5))
    dspdp = 2*np.pi*np.cos(2*np.pi*(phase-0.5))
    unscaled_kinematics_grad[:,0] = scaled_kinematics_grad[:,0]*dcpdp + scaled_kinematics_grad[:,1]*dspdp
    
    #store the rest of the jacobian
    unscaled_kinematics_grad[:,1:] = scaled_kinematics_grad[:,2:]    
    
    #unscale the gradients by the measurement scales
    for i in range(rows):
        lb = meas_scale[i,0]
        ub = meas_scale[i,1]
        unscaled_kinematics_grad[i,:] = (ub - lb) * unscaled_kinematics_grad[i,:]
    
    #unscale the gradients by the gait state variable states
    
    #speed
    lb = speed_scale[0]
    ub = speed_scale[1]
    unscaled_kinematics_grad[:,1] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,1]
    
    #incline
    lb = incline_scale[0]
    ub = incline_scale[1]
    unscaled_kinematics_grad[:,2] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,2]
    
    #stairs
    lb = stair_height_scale[0]
    ub = stair_height_scale[1]
    unscaled_kinematics_grad[:,3] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,3]
    
    return unscaled_kinematics_grad

def unscale_kinematics_gradient_phase_trig(scaled_kinematics_grad, meas_scale, speed_scale, incline_scale, stair_height_scale):
    """This function unscales a kinematics gradient/Jacobian which has phase encoded in trig form,
        that contains the partial derivatives of scaled kinematics and scaled gait states. 

    Args:
        scaled_kinematics_grad (2D np array): the scaled jacobian
        meas_scale (2D np.array, dim: (7,2)): the ordered bounds to normalize the kinematics by
        speed_scale (tuple): the range of the speeds to scale
        incline_scale (tuple): the range of the inclines to scale
        stair_height_scale (tuple): the range of the stair heights to scale

    Returns:
        np array: the unscaled Jacobian
    """    
    rows, cols = scaled_kinematics_grad.shape
    unscaled_kinematics_grad = scaled_kinematics_grad
    
    #unscale the gradients by the measurement scales
    for i in range(rows):
        lb = meas_scale[i,0]
        ub = meas_scale[i,1]
        unscaled_kinematics_grad[i,:] = (ub - lb) * unscaled_kinematics_grad[i,:]
    
    #unscale the gradients by the gait state variable states
    
    #speed
    lb = speed_scale[0]
    ub = speed_scale[1]
    unscaled_kinematics_grad[:,2] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,2]
    
    #incline
    lb = incline_scale[0]
    ub = incline_scale[1]
    unscaled_kinematics_grad[:,3] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,3]
    
    #stairs
    lb = stair_height_scale[0]
    ub = stair_height_scale[1]
    unscaled_kinematics_grad[:,4] =  ((1 - 0)/(ub - lb)) * unscaled_kinematics_grad[:,4]
    
    return unscaled_kinematics_grad

    
def phase_to_trig(phase):
    """utility function that returns the trig encoding of a phase value

    Args:
        phase (float): phase value between 0 and 1

    Returns:
        tuple: cos and sin of the phase value
    """    
    phase_as_angle = 2*np.pi*(phase-0.5)
    cp = np.cos(phase_as_angle)
    sp = np.sin(phase_as_angle)
    
    return (cp, sp)

def trig_to_phase(cp, sp):
    """utility function that undoes the trig encoding of a phase value

    Args:
        cp (float): cosine of a phase value
        sp (float): sine of a phase value

    Returns:
        float: phase
    """    
    #undo the trig on phase
    x = np.arctan2(sp,cp)
    phase_p = ((x)/(2*np.pi)) + 0.5
    return phase_p

def convert_cov_mat_to_vector(cov_mat):
    """Converts a square covariance matrix to a vector containing its unique entries.
    The unique entries are the entries to the top right and including the primary diagonal of the matrix

    Args:
        cov_mat (square np array): the covariance matrix

    Returns:
        1D np array: the unique elements of cov_mat
    """    
    #handle batched inputs in the first dimension
    if len(cov_mat.shape) == 3:
        B = cov_mat.shape[0]
        N = cov_mat.shape[1]
    else:
        N = cov_mat.shape[0]

    #set up the length of the flattened vector of the unique variances
    L = np.sum([i+1 for i in range(N)])
    
    if len(cov_mat.shape) == 3: #if batched
        flattened_output = np.zeros((B,L))
        for jj in range(B):
            start_idx = 0
            for i in range(0,N):
                row = cov_mat[jj,i,i:N]
                flattened_output[jj,start_idx:start_idx+(N-i)] = row
                start_idx += N-i
                
    else:
        flattened_output = np.zeros((L,))
        start_idx = 0
        for i in range(0,N):
            row = cov_mat[i,i:N]
            flattened_output[start_idx:start_idx+(N-i)] = row
            start_idx += N-i
            
    return flattened_output

def convert_unique_cov_vector_to_mat(cov_vector, dim_mat):
    """Converts a vector containing the unique elements of a square covariance matrix
    back into the matrix

    Args:
        cov_vector (np array): the unique elements of the matrix
        dim_mat (int): the original dimension of the square matrix

    Returns:
        2D np array: the original matrix
    """    
    #squeeze if necessary
    vector = np.squeeze(cov_vector,)
    
    L = vector.shape[0]
    
    #allocate cov matrix
    cov_mat = np.zeros((dim_mat,dim_mat))
    
    #fill upper triangular part of matrix
    start_idx = 0
    for i in range(0,dim_mat):
        row = vector[start_idx:start_idx+(dim_mat-i)]
        cov_mat[i,i:dim_mat] = row
        
        start_idx += dim_mat-i
    
    #copy upper triangular parts of matrix to lower triangular
    cov_mat = cov_mat + cov_mat.T - np.diag(np.diag(cov_mat))
    
    return cov_mat

def unscale_cov_mat(cov_mat, meas_scale):
    """This function applies a scaling to the elements of a covariance matrix.
    Assume Covariance = ( E(x) - E(x_bar ) * ( E(x) - E(x_bar )^T 
    where x is a random variable.
    The scaling in meas_scale contains the individual scalings for the elements of E(x)

    Args:
        cov_mat (2d np array): the square covariance matrix
        meas_scale (2d np array): the scalings for each element of the random variables used to calculate the covariance

    Returns:
        2d np array: the scaled covariance matrix
    """    
    N, _ = meas_scale.shape
    
    scales = (meas_scale[:,1] - meas_scale[:,0]).reshape(-1,1)
    scale_mat = scales @ scales.T
    # print('scale_mat')
    # print(scale_mat)
    return cov_mat * (scale_mat)
    

def calculate_gait_state_errors(predictions, true_labels, STAIRS_THRESHOLD_ROUND=0.05, DO_PRINT=True):
    """this convenience function outputs the RMSEs of predicted gait states wrt ground truth states.
    The gait states are in order: [phase, speed, incline, is_stairs]

    Args:
        predictions (2d np array): predicted gait states
        true_labels (_type_): ground truth gait states
        STAIRS_THRESHOLD_ROUND (float, optional): the threshold above which to round the stairs gait state to -1 or 1. Defaults to 0.05.
        DO_PRINT (bool, optional): whether to print the RMSEs. Defaults to True.

    """    
    #round stairs to three locomotion modes
    true_labels[true_labels[:,3] < -STAIRS_THRESHOLD_ROUND,3] = -1
    true_labels[np.logical_and(true_labels[:,3] >= -STAIRS_THRESHOLD_ROUND, true_labels[:,3] <= STAIRS_THRESHOLD_ROUND),3] = 0
    true_labels[true_labels[:,3] > STAIRS_THRESHOLD_ROUND,3] = 1

    predictions[predictions[:,3] < -STAIRS_THRESHOLD_ROUND,3] = -1
    predictions[np.logical_and(predictions[:,3] >= -STAIRS_THRESHOLD_ROUND, predictions[:,3] <= STAIRS_THRESHOLD_ROUND),3] = 0
    predictions[predictions[:,3] > STAIRS_THRESHOLD_ROUND,3] = 1


    #compute RMSEs
    phase_losses = np.sqrt(np.mean(phase_dist(predictions[:,0], true_labels[:,0])**2))
    speed_losses = np.sqrt(np.mean((predictions[:,1] - true_labels[:,1])**2))
    incline_losses = np.sqrt(np.mean((predictions[:,2] - true_labels[:,2])**2))
    stair_height_accuracy = np.sum(true_labels[:,3] == predictions[:,3])/len(true_labels[:,3])
    stair_height_accuracy_ascent = np.sum(true_labels[true_labels[:,3] == 1,3] == predictions[true_labels[:,3] == 1,3])/len(true_labels[true_labels[:,3] == 1,3])
    stair_height_accuracy_descent = np.sum(true_labels[true_labels[:,3] == -1,3] == predictions[true_labels[:,3] == -1,3])/len(true_labels[true_labels[:,3] == -1,3])


    if DO_PRINT:
        print(f'Phase Losses: {phase_losses:.3f}')
        print(f'Speed Losses: {speed_losses:.3f}')
        print(f'Incline Losses: {incline_losses:.3f}')
        print(f'Is Stairs Accuracy: {stair_height_accuracy:.3f}')
        print(f'Is Stairs Accuracy, Ascent: {stair_height_accuracy_ascent:.3f}')
        print(f'Is Stairs Accuracy, Descent: {stair_height_accuracy_descent:.3f}')

    return phase_losses, speed_losses, incline_losses, stair_height_accuracy, stair_height_accuracy_ascent, stair_height_accuracy_descent

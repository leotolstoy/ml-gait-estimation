import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os, sys
import time
from scipy.signal import butter, lfilter

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)
sys.path.append(thisdir + '/utils')
sys.path.append('/home/jupyter/ml-gait-estimation')
sys.path.append('/home/jupyter/ml-gait-estimation/utils')

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')


import matplotlib
import matplotlib.pyplot as plt
import torch
from gait_transformer import GaitTransformer
from gait_kinematics_model import GaitModel, HeteroscedasticModel
from torque_profile import TorqueProfile
from attitude_ekf import AttitudeEKF
from phase_ekf import PhaseEKF, StairsOnlyEKF
from torch.autograd.functional import jacobian
from training_utils import generate_unit_vector, unscale_gait_state, unscale_kinematics, normalize_kinematics, unscale_kinematics_gradient, scale_gait_state, unscale_kinematics_gradient_phase_trig, convert_cov_mat_to_vector, convert_unique_cov_vector_to_mat, calculate_gait_state_errors
from training_utils import phase_dist
from torch_training_utils import enum_parameters


from filter_classes import FirstOrderLowPassLinearFilter

vicon_filenames = [
            'circuit1/circuit1_seg1/exoboot_Vicon_AB11-Frihet_circuit1_seg1_processed.csv',
            'circuit1/circuit1_seg2/exoboot_Vicon_AB11-Frihet_circuit1_seg2_processed.csv',
            'circuit1/circuit1_seg3/exoboot_Vicon_AB11-Frihet_circuit1_seg3_processed.csv',
            'circuit1/circuit1_seg4/exoboot_Vicon_AB11-Frihet_circuit1_seg4_processed.csv',
            'circuit2/circuit2_seg1/exoboot_Vicon_AB11-Frihet_circuit2_seg1_processed.csv',
            'circuit2/circuit2_seg2/exoboot_Vicon_AB11-Frihet_circuit2_seg2_processed.csv',
            'circuit2/circuit2_seg3/exoboot_Vicon_AB11-Frihet_circuit2_seg3_processed.csv',
            'circuit2/circuit2_seg4/exoboot_Vicon_AB11-Frihet_circuit2_seg4_processed.csv',
            'circuit3/circuit3_seg1/exoboot_Vicon_AB11-Frihet_circuit3_seg1_processed.csv',
            'circuit3/circuit3_seg2/exoboot_Vicon_AB11-Frihet_circuit3_seg2_processed.csv',
            'circuit3/circuit3_seg3/exoboot_Vicon_AB11-Frihet_circuit3_seg3_processed.csv',
            'circuit3/circuit3_seg4/exoboot_Vicon_AB11-Frihet_circuit3_seg4_processed.csv'
            ]

phase_rmses = []
speed_rmses = []
incline_rmses = []
stair_height_accuracies = []

phase_rmses_ss = []
speed_rmses_ss = []
incline_rmses_ss = []
stair_height_accuracies_ss = []

phase_rmses_nss = []
speed_rmses_nss = []
incline_rmses_nss = []
stair_height_accuracies_nss = []


Ns = []
start_idx = 200
predictions_total = np.array([])
true_labels_total = np.array([])
is_steady_state_total = np.array([])

for i, vicon_filename in enumerate(vicon_filenames):

    data = np.loadtxt(vicon_filename, delimiter=',',skiprows=1) 

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


    #SIM

    N_data = data.shape[0]
    print(N_data)

    model_dict = {'model_filepath': '../../torque_profile/torque_profile_coeffs.csv',
				'phase_order': 20,
				'speed_order': 1,
				'incline_order': 1}

    model_dict_stairs = {'model_filepath': '../../torque_profile/torque_profile_stairs_coeffs.csv',
                    'phase_order': 20,
                    'speed_order': 1,
                    'stair_height_order': 1}

    torque_profile = TorqueProfile(model_dict=model_dict,model_dict_stairs=model_dict_stairs)
    
    ML_WINDOW_SIZE = 150
    
    #set up neural network
    # Model parameters
    dim_val = 32 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 4 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    n_decoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 7 # The number of input variables. 1 if univariate forecasting.
    enc_seq_len = ML_WINDOW_SIZE # length of input given to encoder. Can have any integer value.
    dec_seq_len = 1 # length of input given to decoder. Can have any integer value.

    dropout_encoder = 0.1
    dropout_decoder = 0.1
    dropout_pos_enc = 0.0
    dropout_regression = 0.1
    dim_feedforward_encoder = 512
    dim_feedforward_decoder = 512

    num_predicted_features = 5 # The number of output variables. 

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
        
    #INITIALIZE BEST GENERAL MODEL
    gait_state_estimator = GaitTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        enc_seq_len=enc_seq_len,
        dropout_encoder=dropout_encoder,
        dropout_decoder=dropout_decoder,
        dropout_pos_enc=dropout_pos_enc,
        dropout_regression=dropout_regression,
        num_predicted_features=num_predicted_features,
        dim_feedforward_encoder=dim_feedforward_encoder,
        dim_feedforward_decoder=dim_feedforward_decoder,
    )
    
    enum_parameters(gait_state_estimator)

    gait_state_estimator.to(device)

    model_nickname = 'apollyon-three-stairs'
    general_model_dir = f'../../full_models/{model_nickname}/model_save_xval/'

    print(general_model_dir)
    
    checkpoint = torch.load(general_model_dir+'ml_gait_estimator_dec_best_model.tar',map_location=torch.device(device))
    
    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss General: {loss}')
    gait_state_estimator.load_state_dict(g)

    # Put models in evaluation mode
    gait_state_estimator.eval()
    
    #Set up start of sequence token
    SOS_token = 100 * torch.ones(1, 1, num_predicted_features).to(device).requires_grad_(False)
    
    
    # INITIALIZE KINEMATICS MODEL
    dim_val = 64 # 
    n_hidden_layers=4
    input_size = 5 # The number of input variables. 1 if univariate forecasting.
    num_predicted_features = 6 # The number of output variables. 

    gait_model = GaitModel(
        input_size=input_size,
        num_predicted_features=num_predicted_features,
        dim_val=dim_val,  
        n_hidden_layers=n_hidden_layers
    )
    gait_model.to(device)
    
    kinematics_model_nickname = 'gait-model-three-stairs'
    kinematics_model_dir = f'../../full_models/{kinematics_model_nickname}/model_save_xval/'
    checkpoint = torch.load(kinematics_model_dir+'best_gait_model.tar',map_location=torch.device(device))
    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss: {loss}')
    gait_model.load_state_dict(g)

    epoch = checkpoint['epoch']

    # Put model in evaluation mode
    gait_model.eval()
    
    enum_parameters(gait_model)
    
    #INITIALIZE HETEROSCEDASTIC MODEL
    
    dim_val = 64 # 
    n_hidden_layers=2
    input_size = 2 # The number of input variables. 1 if univariate forecasting.
    num_predicted_variables = 6

    heteroscedastic_model = HeteroscedasticModel(
        input_size=input_size,
        num_predicted_features=np.sum([i+1 for i in range(num_predicted_variables)]),
        dim_val=dim_val,  
        n_hidden_layers=n_hidden_layers
    )
    heteroscedastic_model.to(device)
    
    heteroscedastic_model_nickname = 'heteroscedastic-covariance-model'
    heteroscedastic_model_dir = f'../../full_models/{heteroscedastic_model_nickname}/model_save_xval/'
    checkpoint = torch.load(heteroscedastic_model_dir+'best_heteroscedastic_model.tar',map_location=torch.device(device))
    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss: {loss}')
    heteroscedastic_model.load_state_dict(g)

    epoch = checkpoint['epoch']

    # Put model in evaluation mode
    heteroscedastic_model.eval()
    enum_parameters(heteroscedastic_model)
    
    
    #Set up start of sequence token
    SOS_token_stairs = 100 * torch.ones(1, 1, num_predicted_features).to(device).requires_grad_(False)
    
    
    #Extract index for time steps
    DT_IDX = 6  

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
    

    #set up measurement buffer for neural network
    meas_buffer = np.zeros((ML_WINDOW_SIZE,7),dtype="float32")
    
    #set up filters for the heel acc's
    DO_FILTER = True
    heel_forward_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    heel_up_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
    
    
    
    # set up covariance to compute mahalanobis distances
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
    
    m_dist_importance = 10
        
    sigma_q_phase=7e-2
    sigma_q_speed=4e-2
    sigma_q_incline=7e-1
    sigma_q_is_stairs=7e-2
    
    R_meas_gait_state = np.diag([
        0.02**2,
        0.09**2,
        1.0**2,
        0.1**2,
        ])
    
    #set up EKF to smooth estimates
    sigma_foot = 3
    sigma_foot_vel = 30
    sigma_shank = 20
    sigma_shank_vel = 100
    sigma_heel_acc_forward = 3
    sigma_heel_acc_up = 2.5
    
    # R_meas_kinematics = confidence_covariance
    R_meas_kinematics = np.diag([sigma_foot**2,
    sigma_foot_vel**2,\
    sigma_shank**2,
    sigma_shank_vel**2,\
    sigma_heel_acc_forward**2, 
    sigma_heel_acc_up**2
    ])
    
    DO_EKF = True
    
    DO_GAIT_MODEL_IN_EKF = True and DO_EKF
    DO_HETEROSCEDASTIC = True and DO_GAIT_MODEL_IN_EKF
    
    if DO_GAIT_MODEL_IN_EKF:
        # print(R_meas_kinematics.shape)
        R_meas = np.block([
                    [R_meas_gait_state*1e0,               np.zeros((4, 6))],
                    [np.zeros((6, 4)), R_meas_kinematics*1e0]
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
    

    #handle replicating synch count delay
    DO_SYNCH_DELAY = True
    ACCOUNT_FOR_SYNCH_DELAY = not True and DO_SYNCH_DELAY
    SYNCH_DELAY_TIME = 0.015
    
    #set up plotting options
    SHOW_FULL_STATE = True
    PLOT_MEASURED = True
    PLOT_MEASURED_NORM = not True

    plot_mahalanobis_dist = np.zeros((N_data,3))
    
    m_distance_ekf_predictions_buffer = np.zeros((ML_WINDOW_SIZE,))
    m_distance_stairs_ekf_predictions_buffer = np.zeros((ML_WINDOW_SIZE,))
    
    timeSec_vec_sim = np.zeros((N_data,1))
    plot_gait_state_neural_network = np.zeros((N_data,4))
    plot_gait_state_ekf = np.zeros((N_data,4))
    plot_gait_state_ekf_estimate = np.zeros((N_data,4))
    plot_data_measured = np.zeros((N_data,7))
    plot_measurements_prediction_ekf = np.zeros((N_data,6))
    plot_measurements_prediction_neural_network = np.zeros((N_data,6))
    
    plot_des_torque = np.zeros((N_data,2))
    
    is_skipping_ekf_update = np.ones((N_data,))

    prev=0

    tic = time.time()

    # EXTRACT ACT VARIABLES
    footAngle_meas_vec = data[:,0]
    footAngleVel_meas_vec = data[:,1]
    shankAngle_meas_vec = data[:,2]
    shankAngleVel_meas_vec = data[:,3]
    heelAccForward_meas_vec = data[:,4] #92
    heelAccUp_meas_vec = data[:,5]#71
    dt_vec = data[:,6]
    timeSec_vec = np.cumsum(dt_vec,axis=0)
    actTorque_vec = data[:,11]
    desTorque_vec = data[:,12]
    phase_vec = data[:,13]
    speed_vec = data[:,14]
    incline_vec = data[:,15]
    is_stairs_vec = data[:,16]
    
    synch_count_vec = data[:,17]
    
    #exoboot data is already filtered, so don't filter it
    DO_FILTER = False
    
    if not DO_SYNCH_DELAY:
        print('ignoring synch count')
        synch_count_vec = np.arange(0,len(timeSec_vec))
        
    prev_synch_count = 0
    
    #initialize quantities we'd receive from the big Linux computer
    predicted_gait_state_neural_net = np.zeros((4,1))
    predicted_stairs_neural_net = np.zeros((1,1))
    predicted_kinematics_neural_net = np.zeros((6,1))
    z_model_kinematics = np.zeros((6,1))
    R_heteroscedastic = confidence_covariance

    for i,x in enumerate(data[:]):

        timeSec=timeSec_vec[i]
        dt = dt_vec[i]

        prev=timeSec
        footAngle_meas = footAngle_meas_vec[i]
        shankAngle_meas = shankAngle_meas_vec[i]
        footAngleVel_meas = footAngleVel_meas_vec[i]
        shankAngleVel_meas = shankAngleVel_meas_vec[i]

        heelAccForward_meas = heelAccForward_meas_vec[i]
        heelAccUp_meas = heelAccUp_meas_vec[i]
        synch_count = synch_count_vec[i]
        # print(synch_count)
        
        RECEIVED_NEW_DATA = not (synch_count == prev_synch_count)
                                                       
        #filter
        if DO_FILTER:
            heelAccForward_meas = heel_forward_acc_filter.step(i, heelAccForward_meas)
            heelAccUp_meas = heel_up_acc_filter.step(i, heelAccUp_meas)
                                                       
        z_measured = np.array([footAngle_meas, footAngleVel_meas, shankAngle_meas, shankAngleVel_meas, heelAccForward_meas, heelAccUp_meas, dt])
        
        z_measured = z_measured.reshape((1,-1))
        
        z_measured_norm = normalize_kinematics(z_measured, meas_scale)
        # print(z_measured_norm)
        # print(z_measured_norm.shape)
        # input()
        
        # run neural network
        #move all the measurements up one row, 
        # clearing the last row to be overwritten by new data
        meas_buffer[:-1,:] = meas_buffer[1:,:]

        #overwrite last row with most recent data
        meas_buffer[-1,:] = z_measured_norm

        # convert to torch tensor
        meas = torch.tensor(meas_buffer).to(device)

        #process numpy arrays into torch tensors for input
        meas = torch.unsqueeze(meas, dim=0)
        
        # print(meas)
        # print(meas.shape)
        # input()
        
        if RECEIVED_NEW_DATA:
            tgt = SOS_token.to(device)
            dts = meas[:,:,DT_IDX]
            dts = torch.unsqueeze(dts, dim=-1)
            with torch.no_grad():
                #predict gait state
                predicted_gait_state_tensor = gait_state_estimator(meas,tgt, dts)

            # Move logits and labels to CPU and convert to numpy
            predicted_gait_state = predicted_gait_state_tensor.detach().to('cpu').numpy()
            predicted_gait_state_neural_net = np.squeeze(predicted_gait_state, axis=1)

            #unscale gait state
            #outputs contains [phase, speed, incline, is_stairs, is_moving]
            predicted_gait_state_neural_net = unscale_gait_state(predicted_gait_state_neural_net, speed_scale, incline_scale, stair_height_scale).reshape(-1,1)
            
            #account for synch delay
            #add the delay time as a phase to the predicted state from the neural net
            if ACCOUNT_FOR_SYNCH_DELAY:
                predicted_gait_state_neural_net[0,0] += SYNCH_DELAY_TIME
                predicted_gait_state_neural_net[0,0] = predicted_gait_state_neural_net[0,0] % 1
                
            #compute predicted kinematics from the output of the neural network
            b_state = predicted_gait_state_neural_net.reshape(1,-1)
            b_state = scale_gait_state(b_state, speed_scale, incline_scale, stair_height_scale)
            b_state = torch.from_numpy(b_state).float().to(device)
            predicted_gait_state_tensor = torch.unsqueeze(b_state,dim=0)

            with torch.no_grad():
                predicted_kinematics_tensor = gait_model(predicted_gait_state_tensor)
                #process predicted kinematics
            predicted_kinematics = predicted_kinematics_tensor.detach().to('cpu').numpy()
            predicted_kinematics = np.squeeze(predicted_kinematics, axis=1)
            predicted_kinematics_neural_net = unscale_kinematics(predicted_kinematics, meas_scale).reshape(-1,1)
            
        #compute mahalanobis distances for kinematics
        d = z_measured[:,:-1].T - predicted_kinematics_neural_net
        m_distance_neural_net_predictions = np.sqrt(d.T @ confidence_covariance_inv @ d)    
        
        m_distance_ekf_predictions = 0
        
        #run through the EKF
        if DO_EKF:
            #change functionality st it doesnt iterate vefore ML window size
            phase_ekf.step(i, dt)
            
            if RECEIVED_NEW_DATA:
                #compute predicted kinematics from the prediction step of the EKF
                #this is used in the update step of the EKF
                b_state = phase_ekf.x_state.reshape(1,-1)
                
                if ACCOUNT_FOR_SYNCH_DELAY:
                    b_state[0,0] += SYNCH_DELAY_TIME
                    b_state[0,0] = b_state[0,0] % 1
                    
                b_state = scale_gait_state(b_state, speed_scale, incline_scale, stair_height_scale)
                b_state = torch.from_numpy(b_state).float().to(device)
                predicted_gait_state_tensor_ekf = torch.unsqueeze(b_state,dim=0)
                
                with torch.no_grad():
                    predicted_kinematics_tensor = gait_model(predicted_gait_state_tensor_ekf)
                predicted_kinematics = predicted_kinematics_tensor.detach().to('cpu').numpy()
                predicted_kinematics = np.squeeze(predicted_kinematics, axis=1)
                z_model_kinematics = unscale_kinematics(predicted_kinematics, meas_scale).reshape(-1,1)
            
            #compute ekf mahalanobis distance
            predicted_kinematics_ekf = z_model_kinematics.reshape(-1,1)
            d_ekf = z_measured[:,:-1].T - predicted_kinematics_ekf
            m_distance_ekf_predictions = np.sqrt(d_ekf.T @ confidence_covariance_inv @ d_ekf) 
            
            if DO_HETEROSCEDASTIC and RECEIVED_NEW_DATA:
                #generate heteroscedastic matrix using ekf state estimate
                h_vec = heteroscedastic_model(predicted_gait_state_tensor_ekf[:,:,0:2])
                #convert to numpy
                h_vec = h_vec.detach().to('cpu').numpy().squeeze(axis=1)
                # print(h_vec.shape)
                #apply unscale and exp operations
                h_vec = heteroscedastic_model.process_unique_covar_elements(h_vec)
                # print(h_vec)
                R_heteroscedastic = convert_unique_cov_vector_to_mat(h_vec,dim_mat=6)
                
                #recompute mahalanobis distance with effective R matrix from ekf
                # m_distance_ekf_predictions = np.sqrt(d_ekf.T @ np.linalg.solve(R_heteroscedastic, d_ekf)) 
                
            
            z_measured_ekf = predicted_gait_state_neural_net
            
            #only update if we received new data
            if RECEIVED_NEW_DATA:
                
                if DO_GAIT_MODEL_IN_EKF:

                    #compute numerical gradient
                    with torch.no_grad():
                        gradient_num = gait_model.compute_numerical_gradient(predicted_gait_state_tensor_ekf, predicted_kinematics_tensor, device)

                    kinematics_gradient = unscale_kinematics_gradient_phase_trig(gradient_num, meas_scale, speed_scale, incline_scale, stair_height_scale)
                    
                    #generate measurements for ekf by stacking the measured gait state and the measurements of the kinematics
                    z_measured_ekf = np.vstack((predicted_gait_state_neural_net, z_measured[:,:-1].T))
                        
                    phase_ekf.update(i, dt, z_measured_ekf, z_model_kinematics=z_model_kinematics, kinematics_gradient=kinematics_gradient,m_dist=m_distance_neural_net_predictions, R_heteroscedastic=R_heteroscedastic)
                    is_skipping_ekf_update[i] = 0
                    
                
                else:
                    phase_ekf.update(i, dt, z_measured_ekf, m_dist=m_distance_neural_net_predictions)
                    is_skipping_ekf_update[i] = 0
            
            predicted_gait_state_ekf = phase_ekf.x_state.reshape(-1,1)
            
            
            #get state estimate of ekf
            x_state_estimate_ekf = phase_ekf.x_state_estimate.reshape(-1,1)
            
            #update buffer
            m_distance_ekf_predictions_buffer[:-1] = m_distance_ekf_predictions_buffer[1:]
            m_distance_ekf_predictions_buffer[1] = m_distance_ekf_predictions
                        
            # input()
            
        #update synch_count
        prev_synch_count = synch_count
        
        #round is_stairs
        # predicted_gait_state[3] = np.round(predicted_gait_state[3])
        #round is_moving
        # predicted_gait_state[4] = np.round(predicted_gait_state[4])
        
        # print(predicted_kinematics.shape)
        
        
        timeSec_vec_sim[i] = timeSec
        plot_gait_state_neural_network[i,:] = predicted_gait_state_neural_net.flatten()
        plot_data_measured[i,:] = z_measured.flatten()
        plot_measurements_prediction_neural_network[i,:] = predicted_kinematics_neural_net.flatten()
        plot_mahalanobis_dist[i,0] = m_distance_neural_net_predictions
 
        if DO_EKF:
            plot_gait_state_ekf[i,:] = predicted_gait_state_ekf.flatten()
            plot_gait_state_ekf_estimate[i,:] = x_state_estimate_ekf.flatten()
            plot_measurements_prediction_ekf[i,:] = predicted_kinematics_ekf.flatten()
            plot_mahalanobis_dist[i,1] = m_distance_ekf_predictions
            
            des_torque_sim_ekf = torque_profile.evalTorqueProfile(predicted_gait_state_ekf[0,0], predicted_gait_state_ekf[1,0], predicted_gait_state_ekf[2,0], predicted_gait_state_ekf[3,0])
            
            plot_des_torque[i,1] = des_torque_sim_ekf

        des_torque_sim_nn = torque_profile.evalTorqueProfile(predicted_gait_state_neural_net[0,0], predicted_gait_state_neural_net[1,0], predicted_gait_state_neural_net[2,0], predicted_gait_state_neural_net[3,0])
        plot_des_torque[i,0] = des_torque_sim_nn
        
        
    toc = time.time()
    print(f"Ran simulation loop in {toc - tic:0.4f} seconds")

    # print sampling rate

    print('Sampling Rate')
    sample_rate = 1/np.mean(np.diff(data[:,0]))
    print(sample_rate)
    
    #output stats for exoboot data
    #generate losses for finetuned model
    phase_vec_comp = plot_gait_state_neural_network[:,0]
    speed_vec_comp = plot_gait_state_neural_network[:,1]
    incline_vec_comp = plot_gait_state_neural_network[:,2]
    is_stairs_vec_comp = plot_gait_state_neural_network[:,3]
    
    if DO_EKF:
        phase_vec_comp = plot_gait_state_ekf[:,0]
        speed_vec_comp = plot_gait_state_ekf[:,1]
        incline_vec_comp = plot_gait_state_ekf[:,2]
        is_stairs_vec_comp = plot_gait_state_ekf[:,3]
        
    predictions = np.hstack((phase_vec_comp.reshape(-1,1),
                                        speed_vec_comp.reshape(-1,1),
                                        incline_vec_comp.reshape(-1,1),
                                        is_stairs_vec_comp.reshape(-1,1),
                                        ))

    if SHOW_FULL_STATE:
        fig, axs = plt.subplots(4,1,sharex=True,figsize=(10,12))
        axs[0].plot(timeSec_vec, phase_vec,'b', label=r"$Phase_{hardware}$")
        axs[0].plot(timeSec_vec_sim, plot_gait_state_neural_network[:,0],'g', label=r"$Phase_{nn}$")
        axs[0].plot(timeSec_vec_sim, plot_gait_state_ekf[:,0],'r', label=r"$Phase_{ekf}$")
        axs[0].plot(timeSec_vec, phase_ground_truth,'k', label="Phase_{ground truth}")


        

        axs[0].legend()
        
        axs[1].plot(timeSec_vec, speed_vec,'b', label=r"$Speed_{hardware}$")
        axs[1].plot(timeSec_vec_sim, plot_gait_state_neural_network[:,1],'g', label=r"$Speed_{nn}$")
        axs[1].plot(timeSec_vec_sim, plot_gait_state_ekf[:,1],'r', label=r"$Speed_{ekf}$")
        axs[1].plot(timeSec_vec, speed_ground_truth,'k', label="Speed_{ground truth}")

        axs[1].legend()
        
        axs[2].plot(timeSec_vec, incline_vec,'b', label=r"$Incline_{hardware}$")
        axs[2].plot(timeSec_vec_sim, plot_gait_state_neural_network[:,2],'g', label=r"$Incline_{nn}$")
        axs[2].plot(timeSec_vec_sim, plot_gait_state_ekf[:,2],'r', label=r"$Incline_{ekf}$")
        
        axs[2].plot(timeSec_vec, incline_ground_truth,'k', label="Incline_{ground truth}")

        axs[2].legend()
        
        axs[3].plot(timeSec_vec, is_stairs_vec,'b', label=r"$stair height_{hardware}$")
        axs[3].plot(timeSec_vec_sim, plot_gait_state_neural_network[:,3],'g.', label=r"$stair height_{nn}$")
        axs[3].plot(timeSec_vec_sim, plot_gait_state_ekf[:,3],'r.', label=r"$stair height_{ekf}$")
        axs[3].plot(timeSec_vec, stairs_ground_truth,'k', label="Stairs_{ground truth}")

        
        axs[3].legend()
        
        axs[-1].set_xlabel("time (sec)")
        print("this is done (show state)")


    #END SIM

    true_labels = np.hstack((phase_ground_truth, speed_ground_truth, incline_ground_truth, stairs_ground_truth))

    #only consider the predictions starting from start_idx
    predictions = predictions[start_idx:,:]
    true_labels = true_labels[start_idx:,:]
    is_steady_state = is_steady_state[start_idx:,:]

    #reshape phase events to be 1D so np.nonzeros returns a 1D array
    phase_events = phase_events[start_idx:,:].reshape(-1)


    phase_event_idxs = np.nonzero(phase_events == 1)
    #insert idxs for beginning and end of trial
    phase_event_idxs = np.insert(phase_event_idxs,0,0)
    phase_event_idxs = np.append(phase_event_idxs,len(phase_events)-1)

    #extract number of phase events
    num_phase_events = len(phase_event_idxs)
    # print(num_phase_events)
    # print(phase_event_idxs)
    # input()
    for i in range(num_phase_events-1):
        current_idx = phase_event_idxs[i]
        next_idx = phase_event_idxs[i+1]

        # print(f'current_idx: {current_idx}')
        # print(f'next_idx: {next_idx}')

        predictions_step = predictions[current_idx:next_idx,:]
        true_labels_step = true_labels[current_idx:next_idx,:]
        is_steady_state_step = is_steady_state[current_idx:next_idx,:]

        #calculate RMSE for each stride
        (phase_rmse_step, 
        speed_rmse_step, 
        incline_rmse_step, 
        stair_height_accuracy_step, 
        stair_height_accuracy_ascent_step, 
        stair_height_accuracy_descent_step) = calculate_gait_state_errors(predictions_step, true_labels_step, STAIRS_THRESHOLD_ROUND=0.5, DO_PRINT=False)

        #update RMSE overall vector
        phase_rmses.append(phase_rmse_step)
        speed_rmses.append(speed_rmse_step)
        incline_rmses.append(incline_rmse_step)
        stair_height_accuracies.append(stair_height_accuracy_step)

        IS_STEADY_STATE = np.mean(is_steady_state_step) >= 0.5
        # print(IS_STEADY_STATE)
        # input()

        if IS_STEADY_STATE:
            phase_rmses_ss.append(phase_rmse_step)
            speed_rmses_ss.append(speed_rmse_step)
            incline_rmses_ss.append(incline_rmse_step)
            stair_height_accuracies_ss.append(stair_height_accuracy_step)

        else:
            phase_rmses_nss.append(phase_rmse_step)
            speed_rmses_nss.append(speed_rmse_step)
            incline_rmses_nss.append(incline_rmse_step)
            stair_height_accuracies_nss.append(stair_height_accuracy_step)



phase_rmses = np.array(phase_rmses)
speed_rmses = np.array(speed_rmses)
incline_rmses = np.array(incline_rmses)
stair_height_accuracies = np.array(stair_height_accuracies)

phase_rmses_ss = np.array(phase_rmses_ss)
speed_rmses_ss = np.array(speed_rmses_ss)
incline_rmses_ss = np.array(incline_rmses_ss)
stair_height_accuracies_ss = np.array(stair_height_accuracies_ss)

phase_rmses_nss = np.array(phase_rmses_nss)
speed_rmses_nss = np.array(speed_rmses_nss)
incline_rmses_nss = np.array(incline_rmses_nss)
stair_height_accuracies_nss = np.array(stair_height_accuracies_nss)



print('Overall')
print(f'phase_loss_avg: {np.mean(phase_rmses)} +- {np.std(phase_rmses)}')
print(f'speed_loss_avg: {np.mean(speed_rmses)} +- {np.std(speed_rmses)}')
print(f'incline_loss_avg: {np.mean(incline_rmses)} +- {np.std(incline_rmses)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies)} +- {np.std(stair_height_accuracies)}')
print()

print('Steady State')
print(f'phase_loss_avg: {np.mean(phase_rmses_ss)} +- {np.std(phase_rmses_ss)}')
print(f'speed_loss_avg: {np.mean(speed_rmses_ss)} +- {np.std(speed_rmses_ss)}')
print(f'incline_loss_avg: {np.mean(incline_rmses_ss)} +- {np.std(incline_rmses_ss)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies_ss)} +- {np.std(stair_height_accuracies_ss)}')
print()

print('Transitory')
print(f'phase_loss_avg: {np.mean(phase_rmses_nss)} +- {np.std(phase_rmses_nss)}')
print(f'speed_loss_avg: {np.mean(speed_rmses_nss)} +- {np.std(speed_rmses_nss)}')
print(f'incline_loss_avg: {np.mean(incline_rmses_nss)} +- {np.std(incline_rmses_nss)}')
print(f'stair_height_accuracy_avg: {np.mean(stair_height_accuracies_nss)} +- {np.std(stair_height_accuracies_nss)}')
print()




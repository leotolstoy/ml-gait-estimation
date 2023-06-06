'''
Live script to run on a powerful computer that evaluates the GaitTransformer
'''

import os, sys
from time import sleep, time, strftime, perf_counter
import traceback
import csv
import numpy as np
from StatProfiler import StatProfiler, SSProfile
from SoftRealtimeLoop import SoftRealtimeLoop
# from ActPackMan import ActPackMan, FlexSEA
# import gc
import torch
from gait_transformer import GaitTransformer
from gait_kinematics_model import GaitModel, HeteroscedasticModel
from torch_training_utils import enum_parameters
from training_utils import unscale_gait_state, unscale_kinematics, normalize_kinematics, unscale_kinematics_gradient, scale_gait_state, unscale_kinematics_gradient_phase_trig
from UdpBinarySynch import UdpBinarySynchA

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append(thisdir + '/utils')
# gc.disable()

np.set_printoptions(precision=4)


thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

FADE_IN_TIME = 1.0
SPOOF_PI = False
ML_WINDOW_SIZE = 150
def main(): # exo_right, writer, fd_l, am, run_time = 60*10

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
        
    #INITIALIZE BEST GENERAL GAIT PREDICTOR MODEL
    best_general_model = GaitTransformer(                       
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
        dim_feedforward_decoder=dim_feedforward_decoder
    )
    
    enum_parameters(best_general_model)
    
    best_general_model.to(device)

    #SELECT YOUR MODEL
    model_nickname = 'apollyon-three-stairs'

    # general_model_dir = f'./full_models/{model_nickname}/model_save_xval/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB01_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB02_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB03_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB04_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB05_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB06_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB07_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB08_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB09_exoboot/'
    # general_model_dir = f'./full_models/{model_nickname}/finetune_AB10_exoboot/'

    general_model_dir = f'./full_models/{model_nickname}/finetune_XSUB_exoboot/'

    checkpoint = torch.load(general_model_dir+'ml_gait_estimator_dec_best_model.tar',map_location=torch.device(device))

    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss General: {loss}')
    best_general_model.load_state_dict(g)

    # Put models in evaluation mode
    best_general_model.eval()
    #Set up start of sequence token
    SOS_token = 100 * torch.ones(1, 1, num_predicted_features).to(device).requires_grad_(False)
    

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
    kinematics_model_dir = f'./full_models/{kinematics_model_nickname}/model_save_xval/'
    checkpoint = torch.load(kinematics_model_dir+'best_gait_model.tar')
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
    heteroscedastic_model_dir = f'./full_models/{heteroscedastic_model_nickname}/model_save_xval/'
    checkpoint = torch.load(heteroscedastic_model_dir+'best_heteroscedastic_model.tar')
    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss: {loss}')
    heteroscedastic_model.load_state_dict(g)

    epoch = checkpoint['epoch']

    # Put model in evaluation mode
    heteroscedastic_model.eval()
    enum_parameters(heteroscedastic_model)
    
        

    
    #Extract index for time steps
    DT_IDX = 6  
    
    #handle synch count delay
    ACCOUNT_FOR_SYNCH_DELAY = True
    SYNCH_DELAY_TIME = 0.015
    
    synch = UdpBinarySynchA(
        recv_IP="192.168.1.127",
        recv_port=5557,
        send_IP="192.168.1.104", #104
        send_port=5558)

    loop = SoftRealtimeLoop(dt=0.01, report=True, fade=0.1)
    N_MESSAGE_SEND = 4 + 6 + 6 + (6*5) + (6+5+4+3+2+1)
    
    
    output_data = np.array([0.0]*N_MESSAGE_SEND)

    #set up measurement buffer
    meas_buffer = np.zeros((ML_WINDOW_SIZE,7),dtype="float32")
    i = 0

    for t in loop:# inProcedure:
        
        SSProfile("full loop").tic()
        #LOAD IN DATA FROM PI
        #send data to pi and receive a column vector of :
        # - z_measured, the measured kinematics (and dt) from the pi (7x1)
        # - x_state, the current state of the ekf (5x1)
        
        data_from_pi = synch.update(output_data)
        
        
        if data_from_pi is None:
            print("NONE")
            continue
        if not np.all([np.isfinite(x) for x in data_from_pi]):
            print("NO DATA")
            continue
        # print("\t\t\t",data_from_pi)
        
        z_measured = data_from_pi[0:7]
        x_state_ekf = data_from_pi[7:11]
        

        z_measured = z_measured.reshape((1,-1))
        dt = z_measured[0, DT_IDX]
        if SPOOF_PI: z_measured = np.random.random((7))
        z_measured_norm = normalize_kinematics(z_measured, meas_scale)


        # print(z_measured)

        #DO CONCATENATION OF DATA
        #move all the measurements up one row, 
        # clearing the last row to be overwritten by new data
        meas_buffer[:-1,:] = meas_buffer[1:,:]

        #overwrite last row with most recent data
        meas_buffer[-1,:] = z_measured_norm

        # print(meas_buffer)
        # input()

        # convert to torch tensor
        meas = torch.tensor(meas_buffer).to(device)

        #process numpy arrays into torch tensors for input
        meas = torch.unsqueeze(meas, dim=0)
            
        tgt = SOS_token.to(device)
        dts = meas[:,:,DT_IDX]
        dts = torch.unsqueeze(dts, dim=-1)
        # print(dts.shape)

        # Compute predicted gait state from the neural net
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            predicted_gait_state_tensor = best_general_model(meas,tgt, dts)
        
        predicted_gait_state = predicted_gait_state_tensor.detach().to('cpu').numpy()
        predicted_gait_state_neural_net = np.squeeze(predicted_gait_state, axis=1)
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
        
        
        #compute predicted kinematics from the current EKF state estimate
        #this is used in the update step of the EKF
        b_state = np.copy(x_state_ekf).reshape(1,-1)
        
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
        
        #compute numerical gradient for use in the gait model update step
        with torch.no_grad():
            gradient_num = gait_model.compute_numerical_gradient(predicted_gait_state_tensor_ekf, predicted_kinematics_tensor, device)

        kinematics_gradient = unscale_kinematics_gradient_phase_trig(gradient_num, meas_scale, speed_scale, incline_scale, stair_height_scale)
        
        #generate heteroscedastic matrix using ekf state estimate
        heteroscedastic_vec = heteroscedastic_model(predicted_gait_state_tensor_ekf[:,:,0:2])
        #convert to numpy
        heteroscedastic_vec = heteroscedastic_vec.detach().to('cpu').numpy().squeeze(axis=1)
        
        
        #PUSH OUT DATA
        output_data = np.vstack((predicted_gait_state_neural_net, 
                                 predicted_kinematics_neural_net,
                                 z_model_kinematics,
                                 kinematics_gradient.reshape(-1,1),
                                 heteroscedastic_vec.reshape(-1,1)))
        # print(predicted_gait_state_ekf)
        i += 1
        SSProfile("full loop").toc()

        
    return True



if __name__ == '__main__':
    main()
    

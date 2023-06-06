"""This script plots the gait model at a few select gait states
"""
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
import time
import gc
import os, sys

import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import torch
from gait_transformer import unscale_gait_state, unscale_kinematics, normalize_kinematics, unscale_kinematics_gradient, scale_gait_state
from gait_kinematics_model import GaitModel

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)
sys.path.append(thisdir + '/utils')

def main():

    # SET UP KINEMATICS MODEL
    dim_val = 64 # 
    n_hidden_layers=4
    input_size = 6 # The number of input variables. 1 if univariate forecasting.
    num_predicted_features = 6 # The number of output variables. 

    best_model = GaitModel(
        input_size=input_size,
        num_predicted_features=num_predicted_features,
        dim_val=dim_val,  
        n_hidden_layers=n_hidden_layers
    )


    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")


    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    best_model.to(device)


    model_nickname = 'gait-model'
    REMOVE_SUBS_XVAL = True
    model_dir = f'./full_models/{model_nickname}/model_save/'
    if REMOVE_SUBS_XVAL:
        model_dir = f'./full_models/{model_nickname}/model_save_xval/'

    checkpoint = torch.load(model_dir+'best_gait_model.tar')
    g = checkpoint['model_state_dict']
    loss = checkpoint['loss']
    print(f'Lowest Loss: {loss}')
    best_model.load_state_dict(g)

    epoch = checkpoint['epoch']

    # Put model in evaluation mode
    best_model.eval()

    speed_vec = np.linspace(0.5,1.0)
    phase_vec = np.linspace(0.0,1.0,1000)
    speed_scale = (0,2)
    incline_scale = (-10,10)
    stair_height_scale = (-0.2,0.2)
    meas_scale = np.array([[-69.35951035,  27.62815047],\
                            [-456.18013759,  401.13782617],\
                            [-63.71649984,  22.06632622],\
                            [-213.4786175,   396.93801619],\
                            [-35.26603985,  20.78473636],\
                            [-20.95456523,  14.63961137],\
                              [0,1]])

    xv, yv = np.meshgrid(phase_vec, speed_vec, sparse=False, indexing='ij')

    foot_angles_ramp = np.zeros((xv.shape))
    foot_angles_stairs = np.zeros((xv.shape))

    heel_acc_forward_ramp = np.zeros((xv.shape))
    heel_acc_forward_stairs = np.zeros((xv.shape))

    heel_acc_up_ramp = np.zeros((xv.shape))
    heel_acc_up_stairs = np.zeros((xv.shape))


    for i in range(len(phase_vec)):
        for j in range(len(speed_vec)):
            b_state_ramp = np.array([[phase_vec[i],speed_vec[j],0,0,1]])
            b_state_stairs = np.array([[phase_vec[i],speed_vec[j],0,0.17,1]])

            b_state_ramp = scale_gait_state(b_state_ramp, speed_scale, incline_scale, stair_height_scale)
            b_state_stairs = scale_gait_state(b_state_stairs, speed_scale, incline_scale, stair_height_scale)

            b_state_ramp = torch.from_numpy(b_state_ramp).float().to(device)
            b_state_stairs = torch.from_numpy(b_state_stairs).float().to(device)

            b_state_ramp = torch.unsqueeze(b_state_ramp,dim=0)
            b_state_stairs = torch.unsqueeze(b_state_stairs,dim=0)

            # print(b_state_ramp.shape)
            # print(b_state_ramp.device)

            with torch.no_grad():
                # outputs_ramp = best_model(b_state_ramp,tgt=SOS_token)
                # outputs_stairs = best_model(b_state_stairs,tgt=SOS_token)
                outputs_ramp = best_model(b_state_ramp)
                outputs_stairs = best_model(b_state_stairs)

            foot_angles_ramp[i,j] = (outputs_ramp[0,0,0] * (meas_scale[0,1] - meas_scale[0,0])) + meas_scale[0,0]
            foot_angles_stairs[i,j] = (outputs_stairs[0,0,0] * (meas_scale[0,1] - meas_scale[0,0])) + meas_scale[0,0]
            heel_acc_forward_ramp[i,j] = (outputs_ramp[0,0,4] * (meas_scale[4,1] - meas_scale[4,0])) + meas_scale[4,0]
            heel_acc_forward_stairs[i,j] = (outputs_stairs[0,0,4] * (meas_scale[4,1] - meas_scale[4,0])) + meas_scale[4,0]
            heel_acc_up_ramp[i,j] = (outputs_ramp[0,0,5] * (meas_scale[5,1] - meas_scale[5,0])) + meas_scale[5,0]
            heel_acc_up_stairs[i,j] = (outputs_stairs[0,0,5] * (meas_scale[5,1] - meas_scale[5,0])) + meas_scale[5,0]


    fig, axs = plt.subplots(3,2,subplot_kw={'projection':'3d'},figsize=(8,8),sharex=True)

    axs[0,0].plot_surface(xv, yv, foot_angles_ramp,cmap='viridis')
    axs[0,0].set_xlabel('Phase')
    axs[0,0].set_ylabel('Speed (m/s)')
    axs[0,0].set_zlabel('Foot Angle Ramp (deg)')

    # axs[0,1].plot_surface(xv, yv, foot_angles_ramp,cmap='viridis')
    axs[0,1].plot_surface(xv, yv, foot_angles_stairs,cmap='jet')
    axs[0,1].set_xlabel('Phase')
    axs[0,1].set_ylabel('Speed (m/s)')
    axs[0,1].set_zlabel('Foot Angle Stairs (deg)')

    axs[1,0].plot_surface(xv, yv, heel_acc_forward_ramp,cmap='viridis')
    axs[1,0].set_xlabel('Phase')
    axs[1,0].set_ylabel('Speed (m/s)')
    axs[1,0].set_zlabel('Heel Acc Forward Ramp (deg)')

    # axs[1,1].plot_surface(xv, yv, heel_acc_forward_ramp,cmap='viridis')
    axs[1,1].plot_surface(xv, yv, heel_acc_forward_stairs,cmap='jet')
    axs[1,1].set_xlabel('Phase')
    axs[1,1].set_ylabel('Speed (m/s)')
    axs[1,1].set_zlabel('Heel Acc Forward Stairs (deg)')

    axs[2,0].plot_surface(xv, yv, heel_acc_up_ramp,cmap='viridis')
    axs[2,0].set_xlabel('Phase')
    axs[2,0].set_ylabel('Speed (m/s)')
    axs[2,0].set_zlabel('Heel Acc Up Ramp (deg)')

    # axs[1,1].plot_surface(xv, yv, heel_acc_forward_ramp,cmap='viridis')
    axs[2,1].plot_surface(xv, yv, heel_acc_up_stairs,cmap='jet')
    axs[2,1].set_xlabel('Phase')
    axs[2,1].set_ylabel('Speed (m/s)')
    axs[2,1].set_zlabel('Heel Acc Up Stairs (deg)')

    plt.show()
if __name__ == '__main__':
    main()
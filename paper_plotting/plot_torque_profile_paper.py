""" Plots the torque profile"""
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../utils/')

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')


import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"
from torque_profile import TorqueProfile
from matplotlib import cm
from fig_colors import * 

ramp_vec = np.linspace(-10,10)
phase_vec = np.linspace(0,1,500)

xv, yv = np.meshgrid(phase_vec, ramp_vec, sparse=False, indexing='ij')

torques_ramp = np.zeros((xv.shape))
model_dict = {'model_filepath': '../torque_profile/torque_profile_coeffs.csv',
				'phase_order': 20,
				'speed_order': 1,
				'incline_order': 1}

model_dict_stairs = {'model_filepath': '../torque_profile/torque_profile_stairs_coeffs.csv',
				'phase_order': 20,
				'speed_order': 1,
				'stair_height_order': 1}

torque_profile = TorqueProfile(model_dict=model_dict,model_dict_stairs=model_dict_stairs)


# Generate torque profile for ramp
for i in range(len(phase_vec)):
	for j in range(len(ramp_vec)):
		torques_ramp[i,j] = torque_profile.evalTorqueProfile(phase_vec[i],1,ramp_vec[j],0)


# Generate torque profile for stairs
torques_stair_ascent = np.zeros((phase_vec.shape))
torques_stair_descent = np.zeros((phase_vec.shape))

for i in range(len(phase_vec)):
    torques_stair_ascent[i] = torque_profile.evalTorqueProfile(phase_vec[i],1,0,1)
    torques_stair_descent[i] = torque_profile.evalTorqueProfile(phase_vec[i],1,0,-1)	


# Set up fig hyperparameters like font sizes and the size of the figure
figWidth = 4
figHeight = 7
fontSizeAxes = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

fig = plt.figure(figsize=(figWidth,figHeight))

ax1 = fig.add_subplot(2,1,1, projection='3d')


# Plot ramp profile
ax1.plot_surface(xv, yv, torques_ramp,cmap='viridis')
ax1.set_xlabel('Phase', fontsize=fontSizeAxes)
ax1.set_ylabel('Incline (deg)', fontsize=fontSizeAxes)
ax1.set_zlabel('Torque (N-m)', fontsize=fontSizeAxes)
ax1.set_xlim(0,1)
ax1.set_ylim(-10,10)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.xaxis.set_tick_params(labelsize=fontSizeAxes)
ax1.yaxis.set_tick_params(labelsize=fontSizeAxes)
ax1.zaxis.set_tick_params(labelsize=fontSizeAxes)
ax1.xaxis.set_tick_params(width=1.5)
ax1.yaxis.set_tick_params(width=1.5)
ax1.zaxis.set_tick_params(width=1.5)


# Plot stairs profile
ax2 = fig.add_subplot(2,1,2)

ax2.plot(phase_vec, torques_stair_ascent,color=blueColor,label='Stair Ascent',linewidth=2)
ax2.plot(phase_vec, torques_stair_descent,color=redColor,label='Stair Descent',linewidth=2)

ax2.set_xlabel('Phase', fontsize=fontSizeAxes)
ax2.set_ylabel('Torque (N-m)', fontsize=fontSizeAxes)
ax2.set_xlim(0,1)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
ax2.xaxis.set_tick_params(labelsize=fontSizeAxes)
ax2.yaxis.set_tick_params(labelsize=fontSizeAxes)
ax2.xaxis.set_tick_params(width=1.5)
ax2.yaxis.set_tick_params(width=1.5)
ax2.legend(frameon=False,fontsize=fontSizeAxes)

# add vertical space between subplots
fig.subplots_adjust(hspace=0.4)
fig.text(0.1,0.85,'(a)', fontsize=MEDIUM_SIZE)
fig.text(0.1,0.47,'(b)', fontsize=MEDIUM_SIZE)

# plt.tight_layout()

filename = f'torque_profile_raw.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'torque_profile_raw.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')

plt.show()












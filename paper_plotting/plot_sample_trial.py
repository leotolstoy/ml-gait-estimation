""" Plots the results of an exoskeleton trial """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
import pandas as pd
from fig_colors import *

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"

# filename = "../live_data/AB04-Kronecker/circuit1/circuit1_seg1/exoboot_Vicon_AB04-Kronecker_circuit1_seg1_processed.csv"
# filename = "../live_data/AB05-Eagle/circuit3/circuit3_seg1/exoboot_Vicon_AB05-Eagle_circuit3_seg1_processed.csv"
filename = "../live_data/AB09-Brand/circuit3/circuit3_seg1/exoboot_Vicon_AB09-Brand_circuit3_seg1_processed.csv"


df = pd.read_csv(filename)

dt = df['dt'].to_numpy()
phase_hardware = df['phase_hardware'].to_numpy()
speed_hardware = df['speed_hardware'].to_numpy()
incline_hardware = df['incline_hardware'].to_numpy()
stairs_hardware = df['stairs_hardware'].to_numpy()

timeSec = np.cumsum(dt)

phase_ground_truth = df['phase_ground_truth'].to_numpy()
speed_ground_truth = df['speed_ground_truth'].to_numpy()
incline_ground_truth = df['incline_ground_truth'].to_numpy()
stairs_ground_truth = df['stairs_ground_truth'].to_numpy()


fontSizeAxes = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

figWidth = 8
figHeight = 5

#PLOT STATES

fig, axs = plt.subplots(4,1,sharex=True,figsize=(figWidth,figHeight))

axs[0].plot(timeSec, phase_ground_truth, label="Ground Truth", color=blueColor,linewidth=2)
axs[0].plot(timeSec, phase_hardware, label="Hardware", color=redColor,linewidth=2)
axs[0].set_ylabel("Phase")
axs[0].legend(frameon=False,fontsize=fontSizeAxes)

axs[1].plot(timeSec, speed_ground_truth, label="Ground Truth", color=blueColor,linewidth=2)
axs[1].plot(timeSec, speed_hardware, label="Hardware", color=redColor,linewidth=2)
axs[1].set_ylabel("Speed (m/s)")
# axs[1].legend()

axs[2].plot(timeSec, incline_ground_truth, label="Ground Truth", color=blueColor,linewidth=2)
axs[2].plot(timeSec, incline_hardware, label="Hardware", color=redColor,linewidth=2)
axs[2].set_ylabel("Incline (deg)")
# axs[2].legend()

axs[3].plot(timeSec, stairs_ground_truth, label="Ground Truth", color=blueColor,linewidth=2)
axs[3].plot(timeSec, stairs_hardware, label="Hardware", color=redColor,linewidth=2)
axs[3].set_ylabel("Is Stairs")
# axs[3].legend()

axs[-1].set_xlabel("Time (sec)")
print("this is done")


for i in range(4):
    ax = axs[i]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.xaxis.set_tick_params(labelsize=fontSizeAxes)
    ax.yaxis.set_tick_params(labelsize=fontSizeAxes)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)


filename = f'sample_trial.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'sample_trial.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')
plt.show()



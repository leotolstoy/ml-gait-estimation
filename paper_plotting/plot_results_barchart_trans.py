"""This script plots all the results of the transitory strides in all trials in barchart form
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t as t_dist
from fig_colors import *

#HISTOGRAM

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"


filename = "../live_data/subj_results.xlsx"
df = pd.read_excel(filename, sheet_name='Summary',index_col=0, engine='openpyxl')

# print(df.head())
# print(df['Phase Mean']['ML-Personalized-SteadyState'])

# input()
figWidth = 4
figHeight = 4

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# PULL RESULTS FROM SPREADSHEET
# personalized Trans (rights, go vote)
phase_personalized_mean_rmse_trans = df['Phase Mean']['ML-Personalized-Transitory']*100
phase_personalized_std_rmse_trans = df['Phase Std']['ML-Personalized-Transitory']*100
speed_personalized_mean_rmse_trans = df['Speed Mean']['ML-Personalized-Transitory']
speed_personalized_std_rmse_trans = df['Speed Std']['ML-Personalized-Transitory']
incline_personalized_mean_rmse_trans = df['Incline Mean']['ML-Personalized-Transitory']
incline_personalized_std_rmse_trans = df['Incline Std']['ML-Personalized-Transitory']
stairs_personalized_mean_acc_trans = df['Stairs Mean']['ML-Personalized-Transitory']*100
stairs_personalized_std_acc_trans = df['Stairs Std']['ML-Personalized-Transitory']*100

# xval trans
phase_xval_mean_rmse_trans = df['Phase Mean']['ML-XVal-Transitory']*100
phase_xval_std_rmse_trans = df['Phase Std']['ML-XVal-Transitory']*100
speed_xval_mean_rmse_trans = df['Speed Mean']['ML-XVal-Transitory']
speed_xval_std_rmse_trans = df['Speed Std']['ML-XVal-Transitory']
incline_xval_mean_rmse_trans = df['Incline Mean']['ML-XVal-Transitory']
incline_xval_std_rmse_trans = df['Incline Std']['ML-XVal-Transitory']
stairs_xval_mean_acc_trans = df['Stairs Mean']['ML-XVal-Transitory']*100
stairs_xval_std_acc_trans = df['Stairs Std']['ML-XVal-Transitory']*100

# Gen trans
phase_gen_mean_rmse_trans = df['Phase Mean']['ML-Gen-Transitory']*100
phase_gen_std_rmse_trans = df['Phase Std']['ML-Gen-Transitory']*100
speed_gen_mean_rmse_trans = df['Speed Mean']['ML-Gen-Transitory']
speed_gen_std_rmse_trans = df['Speed Std']['ML-Gen-Transitory']
incline_gen_mean_rmse_trans = df['Incline Mean']['ML-Gen-Transitory']
incline_gen_std_rmse_trans = df['Incline Std']['ML-Gen-Transitory']
stairs_gen_mean_acc_trans = df['Stairs Mean']['ML-Gen-Transitory']*100
stairs_gen_std_acc_trans = df['Stairs Std']['ML-Gen-Transitory']*100

# EKF trans
phase_ekf_mean_rmse_trans = df['Phase Mean']['EKF-Transitory']*100
phase_ekf_std_rmse_trans = df['Phase Std']['EKF-Transitory']*100
speed_ekf_mean_rmse_trans = df['Speed Mean']['EKF-Transitory']
speed_ekf_std_rmse_trans = df['Speed Std']['EKF-Transitory']
incline_ekf_mean_rmse_trans = df['Incline Mean']['EKF-Transitory']
incline_ekf_std_rmse_trans = df['Incline Std']['EKF-Transitory']

# TBE trans
phase_tbe_mean_rmse_trans = df['Phase Mean']['TBE-Transitory']*100
phase_tbe_std_rmse_trans = df['Phase Std']['TBE-Transitory']*100

# aggregate results into vectors
phases = [phase_personalized_mean_rmse_trans, phase_xval_mean_rmse_trans, phase_gen_mean_rmse_trans, phase_ekf_mean_rmse_trans, phase_tbe_mean_rmse_trans]
phases_std = [phase_personalized_std_rmse_trans, phase_xval_std_rmse_trans, phase_gen_std_rmse_trans, phase_ekf_std_rmse_trans, phase_tbe_std_rmse_trans]

speeds = [speed_personalized_mean_rmse_trans, speed_xval_mean_rmse_trans, speed_gen_mean_rmse_trans, speed_ekf_mean_rmse_trans]
speeds_std = [speed_personalized_std_rmse_trans, speed_xval_std_rmse_trans, speed_gen_std_rmse_trans, speed_ekf_std_rmse_trans]

inclines = [incline_personalized_mean_rmse_trans, incline_xval_mean_rmse_trans, incline_gen_mean_rmse_trans, incline_ekf_mean_rmse_trans]
inclines_std = [incline_personalized_std_rmse_trans, incline_xval_std_rmse_trans, incline_gen_std_rmse_trans, incline_ekf_std_rmse_trans]

stairs = [stairs_personalized_mean_acc_trans, stairs_xval_mean_acc_trans, stairs_gen_mean_acc_trans]
stairs_std = [stairs_personalized_std_acc_trans, stairs_xval_std_acc_trans, stairs_gen_std_acc_trans]


conditions = ['ML\nInd.', 'ML\nDev.', 'ML\nGen.', 'EKF', 'TBE']
colors = [blueColor, purpleColor, greenColor, redColor, grayColor]
x_pos = np.arange(len(conditions))

# print(phases)
# print(phases_std)
# print(conditions)
# print(colors)
# print(x_pos[:5])

fig, axs = plt.subplots(2,2,figsize=(figWidth,figHeight))
axs[0,0].bar(x_pos, phases, yerr=phases_std, color=colors, align='center', ecolor='black')
axs[0,0].set_xticks(x_pos)
axs[0,0].set_xticklabels(conditions)
axs[0,0].set_ylabel('Phase RMSE (%)', fontsize=MEDIUM_SIZE)


axs[0,1].bar(x_pos[:4], speeds, yerr=speeds_std, color=colors[:4], align='center', ecolor='black')
axs[0,1].set_xticks(x_pos[:4])
axs[0,1].set_xticklabels(conditions[:4])
axs[0,1].set_ylabel('Speed RMSE (m/s)', fontsize=MEDIUM_SIZE)

axs[1,0].bar(x_pos[:4], inclines, yerr=inclines_std, color=colors[:4], align='center', ecolor='black')
axs[1,0].set_xticks(x_pos[:4])
axs[1,0].set_xticklabels(conditions[:4])
axs[1,0].set_ylabel('Incline RMSE (deg)', fontsize=MEDIUM_SIZE)


axs[1,1].bar(x_pos[:3], stairs, yerr=stairs_std, color=colors[:3], align='center', ecolor='black')
axs[1,1].set_xticks(x_pos[:3])
axs[1,1].set_xticklabels(conditions[:3])
axs[1,1].set_ylabel('Is Stairs Accuracy (%)', fontsize=MEDIUM_SIZE)

fig.subplots_adjust(wspace=1.0)

fig.text(0.08,0.92,'(a)', fontsize=MEDIUM_SIZE)
fig.text(0.57,0.92,'(b)', fontsize=MEDIUM_SIZE)
fig.text(0.08,0.42,'(c)', fontsize=MEDIUM_SIZE)
fig.text(0.57,0.42,'(d)', fontsize=MEDIUM_SIZE)




for i in range(2):
    for j in range(2):
        ax = axs[i,j]
        ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)
        ax.yaxis.set_tick_params(labelsize=SMALL_SIZE)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)


        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

fig.subplots_adjust(hspace=0.7)
fig.subplots_adjust(wspace=0.7)
# plt.tight_layout()


filename = f'results_barchart_trans_raw.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'results_barchart_trans_raw.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')



plt.show()
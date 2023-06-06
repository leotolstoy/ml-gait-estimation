"""This script plots all the results of the steady-state strides in all trials in barchart form
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
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
figWidth = 8
figHeight = 2.5

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# PULL RESULTS FROM SPREADSHEET
# personalized Steady state
phase_personalized_mean_rmse_ss = df['Phase Mean']['ML-Personalized-SteadyState']*100
phase_personalized_std_rmse_ss = df['Phase Std']['ML-Personalized-SteadyState']*100
speed_personalized_mean_rmse_ss = df['Speed Mean']['ML-Personalized-SteadyState']
speed_personalized_std_rmse_ss = df['Speed Std']['ML-Personalized-SteadyState']
incline_personalized_mean_rmse_ss = df['Incline Mean']['ML-Personalized-SteadyState']
incline_personalized_std_rmse_ss = df['Incline Std']['ML-Personalized-SteadyState']
stairs_personalized_mean_acc_ss = df['Stairs Mean']['ML-Personalized-SteadyState']*100
stairs_personalized_std_acc_ss = df['Stairs Std']['ML-Personalized-SteadyState']*100

# xval Steady state
phase_xval_mean_rmse_ss = df['Phase Mean']['ML-XVal-SteadyState']*100
phase_xval_std_rmse_ss = df['Phase Std']['ML-XVal-SteadyState']*100
speed_xval_mean_rmse_ss = df['Speed Mean']['ML-XVal-SteadyState']
speed_xval_std_rmse_ss = df['Speed Std']['ML-XVal-SteadyState']
incline_xval_mean_rmse_ss = df['Incline Mean']['ML-XVal-SteadyState']
incline_xval_std_rmse_ss = df['Incline Std']['ML-XVal-SteadyState']
stairs_xval_mean_acc_ss = df['Stairs Mean']['ML-XVal-SteadyState']*100
stairs_xval_std_acc_ss = df['Stairs Std']['ML-XVal-SteadyState']*100

# Gen Steady state
phase_gen_mean_rmse_ss = df['Phase Mean']['ML-Gen-SteadyState']*100
phase_gen_std_rmse_ss = df['Phase Std']['ML-Gen-SteadyState']*100
speed_gen_mean_rmse_ss = df['Speed Mean']['ML-Gen-SteadyState']
speed_gen_std_rmse_ss = df['Speed Std']['ML-Gen-SteadyState']
incline_gen_mean_rmse_ss = df['Incline Mean']['ML-Gen-SteadyState']
incline_gen_std_rmse_ss = df['Incline Std']['ML-Gen-SteadyState']
stairs_gen_mean_acc_ss = df['Stairs Mean']['ML-Gen-SteadyState']*100
stairs_gen_std_acc_ss = df['Stairs Std']['ML-Gen-SteadyState']*100

# EKF Steady state
phase_ekf_mean_rmse_ss = df['Phase Mean']['EKF-SteadyState']*100
phase_ekf_std_rmse_ss = df['Phase Std']['EKF-SteadyState']*100
speed_ekf_mean_rmse_ss = df['Speed Mean']['EKF-SteadyState']
speed_ekf_std_rmse_ss = df['Speed Std']['EKF-SteadyState']
incline_ekf_mean_rmse_ss = df['Incline Mean']['EKF-SteadyState']
incline_ekf_std_rmse_ss = df['Incline Std']['EKF-SteadyState']

# TBE Steady state
phase_tbe_mean_rmse_ss = df['Phase Mean']['TBE-SteadyState']*100
phase_tbe_std_rmse_ss = df['Phase Std']['TBE-SteadyState']*100

# aggregate results into vectors
phases = [phase_personalized_mean_rmse_ss, phase_xval_mean_rmse_ss, phase_gen_mean_rmse_ss, phase_ekf_mean_rmse_ss, phase_tbe_mean_rmse_ss]
phases_std = [phase_personalized_std_rmse_ss, phase_xval_std_rmse_ss, phase_gen_std_rmse_ss, phase_ekf_std_rmse_ss, phase_tbe_std_rmse_ss]

speeds = [speed_personalized_mean_rmse_ss, speed_xval_mean_rmse_ss, speed_gen_mean_rmse_ss, speed_ekf_mean_rmse_ss]
speeds_std = [speed_personalized_std_rmse_ss, speed_xval_std_rmse_ss, speed_gen_std_rmse_ss, speed_ekf_std_rmse_ss]

inclines = [incline_personalized_mean_rmse_ss, incline_xval_mean_rmse_ss, incline_gen_mean_rmse_ss, incline_ekf_mean_rmse_ss]
inclines_std = [incline_personalized_std_rmse_ss, incline_xval_std_rmse_ss, incline_gen_std_rmse_ss, incline_ekf_std_rmse_ss]

stairs = [stairs_personalized_mean_acc_ss, stairs_xval_mean_acc_ss, stairs_gen_mean_acc_ss]
stairs_std = [stairs_personalized_std_acc_ss, stairs_xval_std_acc_ss, stairs_gen_std_acc_ss]


conditions = ['ML\nInd.', 'ML\nDev.', 'ML\nGen.', 'EKF', 'TBE']
colors = [blueColor, purpleColor, greenColor, redColor, grayColor]
x_pos = np.arange(len(conditions))

# print(phases)
# print(phases_std)
# print(conditions)
# print(colors)
# print(x_pos[:5])

fig, axs = plt.subplots(1,4,figsize=(figWidth,figHeight))
axs[0].bar(x_pos, phases, yerr=phases_std, color=colors, align='center', ecolor='black')
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(conditions)
axs[0].set_ylabel('Phase RMSE (%)', fontsize=MEDIUM_SIZE)


axs[1].bar(x_pos[:4], speeds, yerr=speeds_std, color=colors[:4], align='center', ecolor='black')
axs[1].set_xticks(x_pos[:4])
axs[1].set_xticklabels(conditions[:4])
axs[1].set_ylabel('Speed RMSE (m/s)', fontsize=MEDIUM_SIZE)

axs[2].bar(x_pos[:4], inclines, yerr=inclines_std, color=colors[:4], align='center', ecolor='black')
axs[2].set_xticks(x_pos[:4])
axs[2].set_xticklabels(conditions[:4])
axs[2].set_ylabel('Incline RMSE (deg)', fontsize=MEDIUM_SIZE)


axs[3].bar(x_pos[:3], stairs, yerr=stairs_std, color=colors[:3], align='center', ecolor='black')
axs[3].set_xticks(x_pos[:3])
axs[3].set_xticklabels(conditions[:3])
axs[3].set_ylabel('Is Stairs Accuracy (%)', fontsize=MEDIUM_SIZE)

fig.subplots_adjust(wspace=1.0)

fig.text(0.06,1.0,'(a)', fontsize=MEDIUM_SIZE)
fig.text(0.31,1.0,'(b)', fontsize=MEDIUM_SIZE)
fig.text(0.56,1.0,'(c)', fontsize=MEDIUM_SIZE)
fig.text(0.81,1.0,'(d)', fontsize=MEDIUM_SIZE)




for i in range(4):
	ax = axs[i]
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

plt.tight_layout()


filename = f'results_barchart_steadystate_raw.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'results_barchart_steadystate_raw.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')



plt.show()
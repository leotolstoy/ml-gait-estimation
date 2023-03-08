""" Simulates the phase estimator ekf using loaded data. """
import numpy as np
import os, sys
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import sys
thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')
# print(sys.path)
import matplotlib
# matplotlib.use('Agg')
from filter_classes import FirstOrderLowPassLinearFilter
import matplotlib.pyplot as plt

import pandas as pd


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


def process_file(ekf_filename, vicon_filename, time_vicon_offset,  DO_PLOTS=False):

	data = np.loadtxt(ekf_filename, delimiter=',')
	timeSec_vec=data[:,0]
	accelVec_corrected=data[:,1:4]
	gyroVec_corrected=data[:,4:7]
	# x_state_AE = data[:,10:16]

	ankleAngle = data[:,44]
	isOverriding_hardware_vec = data[:,73]
	roll = data[:,58]
	pitch = data[:,59]
	yaw = data[:,60]

	x_state_PE = data[:,25:29]
	z_measured_act = data[:,29:35]
	z_model_act = data[:,35:41]
	HSDetected_hardware_vec = data[:,24]
	strideLength_descaled_vec = data[:,45]

	heelAccForward_meas = data[:,61] #92
	heelPosForward_meas_filt = data[:,62] #93
	heelPosUp_meas_filt = data[:,63] #93

	actTorque_hardware_vec = data[:,49]
	desTorque_hardware_vec = data[:,50]


	heelAccSide_meas = data[:,68] #70
	heelAccUp_meas = data[:,69] #71

	heelAccForward_meas_fromDeltaVelocity = data[:,70] #92
	heelAccSide_meas_fromDeltaVelocity = data[:,71] #70
	heelAccUp_meas_fromDeltaVelocity = data[:,72]#71

	# Load in HS data
	df_vicon = pd.read_csv(vicon_filename)

	time_vicon = df_vicon['time'].to_numpy()
	phase_vicon = df_vicon['phase'].to_numpy()
	phase_rate_vicon = df_vicon['phase_rate'].to_numpy()
	stride_length_vicon = df_vicon['stride_length'].to_numpy()
	incline_vicon = df_vicon['incline'].to_numpy()
	RTOE_X_vicon = df_vicon['RTOE_X'].to_numpy()
	HS_times_vicon = df_vicon['HS_vicon'].to_numpy()
	notnanidxs = ~np.isnan(HS_times_vicon)
	HS_times_vicon = HS_times_vicon[notnanidxs]

	#construct heelstrike boolean vector that is 1 where a HS happened
	t_vicon_HS = np.arange(HS_times_vicon[0],HS_times_vicon[-1],0.01)
	# print(t_vicon_HS)
	HSDetected_vicon = np.zeros(t_vicon_HS.shape)
	for i in range(len(HSDetected_vicon)):
		diff = np.abs(t_vicon_HS[i] - HS_times_vicon)
		if np.any(diff < 1e-4):
			HSDetected_vicon[i] = 1
	# print(HS_times_vicon)

	HS_times_vicon += time_vicon_offset


	speed_vicon = phase_rate_vicon * stride_length_vicon

	# From vicon, extract ground truth
	phase_ground_truth = np.interp(timeSec_vec,time_vicon+time_vicon_offset, phase_vicon)
	speed_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, speed_vicon)
	incline_ground_truth = np.interp(timeSec_vec,time_vicon + time_vicon_offset, incline_vicon)
	HSDetected_ground_truth = np.round(np.interp(timeSec_vec,t_vicon_HS + time_vicon_offset, HSDetected_vicon))

	#find where a HS just happened
	HS_idxs = np.diff(HSDetected_ground_truth)
	HS_falling_edge = HS_idxs == -1
	HS_falling_edge = np.insert(HS_falling_edge, 0, HS_falling_edge[0])
	
	#extract idxs between first and last idxs
	idxs = np.argwhere(HS_falling_edge).flatten()
	# print(idxs)
	# idxs = idxs[0]:idxs[-1]

	#update data vector
	
	timeSec_vec = timeSec_vec[idxs[0]:idxs[-1]]
	phase_ground_truth = phase_ground_truth[idxs[0]:idxs[-1]]
	HSDetected_ground_truth = HSDetected_ground_truth[idxs[0]:idxs[-1]]
	speed_ground_truth = speed_ground_truth[idxs[0]:idxs[-1]]
	incline_ground_truth = incline_ground_truth[idxs[0]:idxs[-1]]

	#re-zero time vector plus offset by 0.01 (average refresh rate)
	timeSec_vec = timeSec_vec - timeSec_vec[0] + 0.01

	#create vector of time steps
	dts = np.diff(timeSec_vec)
	dts = np.insert(dts, 0, 0.01)
	print(np.mean(dts))

	foot_angles = z_measured_act[idxs[0]:idxs[-1],0]
	foot_vel_angles = z_measured_act[idxs[0]:idxs[-1],1]
	shank_angles = z_measured_act[idxs[0]:idxs[-1],2]
	shank_vel_angles = z_measured_act[idxs[0]:idxs[-1],3]
	heel_acc_forward = heelAccForward_meas_fromDeltaVelocity[idxs[0]:idxs[-1]]
	heel_acc_up = heelAccUp_meas_fromDeltaVelocity[idxs[0]:idxs[-1]]

	#filter accelerations
	heel_forward_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)
	heel_up_acc_filter = FirstOrderLowPassLinearFilter(fc=5, dt=0.01)

	
	heel_acc_forward_filt = np.zeros(heel_acc_forward.shape)
	heel_acc_up_filt = np.zeros(heel_acc_up.shape)

	for i in range(len(heel_acc_forward_filt)):
		heel_acc_forward_filt[i] = heel_forward_acc_filter.step(i, heel_acc_forward[i])
		heel_acc_up_filt[i] = heel_up_acc_filter.step(i, heel_acc_up[i])

	heel_acc_forward = heel_acc_forward_filt
	heel_acc_up = heel_acc_up_filt

	if DO_PLOTS:
		fig, axs = plt.subplots(3,1)
		axs[0].plot(timeSec_vec, phase_ground_truth)
		axs[0].plot(timeSec_vec, HSDetected_ground_truth, 'k')
		axs[1].plot(timeSec_vec, speed_ground_truth)
		axs[2].plot(timeSec_vec, incline_ground_truth)



		fig1, axs1 = plt.subplots(4,1)
		axs1[0].plot(timeSec_vec, foot_angles)
		axs1[0].plot(timeSec_vec, HSDetected_ground_truth*10, 'k')
		axs1[1].plot(timeSec_vec, foot_vel_angles)
		axs1[2].plot(timeSec_vec, shank_angles)
		axs1[3].plot(timeSec_vec, shank_vel_angles)

		fig2, axs2 = plt.subplots()
		axs2.hist(dts)


	# plt.show()


	return timeSec_vec, phase_ground_truth, speed_ground_truth, incline_ground_truth,\
		foot_angles, foot_vel_angles, shank_angles, shank_vel_angles,\
		heel_acc_forward, heel_acc_up,\
		dts

def generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS):


	time_vicon_offset_P1 = trialA_dict['offset']
	time_vicon_offset_P2 = trialB_dict['offset']
	time_vicon_offset_P3 = trialC_dict['offset']

	ekf_filename_A = trialA_dict['ekf_filename']
	ekf_filename_B = trialB_dict['ekf_filename']
	ekf_filename_C = trialC_dict['ekf_filename']

	vicon_processed_filename_A = trialA_dict['vicon_filename']
	vicon_processed_filename_B = trialB_dict['vicon_filename']
	vicon_processed_filename_C = trialC_dict['vicon_filename']


	timeSec_vec_P1, phase_ground_truth_P1, speed_ground_truth_P1, incline_ground_truth_P1, \
		foot_angles_P1, foot_vel_angles_P1, shank_angles_P1, shank_vel_angles_P1, \
		heel_acc_forward_P1, heel_acc_up_P1, dts_P1 = process_file(ekf_filename_A, vicon_processed_filename_A, time_vicon_offset_P1,DO_PLOTS=DO_PLOTS)

	# input()
	timeSec_vec_P2, phase_ground_truth_P2, speed_ground_truth_P2, incline_ground_truth_P2,\
		foot_angles_P2, foot_vel_angles_P2, shank_angles_P2, shank_vel_angles_P2, \
		heel_acc_forward_P2, heel_acc_up_P2, dts_P2 = process_file(ekf_filename_B,vicon_processed_filename_B, time_vicon_offset_P2,DO_PLOTS=DO_PLOTS)

	timeSec_vec_P3, phase_ground_truth_P3, speed_ground_truth_P3, incline_ground_truth_P3,\
		foot_angles_P3, foot_vel_angles_P3, shank_angles_P3, shank_vel_angles_P3, \
		heel_acc_forward_P3, heel_acc_up_P3, dts_P3 = process_file(ekf_filename_C,vicon_processed_filename_C, time_vicon_offset_P3,DO_PLOTS=DO_PLOTS)

	phase_ground_truth = np.concatenate((phase_ground_truth_P1, phase_ground_truth_P2, phase_ground_truth_P3)).reshape(-1,1)
	speed_ground_truth = np.concatenate((speed_ground_truth_P1, speed_ground_truth_P2, speed_ground_truth_P3)).reshape(-1,1)
	incline_ground_truth = np.concatenate((incline_ground_truth_P1, incline_ground_truth_P2, incline_ground_truth_P3)).reshape(-1,1)
	foot_angles = np.concatenate((foot_angles_P1, foot_angles_P2, foot_angles_P3)).reshape(-1,1)
	foot_vel_angles = np.concatenate((foot_vel_angles_P1, foot_vel_angles_P2, foot_vel_angles_P3)).reshape(-1,1)
	shank_angles = np.concatenate((shank_angles_P1, shank_angles_P2, shank_angles_P3)).reshape(-1,1)
	shank_vel_angles = np.concatenate((shank_vel_angles_P1, shank_vel_angles_P2, shank_vel_angles_P3)).reshape(-1,1)
	heel_acc_forward = np.concatenate((heel_acc_forward_P1, heel_acc_forward_P2, heel_acc_forward_P3)).reshape(-1,1)
	heel_acc_up = np.concatenate((heel_acc_up_P1, heel_acc_up_P2, heel_acc_up_P3)).reshape(-1,1)
	dts = np.concatenate((dts_P1, dts_P2, dts_P3)).reshape(-1,1)

	#update time vectors to form one continuous time vector
	timeSec_vec_P2 += timeSec_vec_P1[-1]
	timeSec_vec_P3 += timeSec_vec_P2[-1]

	timeSec_vec = np.concatenate((timeSec_vec_P1, timeSec_vec_P2, timeSec_vec_P3)).reshape(-1,1)

	

	fig, axs = plt.subplots(3,1)
	axs[0].plot(timeSec_vec, phase_ground_truth)
	axs[1].plot(timeSec_vec, speed_ground_truth)
	axs[2].plot(timeSec_vec, incline_ground_truth)


	fig1, axs1 = plt.subplots(4,1, sharex=True)
	axs1[0].plot(timeSec_vec, foot_angles,label='foot angles')
	axs1[1].plot(timeSec_vec, foot_vel_angles)
	axs1[2].plot(timeSec_vec, shank_angles)
	axs1[3].plot(timeSec_vec, shank_vel_angles)


	fig2, axs2 = plt.subplots(2,1, sharex=True)
	axs2[0].plot(phase_ground_truth, heel_acc_forward, '.',label='heel_acc_forward')
	axs2[1].plot(phase_ground_truth, heel_acc_up, '.',label='heel_acc_up')
	axs2[0].legend()
	axs2[1].legend()

	fig3, axs3 = plt.subplots(2,1, sharex=True)
	axs3[0].plot(timeSec_vec, heel_acc_forward,label='heel_acc_forward')
	axs3[1].plot(timeSec_vec, heel_acc_up,label='heel_acc_up')
	axs3[0].legend()
	axs3[1].legend()
	
	print(timeSec_vec.shape)
	print(dts.shape)
	is_stairs = np.zeros(timeSec_vec.shape)

	data = np.hstack((foot_angles, foot_vel_angles, shank_angles, shank_vel_angles,\
		heel_acc_forward, heel_acc_up, dts,\
		phase_ground_truth, speed_ground_truth, incline_ground_truth, is_stairs))

	columns = ['foot_angles', 'foot_vel_angles', 'shank_angles', 'shank_vel_angles', \
		'heel_acc_forward', 'heel_acc_up', 'dt',\
		'phase_ground_truth', 'speed_ground_truth', 'incline_ground_truth', 'is_stairs']

	df = pd.DataFrame(data, columns=columns)
	print(df.head())

	df.to_csv(export_filename, index=None)

	plt.show()

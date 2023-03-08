""" Simulates the phase estimator ekf using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd

from analyze_subject import phase_dist, process_file, generate_subj_dataset


if __name__ == '__main__':
	time_vicon_offset_P1 = 4.387-0.566
	time_vicon_offset_P2 = 8.438
	time_vicon_offset_P3 = 7.637
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-21_AE3831Standalone_PB_EKF_Test_AB04_TPVA_02.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB04TPVA_02_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-21_AE4543Standalone_PB_EKF_Test_AB04_TPVB_03.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB04TPVB_03_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-22_AE0639Standalone_PB_EKF_Test_AB04_TPVC_04.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB04TPVC_04_processed.csv'

	export_filename = 'exoboot_AB04_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)



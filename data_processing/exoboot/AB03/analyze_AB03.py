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
	time_vicon_offset_P1 = 7.25 - 0.245
	time_vicon_offset_P2 = 6.547
	time_vicon_offset_P3 = 9.072
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-20_AE5407Standalone_PB_EKF_Test_AB03_TPVA_01.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB03TPVA_01_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-21_AE0146Standalone_PB_EKF_Test_AB03_TPVB_02.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB03TPVB_02_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-21_AE0939Standalone_PB_EKF_Test_AB03_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB03TPVC_03_processed.csv'

	export_filename = 'exoboot_AB03_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)

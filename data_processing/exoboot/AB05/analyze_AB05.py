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
	time_vicon_offset_P1 = 6.807
	time_vicon_offset_P2 = 22.81
	time_vicon_offset_P3 = 11.964
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-19_AE5626Standalone_PB_EKF_Test_AB05_TPVA_03.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB05TPVA_03_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-20_AE0058Standalone_PB_EKF_Test_AB05_TPVB_04.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB05TPVB_04_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-20_AE0532Standalone_PB_EKF_Test_AB05_TPVC_05.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB05TPVC_05_processed.csv'

	export_filename = 'exoboot_AB05_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)



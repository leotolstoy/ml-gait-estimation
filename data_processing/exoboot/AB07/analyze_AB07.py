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
	time_vicon_offset_P1 = 4.926
	time_vicon_offset_P2 = 7.61 - 0.199
	time_vicon_offset_P3 = 7.884 - 1.373
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220915-19_AE2742Standalone_PB_EKF_Test_AB07_TPVA_01.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB07TPVA_01_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220915-19_AE4151Standalone_PB_EKF_Test_AB07_TPVB_02.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB07TPVB_02_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220915-19_AE4720Standalone_PB_EKF_Test_AB07_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB07TPVC_03_processed.csv'

	export_filename = 'exoboot_AB07_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)


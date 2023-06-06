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
	time_vicon_offset_P1 = 9.33 - 0.658
	time_vicon_offset_P2 = 8.379
	time_vicon_offset_P3 = 9.114
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-22_AE5322Standalone_PB_EKF_Test_AB02_TPVA_01.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB02TPVA_01_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-22_AE5544Standalone_PB_EKF_Test_AB02_TPVB_02.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB02TPVB_02_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-23_AE0107Standalone_PB_EKF_Test_AB02_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB02TPVC_03_processed.csv'

	export_filename = 'exoboot_AB02_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)



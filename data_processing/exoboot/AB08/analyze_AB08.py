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
	time_vicon_offset_P1 = 5.591 - 1.325
	time_vicon_offset_P2 = 5.858 - 1.354
	time_vicon_offset_P3 = 5.237 - 1.078
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220915-20_AE2625Standalone_PB_EKF_Test_AB08_TPVA_01.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB08TPVA_01_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220915-20_AE3803Standalone_PB_EKF_Test_AB08_TPVB_02.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB08TPVB_02_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220915-20_AE5133Standalone_PB_EKF_Test_AB08_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB08TPVC_03_processed.csv'

	export_filename = 'exoboot_AB08_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)


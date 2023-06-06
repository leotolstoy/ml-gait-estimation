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
	time_vicon_offset_P1 = 10.375-1.454
	time_vicon_offset_P2 = 15.216
	time_vicon_offset_P3 = 10.11
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-20_AE2833Standalone_PB_EKF_Test_AB06_TPVA_01.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB06TPVA_01_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-20_AE3515Standalone_PB_EKF_Test_AB06_TPVB_02.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB06TPVB_02_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-20_AE4055Standalone_PB_EKF_Test_AB06_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB06TPVC_03_processed.csv'

	export_filename = 'exoboot_AB06_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)




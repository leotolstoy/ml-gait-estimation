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
	time_vicon_offset_P1 = 13.37-0.583
	time_vicon_offset_P2 = 0.159 + 10.575
	time_vicon_offset_P3 = 8.447
	DO_PLOTS=not True

	trialA_dict = {}
	trialB_dict = {}
	trialC_dict = {}

	trialA_dict['offset'] = time_vicon_offset_P1
	trialA_dict['ekf_filename'] = "20220404-22_AE4113Standalone_PB_EKF_Test_AB01_TPVA_05.csv"
	trialA_dict['vicon_filename'] = 'Vicon_AB01TPVA_05_processed.csv'

	trialB_dict['offset'] = time_vicon_offset_P2
	trialB_dict['ekf_filename'] = "20220404-22_AE3639Standalone_PB_EKF_Test_AB01_TPVB_04.csv"
	trialB_dict['vicon_filename'] = 'Vicon_AB01TPVB_04_processed.csv'

	trialC_dict['offset'] = time_vicon_offset_P3
	trialC_dict['ekf_filename'] = "20220404-22_AE2733Standalone_PB_EKF_Test_AB01_TPVC_03.csv"
	trialC_dict['vicon_filename'] = 'Vicon_AB01TPVC_03_processed.csv'

	export_filename = 'exoboot_AB01_data.csv'

	generate_subj_dataset(trialA_dict, trialB_dict, trialC_dict, export_filename, DO_PLOTS)


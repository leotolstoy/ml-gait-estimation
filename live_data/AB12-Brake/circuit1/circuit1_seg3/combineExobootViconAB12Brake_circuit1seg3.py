
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os, sys

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')

from process_vicon_data_utils import combineExobootViconData

if __name__ == '__main__':

    exoboot_filename = '20230307-19_AB12-Frame_circuit1_seg3.csv'
    vicon_filename = 'Vicon_AB12-Brake_circuit1_seg3_processed.csv'
    export_filename = 'exoboot_' + vicon_filename

    time_vicon_offset = -1.03+0.011318
    combineExobootViconData(exoboot_filename, vicon_filename, time_vicon_offset, export_filename=export_filename, DO_PLOTS= True)

















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

    exoboot_filename = '20230309-18_AB08-CHaos_circuit3_seg4.csv'
    vicon_filename = 'Vicon_AB08-CHaos_circuit3_seg4_processed.csv'
    export_filename = 'exoboot_' + vicon_filename

    time_vicon_offset = -0.964
    combineExobootViconData(exoboot_filename, vicon_filename, time_vicon_offset, export_filename=export_filename, DO_PLOTS= True)

















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

from process_vicon_data_utils import process_file_for_finetuning

if __name__ == '__main__':
    filename = 'exoboot_Vicon_AB01-Scarlet_stairs1_processed.csv'
    export_filename = 'finetune_'+filename

    process_file_for_finetuning(filename, export_filename)

    
















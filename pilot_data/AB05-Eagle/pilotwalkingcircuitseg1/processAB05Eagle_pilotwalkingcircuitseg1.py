
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

from process_vicon_data_utils import returnViconProcessed

if __name__ == '__main__':

    filename = 'pilotwalkingcircuitseg1.csv'

    AB_SUBJECT = 'AB05-Eagle'

    # Load in HS data
    nrows_events = 24 #row number of the last HS row
    n_skiprows_marker_1 = 28# row num that contains AB0's:
    nrows = 3392 #=  num frames, alsolast idx - row with frame subframe

    returnViconProcessed(filename, AB_SUBJECT, nrows_events,n_skiprows_marker_1,nrows,DO_PLOTS=True,DO_EXPORT=True)














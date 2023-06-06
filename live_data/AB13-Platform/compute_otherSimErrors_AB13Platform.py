import os, sys

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')

from sim_other_models import sim_other_models_over_multiple_files


filenames = [
            'circuit1/circuit1_seg1/20230316-16_AB13-Platform_circuit1_seg1.csv',
            'circuit1/circuit1_seg2/20230316-16_AB13-Platform_circuit1_seg2.csv',
            'circuit1/circuit1_seg3/20230316-16_AB13-Platform_circuit1_seg3.csv',
            'circuit1/circuit1_seg4/20230316-16_AB13-Platform_circuit1_seg4.csv',
            'circuit2/circuit2_seg1/20230316-16_AB13-Platform_circuit2_seg1.csv',
            'circuit2/circuit2_seg2/20230316-16_AB13-Platform_circuit2_seg2.csv',
            'circuit2/circuit2_seg3/20230316-16_AB13-Platform_circuit2_seg3.csv',
            'circuit2/circuit2_seg4/20230316-16_AB13-Platform_circuit2_seg4.csv',
            'circuit3/circuit3_seg1/20230316-16_AB13-Platform_circuit3_seg1.csv',
            'circuit3/circuit3_seg2/20230316-16_AB13-Platform_circuit3_seg2.csv',
            'circuit3/circuit3_seg3/20230316-16_AB13-Platform_circuit3_seg3.csv',
            'circuit3/circuit3_seg4/20230316-16_AB13-Platform_circuit3_seg4.csv'
            ]

vicon_filenames = [
            'circuit1/circuit1_seg1/exoboot_Vicon_AB13-Platform_circuit1_seg1_processed.csv',
            'circuit1/circuit1_seg2/exoboot_Vicon_AB13-Platform_circuit1_seg2_processed.csv',
            'circuit1/circuit1_seg3/exoboot_Vicon_AB13-Platform_circuit1_seg3_processed.csv',
            'circuit1/circuit1_seg4/exoboot_Vicon_AB13-Platform_circuit1_seg4_processed.csv',
            'circuit2/circuit2_seg1/exoboot_Vicon_AB13-Platform_circuit2_seg1_processed.csv',
            'circuit2/circuit2_seg2/exoboot_Vicon_AB13-Platform_circuit2_seg2_processed.csv',
            'circuit2/circuit2_seg3/exoboot_Vicon_AB13-Platform_circuit2_seg3_processed.csv',
            'circuit2/circuit2_seg4/exoboot_Vicon_AB13-Platform_circuit2_seg4_processed.csv',
            'circuit3/circuit3_seg1/exoboot_Vicon_AB13-Platform_circuit3_seg1_processed.csv',
            'circuit3/circuit3_seg2/exoboot_Vicon_AB13-Platform_circuit3_seg2_processed.csv',
            'circuit3/circuit3_seg3/exoboot_Vicon_AB13-Platform_circuit3_seg3_processed.csv',
            'circuit3/circuit3_seg4/exoboot_Vicon_AB13-Platform_circuit3_seg4_processed.csv'
            ]

sim_other_models_over_multiple_files(filenames=filenames, vicon_filenames=vicon_filenames)


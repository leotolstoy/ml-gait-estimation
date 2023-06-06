import os, sys

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/')
sys.path.append('/Users/leo/Documents/Research/Exo/ML Gait Estimation/ml-gait-estimation/utils')

from sim_other_models import sim_other_models_over_multiple_files


filenames = [
            'circuit1/circuit1_seg1/20230222-23_AE0407_gait_transformerC1S1.csv',
            'circuit1/circuit1_seg2/20230222-23_AE0608_gait_transformerC1S2.csv',
            'circuit1/circuit1_seg3/20230222-23_AE0729_gait_transformerC1S3.csv',
            'circuit1/circuit1_seg4/20230222-23_AE0818_gait_transformerC1S4.csv',
            'circuit2/circuit2_seg1/20230222-23_AE0916_gait_transformerC2S1.csv',
            'circuit2/circuit2_seg2/20230222-23_AE1017_gait_transformerC2S2.csv',
            'circuit2/circuit2_seg3/20230222-23_AE1226_gait_transformerC2S3.csv',
            'circuit2/circuit2_seg4/20230222-23_AE1317_gait_transformerC2S4.csv',
            'circuit3/circuit3_seg1/20230222-23_AE1422_gait_transformerC3S1.csv',
            'circuit3/circuit3_seg2/20230222-23_AE1522_gait_transformerC3S2.csv',
            'circuit3/circuit3_seg3/20230222-23_AE1628_gait_transformerC3S3.csv',
            'circuit3/circuit3_seg4/20230222-23_AE1734_gait_transformerC3S4.csv',
            ]

vicon_filenames = [
            'circuit1/circuit1_seg1/exoboot_Vicon_AB05-Eagle_circuit1_seg1_processed.csv',
            'circuit1/circuit1_seg2/exoboot_Vicon_AB05-Eagle_circuit1_seg2_processed.csv',
            'circuit1/circuit1_seg3/exoboot_Vicon_AB05-Eagle_circuit1_seg3_processed.csv',
            'circuit1/circuit1_seg4/exoboot_Vicon_AB05-Eagle_circuit1_seg4_processed.csv',
            'circuit2/circuit2_seg1/exoboot_Vicon_AB05-Eagle_circuit2_seg1_processed.csv',
            'circuit2/circuit2_seg2/exoboot_Vicon_AB05-Eagle_circuit2_seg2_processed.csv',
            'circuit2/circuit2_seg3/exoboot_Vicon_AB05-Eagle_circuit2_seg3_processed.csv',
            'circuit2/circuit2_seg4/exoboot_Vicon_AB05-Eagle_circuit2_seg4_processed.csv',
            'circuit3/circuit3_seg1/exoboot_Vicon_AB05-Eagle_circuit3_seg1_processed.csv',
            'circuit3/circuit3_seg2/exoboot_Vicon_AB05-Eagle_circuit3_seg2_processed.csv',
            'circuit3/circuit3_seg3/exoboot_Vicon_AB05-Eagle_circuit3_seg3_processed.csv',
            'circuit3/circuit3_seg4/exoboot_Vicon_AB05-Eagle_circuit3_seg4_processed.csv'
            ]
sim_other_models_over_multiple_files(filenames=filenames, vicon_filenames=vicon_filenames)




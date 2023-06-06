"""This script conducts an ANOVA and tukey HSD test to check the significance of the transitory trials
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import f_oneway
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#HISTOGRAM


filename = "live_data/subj_results.xlsx"
df = pd.read_excel(filename, sheet_name='ML',index_col=0, engine='openpyxl')

# print(df.head())

# print(df['Steady State Error']['AB01'])

#load ML+EKF data
phases_personalized = [df['Transitory Error']['AB01'],
                       df['Transitory Error']['AB02'],
                       df['Transitory Error']['AB03'],
                       df['Transitory Error']['AB04'],
                       df['Transitory Error']['AB05'],
                       df['Transitory Error']['AB06'],
                       df['Transitory Error']['AB07'],
                       df['Transitory Error']['AB08'],
                       df['Transitory Error']['AB09'],
                       df['Transitory Error']['AB10']
                       ]

speeds_personalized = [df['Unnamed: 19']['AB01'],
                       df['Unnamed: 19']['AB02'],
                       df['Unnamed: 19']['AB03'],
                       df['Unnamed: 19']['AB04'],
                       df['Unnamed: 19']['AB05'],
                       df['Unnamed: 19']['AB06'],
                       df['Unnamed: 19']['AB07'],
                       df['Unnamed: 19']['AB08'],
                       df['Unnamed: 19']['AB09'],
                       df['Unnamed: 19']['AB10']
                       ]

# print(speeds_personalized)
# input()

inclines_personalized = [df['Unnamed: 21']['AB01'],
                       df['Unnamed: 21']['AB02'],
                       df['Unnamed: 21']['AB03'],
                       df['Unnamed: 21']['AB04'],
                       df['Unnamed: 21']['AB05'],
                       df['Unnamed: 21']['AB06'],
                       df['Unnamed: 21']['AB07'],
                       df['Unnamed: 21']['AB08'],
                       df['Unnamed: 21']['AB09'],
                       df['Unnamed: 21']['AB10']
                       ]

stairs_personalized = [df['Unnamed: 23']['AB01'],
                       df['Unnamed: 23']['AB02'],
                       df['Unnamed: 23']['AB03'],
                       df['Unnamed: 23']['AB04'],
                       df['Unnamed: 23']['AB05'],
                       df['Unnamed: 23']['AB06'],
                       df['Unnamed: 23']['AB07'],
                       df['Unnamed: 23']['AB08'],
                       df['Unnamed: 23']['AB09'],
                       df['Unnamed: 23']['AB10']
                       ]

# print(stairs_personalized)
# input()
phases_xval = [df['Transitory Error']['AB11'],
                       df['Transitory Error']['AB12'],
                       df['Transitory Error']['AB13'],
                       df['Transitory Error']['AB14']
                       ]

speeds_xval = [df['Unnamed: 19']['AB11'],
                       df['Unnamed: 19']['AB12'],
                       df['Unnamed: 19']['AB13'],
                       df['Unnamed: 19']['AB14']
                       ]

inclines_xval = [df['Unnamed: 21']['AB11'],
                       df['Unnamed: 21']['AB12'],
                       df['Unnamed: 21']['AB13'],
                       df['Unnamed: 21']['AB14']
                       ]

stairs_xval = [df['Unnamed: 23']['AB11'],
                       df['Unnamed: 23']['AB12'],
                       df['Unnamed: 23']['AB13'],
                       df['Unnamed: 23']['AB14']
                       ]

# print(inclines_xval)


# load gen data
df = pd.read_excel(filename, sheet_name='GenML',index_col=0, engine='openpyxl')

# print(df.head())


phases_gen = [df['Transitory Error']['AB01'],
                       df['Transitory Error']['AB02'],
                       df['Transitory Error']['AB03'],
                       df['Transitory Error']['AB04'],
                       df['Transitory Error']['AB05'],
                       df['Transitory Error']['AB06'],
                       df['Transitory Error']['AB07'],
                       df['Transitory Error']['AB08'],
                       df['Transitory Error']['AB09'],
                       df['Transitory Error']['AB10'],
                       df['Transitory Error']['AB11'],
                       df['Transitory Error']['AB12'],
                       df['Transitory Error']['AB13'],
                       df['Transitory Error']['AB14']
                       ]

speeds_gen = [df['Unnamed: 19']['AB01'],
                       df['Unnamed: 19']['AB02'],
                       df['Unnamed: 19']['AB03'],
                       df['Unnamed: 19']['AB04'],
                       df['Unnamed: 19']['AB05'],
                       df['Unnamed: 19']['AB06'],
                       df['Unnamed: 19']['AB07'],
                       df['Unnamed: 19']['AB08'],
                       df['Unnamed: 19']['AB09'],
                       df['Unnamed: 19']['AB10'],
                       df['Unnamed: 19']['AB11'],
                       df['Unnamed: 19']['AB12'],
                       df['Unnamed: 19']['AB13'],
                       df['Unnamed: 19']['AB14']
                       ]

inclines_gen = [df['Unnamed: 21']['AB01'],
                       df['Unnamed: 21']['AB02'],
                       df['Unnamed: 21']['AB03'],
                       df['Unnamed: 21']['AB04'],
                       df['Unnamed: 21']['AB05'],
                       df['Unnamed: 21']['AB06'],
                       df['Unnamed: 21']['AB07'],
                       df['Unnamed: 21']['AB08'],
                       df['Unnamed: 21']['AB09'],
                       df['Unnamed: 21']['AB10'],
                       df['Unnamed: 21']['AB11'],
                       df['Unnamed: 21']['AB12'],
                       df['Unnamed: 21']['AB13'],
                       df['Unnamed: 21']['AB14']
                       ]

stairs_gen = [df['Unnamed: 23']['AB01'],
                       df['Unnamed: 23']['AB02'],
                       df['Unnamed: 23']['AB03'],
                       df['Unnamed: 23']['AB04'],
                       df['Unnamed: 23']['AB05'],
                       df['Unnamed: 23']['AB06'],
                       df['Unnamed: 23']['AB07'],
                       df['Unnamed: 23']['AB08'],
                       df['Unnamed: 23']['AB09'],
                       df['Unnamed: 23']['AB10'],
                       df['Unnamed: 23']['AB11'],
                       df['Unnamed: 23']['AB12'],
                       df['Unnamed: 23']['AB13'],
                       df['Unnamed: 23']['AB14']
                       ]

# print(stairs_gen)
# input()

#load EKF
df = pd.read_excel(filename, sheet_name='EKF',index_col=0, engine='openpyxl')
# print(df.head())
phases_ekf = [df['Transitory Error']['AB01'],
                       df['Transitory Error']['AB02'],
                       df['Transitory Error']['AB03'],
                       df['Transitory Error']['AB04'],
                       df['Transitory Error']['AB05'],
                       df['Transitory Error']['AB06'],
                       df['Transitory Error']['AB07'],
                       df['Transitory Error']['AB08'],
                       df['Transitory Error']['AB09'],
                       df['Transitory Error']['AB10'],
                       df['Transitory Error']['AB11'],
                       df['Transitory Error']['AB12'],
                       df['Transitory Error']['AB13'],
                       df['Transitory Error']['AB14']
                       ]

speeds_ekf = [df['Unnamed: 15']['AB01'],
                       df['Unnamed: 15']['AB02'],
                       df['Unnamed: 15']['AB03'],
                       df['Unnamed: 15']['AB04'],
                       df['Unnamed: 15']['AB05'],
                       df['Unnamed: 15']['AB06'],
                       df['Unnamed: 15']['AB07'],
                       df['Unnamed: 15']['AB08'],
                       df['Unnamed: 15']['AB09'],
                       df['Unnamed: 15']['AB10'],
                       df['Unnamed: 15']['AB11'],
                       df['Unnamed: 15']['AB12'],
                       df['Unnamed: 15']['AB13'],
                       df['Unnamed: 15']['AB14']
                       ]

inclines_ekf = [df['Unnamed: 17']['AB01'],
                       df['Unnamed: 17']['AB02'],
                       df['Unnamed: 17']['AB03'],
                       df['Unnamed: 17']['AB04'],
                       df['Unnamed: 17']['AB05'],
                       df['Unnamed: 17']['AB06'],
                       df['Unnamed: 17']['AB07'],
                       df['Unnamed: 17']['AB08'],
                       df['Unnamed: 17']['AB09'],
                       df['Unnamed: 17']['AB10'],
                       df['Unnamed: 17']['AB11'],
                       df['Unnamed: 17']['AB12'],
                       df['Unnamed: 17']['AB13'],
                       df['Unnamed: 17']['AB14']
                       ]

# print(inclines_ekf)

#load in TBE
df = pd.read_excel(filename, sheet_name='TBE',index_col=0, engine='openpyxl')
phases_tbe = [df['Transitory Error']['AB01'],
                       df['Transitory Error']['AB02'],
                       df['Transitory Error']['AB03'],
                       df['Transitory Error']['AB04'],
                       df['Transitory Error']['AB05'],
                       df['Transitory Error']['AB06'],
                       df['Transitory Error']['AB07'],
                       df['Transitory Error']['AB08'],
                       df['Transitory Error']['AB09'],
                       df['Transitory Error']['AB10'],
                       df['Transitory Error']['AB11'],
                       df['Transitory Error']['AB12'],
                       df['Transitory Error']['AB13'],
                       df['Transitory Error']['AB14']
                       ]

# print(phases_tbe)


# Phase significance testing
print('PHASE RESULTS')
F_phase, p_phase = f_oneway(phases_personalized, phases_xval, phases_gen, phases_ekf, phases_tbe)
print(f'F: {F_phase}, p: {p_phase}')

# result = scipy.stats.tukey_hsd(phases_personalized, phases_xval, phases_gen, phases_ekf, phases_tbe)
result_phase = pairwise_tukeyhsd(endog=np.array(phases_personalized+phases_xval+phases_gen+phases_ekf+phases_tbe),
                           groups = np.array(['Personalized']*len(phases_personalized) + ['Xval']*len(phases_xval) +
                                             ['Gen']*len(phases_gen) + ['EKF']*len(phases_ekf) + ['TBE']*len(phases_tbe)),
                            alpha=0.05)
print(result_phase)
# print(f'tukey statistic: {result.statistic}, p: {p_phase}')


print('SPEED RESULTS')

F_speed, p_speed = f_oneway(speeds_personalized, speeds_xval, speeds_gen, speeds_ekf)
print(f'F: {F_speed}, p: {p_speed}')

result_speed = pairwise_tukeyhsd(endog=np.array(speeds_personalized+speeds_xval+speeds_gen+speeds_ekf),
                           groups = np.array(['Personalized']*len(speeds_personalized) + ['Xval']*len(speeds_xval) +
                                             ['Gen']*len(speeds_gen) + ['EKF']*len(speeds_ekf)),
                            alpha=0.05)
print(result_speed)


print('INCLINE RESULTS')

F_incline, p_incline = f_oneway(inclines_personalized, inclines_xval, inclines_gen, inclines_ekf)
print(f'F: {F_incline}, p: {p_incline}')

result_incline = pairwise_tukeyhsd(endog=np.array(inclines_personalized+inclines_xval+inclines_gen+inclines_ekf),
                           groups = np.array(['Personalized']*len(inclines_personalized) + ['Xval']*len(inclines_xval) +
                                             ['Gen']*len(inclines_gen) + ['EKF']*len(inclines_ekf)),
                            alpha=0.05)
print(result_incline)

print('STAIRS RESULTS')
F_stairs, p_stairs = f_oneway(stairs_personalized, stairs_xval, stairs_gen)
print(f'F: {F_stairs}, p: {p_stairs}')

result_stairs = pairwise_tukeyhsd(endog=np.array(stairs_personalized+stairs_xval+stairs_gen),
                           groups = np.array(['Personalized']*len(stairs_personalized) + ['Xval']*len(stairs_xval) +
                                             ['Gen']*len(stairs_gen)),
                            alpha=0.05)
print(result_stairs)
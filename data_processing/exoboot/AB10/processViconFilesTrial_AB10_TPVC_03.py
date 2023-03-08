
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob

if __name__ == '__main__':

	filename = 'AB10_EKF_TPVC_03.csv'

	AB_SUBJECT = 'AB10'

	# Load in HS data
	nrows_HS = 304 #row number of the last HS row

	df_HS = pd.read_csv(filename,skiprows=2,nrows=nrows_HS-3)
	# print(df_HS.head())
	# print(df_HS.tail())

	HSs = df_HS.loc[df_HS['Context'] == 'Right']
	HSs = HSs.loc[HSs['Name'] == 'Foot Strike']

	# print(HS.head())
	# print(HS.tail())

	HSs = HSs['Time (s)'].tolist()
	print(HSs)
	# input()


	# MARKER
	n_skiprows_marker_1 = 81094# row num that contains AB0's:
	n_skiprows_marker_2 = n_skiprows_marker_1+1#contains Frame, subframe
	n_skiprows_marker_3 = n_skiprows_marker_1-1#row before marker names

	nrows = 8078 #=  num frames, alsolast idx - row with frame subframe

	# Load in vicon data for markers
	skiprows_vicon = [i for i in range(n_skiprows_marker_1)]
	skiprows_vicon.append(n_skiprows_marker_2)
	# 302516
	df_vicon = pd.read_csv(filename,skiprows=skiprows_vicon,nrows=nrows)
	print(df_vicon.head())
	print(df_vicon.tail())
	print()
	# input()

	markernames = pd.read_csv(filename,skiprows=n_skiprows_marker_3,nrows=1)
	markernames = markernames.loc[:,~markernames.columns.str.contains('^Unnamed')]
	markernames = markernames.columns.tolist()
	print(markernames)
	print(len(markernames))
	col_rename_vicon = {}

	for i in range(len(markernames)*3):

		markername = markernames[i//3]
		# print(markername)
		num = (i+1)//3
		oldstr = ''
		if num == 0:
			pass

		else:
			oldstr = '.'+str(num)

		# print(oldstr)
		# input()
		col_rename_vicon['X' + oldstr] = markername+'.X'
		col_rename_vicon['Y' + oldstr] = markername+'.Y'
		col_rename_vicon['Z' + oldstr] = markername+'.Z'

	df_vicon.rename(columns=col_rename_vicon,inplace=True)

	# print(df_vicon.head())
	# print(df_vicon.tail())

	

	
	frames_vicon = df_vicon['Frame'].to_numpy()
	print('Frames vicon')
	print(len(frames_vicon))
	time_vicon = np.linspace(frames_vicon[0] - frames_vicon[0],frames_vicon[-1] - frames_vicon[0],len(frames_vicon))/100


	RANK_X = df_vicon[f'{AB_SUBJECT}:RANK.X'].to_numpy()
	RANK_Y = df_vicon[f'{AB_SUBJECT}:RANK.Y'].to_numpy()
	RANK_Z = df_vicon[f'{AB_SUBJECT}:RANK.Z'].to_numpy()

	RHEE_X = df_vicon[f'{AB_SUBJECT}:RHEE.X'].to_numpy()
	RHEE_Y = df_vicon[f'{AB_SUBJECT}:RHEE.Y'].to_numpy()
	RHEE_Z = df_vicon[f'{AB_SUBJECT}:RHEE.Z'].to_numpy()

	RTOE_X = df_vicon[f'{AB_SUBJECT}:RTOE.X'].to_numpy()
	RTOE_Y = df_vicon[f'{AB_SUBJECT}:RTOE.Y'].to_numpy()
	RTOE_Z = df_vicon[f'{AB_SUBJECT}:RTOE.Z'].to_numpy()


	# Plot vicon stuff
	fig1, axs1 = plt.subplots(3,1,sharex=True,figsize=(10,6))
	axs1[0].plot(time_vicon, RANK_X, label='RANK_X')
	axs1[0].plot(time_vicon, RANK_Y, label='RANK_Y')
	axs1[0].plot(time_vicon, RANK_Z, label='RANK_Z')

	axs1[1].plot(time_vicon, RHEE_X, label='RHEE_X')
	axs1[1].plot(time_vicon, RHEE_Y, label='RHEE_Y')
	axs1[1].plot(time_vicon, RHEE_Z, label='RHEE_Z')


	axs1[2].plot(time_vicon, RTOE_X, label='RTOE_X')
	axs1[2].plot(time_vicon, RTOE_Y, label='RTOE_Y')
	axs1[2].plot(time_vicon, RTOE_Z, label='RTOE_Z')

	axs1[0].legend()
	axs1[1].legend()
	axs1[2].legend()

	axs1[-1].set_xlabel("time (sec)")


	# input()

	# Calculate state vector quantities
	# timeToIncline = 70 #seconds, timed

	# timeInclineStart = 50
	# timeInclineEnd = 2*60 + 50 #when we start returning to zero incline
	# timeTrialEnd = 300

	# rampTimes = [0,timeInclineStart,timeInclineStart + timeToIncline, timeInclineEnd, timeInclineEnd + timeToIncline, timeTrialEnd]
	# ramps = [0,0,10,10,0,0]


	timeToIncline = 70 #seconds, timed

	timeInclineStart = 10
	timeInclineEnd = 2*60 + 50 #when we start returning to zero incline
	timeTrialEnd = 300

	rampTimes = [0,timeInclineStart,timeInclineStart + timeToIncline]
	ramps = [0,0,-10]


	phase_from_vicon = np.zeros(time_vicon.size)
	phase_rate_from_vicon = np.zeros(time_vicon.size)
	sL_from_vicon = np.zeros(time_vicon.size)
	incline_from_vicon = np.zeros(time_vicon.size)

	speeds_from_vicon = np.zeros(time_vicon.size)

	numSteps = 0
	prevHS_idx = 0
	prevHS_time = 0
	firstIdx = 0
	for i, HS in enumerate(HSs):
		print(HS)

		HS_idx = np.argmin(np.abs(HS - time_vicon))

		HS_time = time_vicon[HS_idx]
		print(HS_time)
		numSteps += 1

		if i == 0:
			firstIdx = HS_idx + 1
		else:
			strideTime = HS_time - prevHS_time
			phaseRate = 1/strideTime

			speed = 1
			speeds_from_vicon[prevHS_idx:HS_idx+1] = speed
			strideLength = (RHEE_Y[HS_idx] - RHEE_Y[prevHS_idx])/1000 + (speed*strideTime)

			phase_from_vicon[prevHS_idx:HS_idx+1] = (time_vicon[prevHS_idx:HS_idx+1] - time_vicon[prevHS_idx])/(strideTime)
			phase_rate_from_vicon[prevHS_idx:HS_idx+1] = phaseRate
			sL_from_vicon[prevHS_idx:HS_idx+1] = strideLength
			incline_from_vicon[prevHS_idx:HS_idx+1] = np.interp(time_vicon[prevHS_idx:HS_idx+1], rampTimes, ramps)


		# input()
		prevHS_idx = HS_idx + 1
		prevHS_time = time_vicon[prevHS_idx]

	time_vicon = time_vicon[firstIdx:prevHS_idx]
	phase_from_vicon = phase_from_vicon[firstIdx:prevHS_idx]
	phase_rate_from_vicon = phase_rate_from_vicon[firstIdx:prevHS_idx]
	sL_from_vicon = sL_from_vicon[firstIdx:prevHS_idx]
	incline_from_vicon = incline_from_vicon[firstIdx:prevHS_idx]
	speeds_from_vicon = speeds_from_vicon[firstIdx:prevHS_idx]


	#Plot Calculated Quantities
	fig3, axs3 = plt.subplots(4,1,sharex=True,figsize=(10,6))


	axs3[0].plot(time_vicon, phase_from_vicon,'r', label='Phase')
	
	axs3[1].plot(time_vicon, phase_rate_from_vicon,'r', label='Phase Rate')
	axs3[2].plot(time_vicon, sL_from_vicon,'r', label='Stride Length')
	axs3[3].plot(time_vicon, incline_from_vicon,'r', label='Incline')

	axs3[0].legend()
	axs3[1].legend()
	axs3[2].legend()
	axs3[3].legend()

	

	axs3[-1].set_xlabel("time (sec)")

	fig, axs = plt.subplots(sharex=True,figsize=(10,6))
	axs.plot(time_vicon, speeds_from_vicon,'r', label='Speed')
	axs.set_xlabel("time (sec)")
	axs.legend()

	for HS in HSs:
		print(HS)
		axs1[1].vlines(np.array(HS),0,1000,'k')

		axs3[0].vlines(np.array(HS),0,1,'k')
		axs3[2].vlines(np.array(HS),0,1,'k')

		axs.vlines(np.array(HS),0,1,'k')



	HSs = np.pad(np.array(HSs), (0, len(RHEE_Y[firstIdx:prevHS_idx]) - len(HSs)),'constant', constant_values=np.nan )
	print(len(HSs))
	# input()
	df = pd.DataFrame({'time': time_vicon,
                   'phase': phase_from_vicon,
                   'phase_rate': phase_rate_from_vicon,
                   'stride_length':sL_from_vicon,
                   'incline':incline_from_vicon,
                   'RTOE_X': RTOE_X[firstIdx:prevHS_idx],
                   'HS_vicon': HSs
                   })
	df.to_csv(f'Vicon_{AB_SUBJECT}TPVC_03_processed.csv',index=False)






	plt.show()















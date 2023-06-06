"""Estimates phase using the timing approach
"""
import traceback

import numpy as np
from time import time


class TimingPhaseEstimator():

	"""A class that estimates phase purely using recorded step durations
	
	Attributes:
	    numStepsTaken (int): The number of recorded steps taken, gets updated by the HP class that encompasses this class
	    numStridesToAvg (int): the number of past stride times that will be averaged
	    phase_dot_estimate_TBE (float): The estimate of the phase rate
	    phase_estimate_TBE (float): The estimate of the phase
	    stepDurations (list): a list of past step durations, gets updated by the HP class that encompasses this class
	    stepStartTime (float): the start time of the current step, gets updated by the HP class that encompasses this class
	    timeStrideMean (float): The average stride time of the last few strides

	"""
	
	def __init__(self,):
		"""Initialize
		"""

		self.phase_estimate_TBE = 0
		self.phase_dot_estimate_TBE = 1

		self.stepStartTime = 0

		self.numStepsTaken = 0
		self.stepDuration = 1
		self.stepDurations = []

		self.numStridesToAvg = 1

		self.timeStrideMean = 1

	def computeAvgStrideTime(self,):
		"""Computes the average stride time of the last few strides
		"""
		if self.numStepsTaken < self.numStridesToAvg:
			self.timeStrideMean = np.mean(self.stepDurations)

		else:
			self.timeStrideMean = np.mean(self.stepDurations[-self.numStridesToAvg:])
		# print(self.stepDurations)
		
	def computeTBEPhase(self,t):
		"""Computes the TBE estimate of phase
		
		Args:
		    t (float): the current time, in seconds
		"""
		self.phase_estimate_TBE = (t - self.stepStartTime)/self.timeStrideMean
		self.phase_estimate_TBE = self.phase_estimate_TBE % 1.0
		self.phase_dot_estimate_TBE = 1/self.timeStrideMean


		








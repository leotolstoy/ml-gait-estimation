import sys

sys.path.append('/utils/')

from evalFourierFuncs_3P import returnFourierBasis_Eval, returnFourier
from evalBezierFuncs_3P import selectOrderBezier

import numpy as np


class TorqueProfile():
	"""This class contains the implementation of a torque profile to control 
	ankle exoskeletons. The profile is split into two parts: a torque profile that is applied during continuous 
	overground locomotion, and a torque profile that is applied during stairs locomotion
	"""	
	def __init__(self, model_dict, model_dict_stairs):
		"""

		Args:
			model_dict (dict): dictionary containing the orders of the phase, speed, and incline basis functions for the overground profile
			model_dict_stairs (dict): dictionary containing the orders of the phase, speed, and incline basis functions for the stairs profile
		"""		
		

		self.params_torque = self._loadCoefficients(model_dict['model_filepath']) #load Bezier curves in
		self.phase_order = model_dict['phase_order']
		self.speed_order = model_dict['speed_order']
		self.incline_order = model_dict['incline_order']
		self.numFuncs = (self.incline_order+1) * (self.speed_order+1) * (self.phase_order+1)

		#load in stairs profile as well

		self.params_stairs_torque = self._loadCoefficients(model_dict_stairs['model_filepath']) #load Bezier curves in
		self.phase_order_stairs = model_dict_stairs['phase_order']
		self.speed_order_stairs = model_dict_stairs['speed_order']
		self.stair_height_order_stairs = model_dict_stairs['stair_height_order']
		self.numFuncs_stairs = (self.stair_height_order_stairs+1) * (self.speed_order_stairs+1) * (self.phase_order_stairs+1)
		self.STAIRS_THRESHOLD = 0.5
		self.SPEED_THRESHOLD = 0.1


	def _loadCoefficients(self,filename):
		"""Load the basis function coefficients in from a file

		Args:
			filename (string): the coefficient filename

		"""		
		data = np.loadtxt(filename,delimiter=',')
		params_ankle_torque = data[:]
		return params_ankle_torque


	def evalTorqueProfile(self,phase, speed, incline, stair_height):
		"""Evaluate the torque profile at a given phase, speed, incline, and stair height

		Args:
			phase_estimate (float): phase at which to evaluate
			speed_estimate (float): stride length at which to evaluate
			incline_estimate (float): incline at which to evaluate
			stair_height (float): stair height at which to evaluate

		Returns:
			float: the torque at the gait state point
		"""		
		torque = 0
		if phase <= 0.65:
			stair_height_eff = 0

			#only apply the stair profile if we're above a threshold
			if np.abs(stair_height) > self.STAIRS_THRESHOLD:
                #convert to an actual stair height, since the value passed in will be between
				# (-1,1) with -1 denoting stair descent and 1 denoting stair ascent
				if stair_height > 0:
					stair_height_eff = 0.152
				elif stair_height < 0:
					stair_height_eff = -0.152

				stairHeightFuncs = selectOrderBezier(self.stair_height_order_stairs, stair_height_eff)
				speedFuncs = selectOrderBezier(self.speed_order_stairs, speed)
				phaseFuncs = returnFourier(phase, self.phase_order_stairs)
				basis = np.kron(stairHeightFuncs, np.kron(speedFuncs, phaseFuncs))
				torque = self.params_stairs_torque @ basis

				#scale down the biomimetic torque
				torque = torque / 7
				
			else:
				#compute the torque during plantarflexion
				torque = self.params_torque @ returnFourierBasis_Eval(phase,speed,incline, self.phase_order, self.speed_order, self.incline_order)
				torque = torque / 7

		#only apply torque if we're moving
		if speed < self.SPEED_THRESHOLD:
			torque = 0
			
		#ensure torque is unidirectional
		if torque < 0:
			torque = 0

		return torque


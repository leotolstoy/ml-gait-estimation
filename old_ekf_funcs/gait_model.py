import numpy as np
from time import time
from evalBezierFuncs_3P import *
from evalFourierFuncs_3P import *
from arctanMapFuncs import *





class GaitModel_Bezier():

	def __init__(self, model_filepath='GaitModel/regressionMatrices_dataport3P.csv',phase_order=3, stride_length_order=1, incline_order=1):
		self.phaseDelins = [0.1,0.5,0.65,1]
		(self.best_fit_params_footAngle,self.best_fit_params_shankAngle) = self.loadCoefficients(model_filepath)
		self.phase_order = phase_order
		self.stride_length_order = stride_length_order
		self.incline_order = incline_order

		self.numFuncs = (incline_order+1) * (stride_length_order+1) * (phase_order+1)


	def loadCoefficients(self,filename):
	    data = np.loadtxt(filename,delimiter=',')

	    best_fit_params_footAngle = data[0,:]
	    best_fit_params_shankAngle = data[1,:]

	    return (best_fit_params_footAngle,best_fit_params_shankAngle)

	#KINEMATICS
	def returnFootAngle(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P(phase,stepLength,incline, self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngle(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P(phase,stepLength,incline, self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);


	#FIRST DERIVATIVES
	def returnFootAngleDeriv_dphase(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dphase(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngleDeriv_dphase(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dphase(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnFootAngleDeriv_dsL(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dsL(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngleDeriv_dsL(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dsL(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnFootAngleDeriv_dincline(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dincline(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngleDeriv_dincline(self, phase,stepLength,incline):
		return returnPiecewiseBezier3PDeriv_dincline(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);
		
	#SECOND DERIVATIVES

	def returnFootAngle2ndDeriv_dphase2(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngle2ndDeriv_dphase2(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnFootAngle2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngle2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnFootAngle2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,stepLength,incline,self.best_fit_params_footAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);

	def returnShankAngle2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,stepLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, 
			self.numFuncs, self.phase_order, self.stride_length_order, self.incline_order);




class GaitModel_Fourier():

	def __init__(self, model_filepath ,phase_order=20, stride_length_order=1, incline_order=1):
		(self.best_fit_params_footAngle, 
			self.best_fit_params_shankAngle,
			self.best_fit_params_heelPosForward,
			self.best_fit_params_heelPosUp) = self.loadCoefficients(model_filepath)
		self.phase_order = phase_order
		self.stride_length_order = stride_length_order
		self.incline_order = incline_order


	def loadCoefficients(self,filename):
	    data = np.loadtxt(filename,delimiter=',')

	    best_fit_params_footAngle = data[0,:]
	    best_fit_params_shankAngle = data[1,:]
	    best_fit_params_heelPosForward = data[2,:]
	    best_fit_params_heelPosUp = data[3,:]

	    return (best_fit_params_footAngle,best_fit_params_shankAngle,best_fit_params_heelPosForward, best_fit_params_heelPosUp)

	#KINEMATICS
	def returnFootAngle(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_Eval(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngle(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_Eval(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForward(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_Eval(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUp(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_Eval(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)



	#FIRST DERIVATIVES
	def returnFootAngleDeriv_dphase(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_DerivEval_dphase(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngleDeriv_dphase(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_DerivEval_dphase(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForwardDeriv_dphase(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_DerivEval_dphase(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUpDeriv_dphase(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_DerivEval_dphase(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)


	def returnFootAngleDeriv_dsL(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_DerivEval_dsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngleDeriv_dsL(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_DerivEval_dsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForwardDeriv_dsL(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_DerivEval_dsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosUpDeriv_dsL(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_DerivEval_dsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)


	def returnFootAngleDeriv_dincline(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_DerivEval_dincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngleDeriv_dincline(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_DerivEval_dincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosForwardDeriv_dincline(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_DerivEval_dincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUpDeriv_dincline(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_DerivEval_dincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)


	#SECOND DERIVATIVES

	def returnFootAngle2ndDeriv_dphase2(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_2ndDerivEval_dphase2(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngle2ndDeriv_dphase2(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_2ndDerivEval_dphase2(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForward2ndDeriv_dphase2(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_2ndDerivEval_dphase2(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUp2ndDeriv_dphase2(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_2ndDerivEval_dphase2(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)


	def returnFootAngle2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_2ndDerivEval_dphasedsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngle2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_2ndDerivEval_dphasedsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForward2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_2ndDerivEval_dphasedsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUp2ndDeriv_dphasedsL(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_2ndDerivEval_dphasedsL(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)


	def returnFootAngle2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return self.best_fit_params_footAngle @ returnFourierBasis_2ndDerivEval_dphasedincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnShankAngle2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return self.best_fit_params_shankAngle @ returnFourierBasis_2ndDerivEval_dphasedincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)

	def returnHeelPosForward2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosForward @ returnFourierBasis_2ndDerivEval_dphasedincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)
	
	def returnHeelPosUp2ndDeriv_dphasedincline(self, phase,stepLength,incline):
		return self.best_fit_params_heelPosUp @ returnFourierBasis_2ndDerivEval_dphasedincline(phase,stepLength,incline, self.phase_order, self.stride_length_order, self.incline_order)







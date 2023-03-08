"""This file contains various utilities that depend on pytorch used to train 
    and evaluate the neural networks in this repo
"""
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy


class GaitLoss(nn.Module):
    """This class defines a custom loss function used during network training 
    that applies different weights to the errors of the different gait state estimates

    Additionally, it contains functionality to weight the errors higher when the gait state
    is on stairs or stopped, to account for the fact that the training sets have less stairs
    data and data when the person is stationary

    """    
    def __init__(self, w_phase=1, w_speed=1, w_incline=10, w_stairs=2):
        super(GaitLoss, self).__init__();
        self.w_phase = w_phase
        self.w_speed = w_speed
        self.w_incline = w_incline
        self.w_stairs = w_stairs

    def forward(self, predictions, target):
        """
        Args:
            predictions (Tensor): (batched) Tensor containing the predicted gait states from the neural network
            target (Tensor): (batched) Tensor containing the actual gait states from the neural network

        Returns:
            float: the weighted loss of the gait state predictor
        """        
        #calculate continuous loss
        #assumes first four elements of the tensor are continuous gait states
        cp_loss_value = nn.functional.mse_loss(predictions[:,:,0], target[:,:,0])
        sp_loss_value = nn.functional.mse_loss(predictions[:,:,1], target[:,:,1])
        incline_loss_value = nn.functional.mse_loss(predictions[:,:,3], target[:,:,3])
        
        #weight zero speeds heavily by 10x the normal speed weight
        speed_loss_value = nn.functional.mse_loss(predictions[:,:,2], target[:,:,2],reduction='none')
        effective_ws_speed = (self.w_speed) * torch.ones(target[:,:,2].shape,device=target.device)
        not_is_moving_idx = target[:,:,2] < 0.01  #threshold low speeds as zero
        effective_ws_speed[not_is_moving_idx] = 10*(self.w_speed)
        speed_loss_weighted = torch.mean(effective_ws_speed*speed_loss_value)


        #Calculate stair loss 
        is_stairs_loss_value = nn.functional.mse_loss(predictions[:,:,4], target[:,:,4],reduction='none')

        # apply the full weight when we receive an on-stairs data point
        # non_stair points get a weight of w_stairs, stair points get 5x w_stairs weighting
        effective_ws_stairs = (self.w_stairs) * torch.ones(target[:,:,4].shape,device=target.device)
        
        #unscaled range of (-1,1) gets turned to (0,1), so norm 0.5 corresponds to 0 stairs label, i.e. not on stairs
        is_stair_ascent_idx = target[:,:,4] > 0.55 #stair ascent
        is_stair_descent_idx = target[:,:,4] < 0.45 #stair descent
        
        effective_ws_stairs[is_stair_ascent_idx] = 5*(self.w_stairs)
        effective_ws_stairs[is_stair_descent_idx] = 5*(self.w_stairs)
        
        stairs_loss_weighted = torch.mean(effective_ws_stairs*is_stairs_loss_value)


        return self.w_phase * (cp_loss_value + sp_loss_value) + \
                            speed_loss_weighted + \
                            self.w_incline * incline_loss_value + \
                            stairs_loss_weighted


class WeightedKinematicsLoss(nn.Module):
    """This class defines a custom loss function used during the kinematics gait model network training 
    calculates the MSE loss for the gait model, with stairs data being heavily weighted

    """    
    def __init__(self, w_stairs=2):
        super(WeightedKinematicsLoss, self).__init__();
        self.w_stairs = w_stairs

    def forward(self, predictions, target, gait_state):
        #calculate is_stairs
        is_stairs = torch.abs(gait_state[:,:,4]) > 0.5

        #calculate continuous loss
        loss = nn.functional.mse_loss(predictions, target, reduction='none')

        loss[is_stairs,:] = self.w_stairs*loss[is_stairs,:]
        loss = torch.mean(loss)
        return loss


class EWC(object):
    """This class implements the Elastic Weight Constraint functionality from
    "Overcoming catastrophic forgetting in neural networks" https://arxiv.org/abs/1612.00796
    during finetuning on personalized exoboot data, such that the network does not overfit to the finetuning
    dataset

    """    
    def __init__(self, model: nn.Module, fisher_importance: float, dataloader: DataLoader, 
                  device: torch.device, SOS_token=None, fisher_mats_path=None, COMPUTE_FISHER_MATS=False, DT_IDX=6,
                    lossfcn = None):
        """

        Args:
            model (nn.Module): the pretrained gait transformer model
            fisher_importance (float): the importance of the performance on the old task
            dataloader (DataLoader): a dataloader containing the data from the old task, used if computing new Fisher info matrices
            device (torch.device): the device on which the models live
            SOS_token (Tensor, optional): the start of sequence token for the gait transformer. Defaults to None.
            fisher_mats_path (string, optional): filepath of the saved Fisher matrices, if not recomputing. Defaults to None.
            COMPUTE_FISHER_MATS (bool, optional): whether to recompute the Fisher info matrices. Defaults to False.
            DT_IDX (int, optional): index of the time steps. Defaults to 6.
            lossfcn (fcn, optional): loss function used to compute Fisher info matrices. Defaults to None.
        """                    

        
        #deep copy the original (non finetuned) model
        self.model = deepcopy(model)
        self.model.to(device)
        self.device = device
        self.DT_IDX = DT_IDX
        self.SOS_token = SOS_token
        self.lossfcn = lossfcn

        #copy trainable parameters
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._baseline_model_params = {}

        #initialize with the precomputed matrices if they exist
        if not COMPUTE_FISHER_MATS:
            precomp_fisher_mats = torch.load(fisher_mats_path)
            # print(precomp_fisher_mats)
            print('Loading precomp mats')
            self._fisher_matrices = precomp_fisher_mats

        else:
            print('Generating Fisher mats')
            print(f'Datapoints used to calculate Fisher mats: {len(dataloader.dataset)}')
            self._fisher_matrices = self._compute_diag_fisher(dataloader)
            torch.save(self._fisher_matrices, fisher_mats_path)
            
        self.importance = fisher_importance
        print(f'Importance: {self.importance}')

        #deepcopy the trainable params from the pretrained model
        for n, p in deepcopy(self.params).items():
            self._baseline_model_params[n] = p.data

    def _compute_diag_fisher(self, dataloader):
        """This internal function computes the Fisher information matrices that describe
        which neurons in the original network are most important for the original task of 
        vicon gait state estimation

        Args:
            dataloader: the dataloader that holds the original training data
        """        
        fisher_matrices = {}

        #initialize fisher matrices
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher_matrices[n] = p.data

        self.model.eval()

        #loop through the batches in the dataloader
        #evaluate the fisher matrix for the batch
        N_total_samples = len(dataloader.dataset)
        for batch in dataloader:

            b_input = batch['meas'].to(self.device)
            b_state = batch['state'].to(self.device)
            
                
            tgt = self.SOS_token.repeat(b_state.shape[0], 1, 1)
            dts = b_input[:,:,self.DT_IDX]
            dts = torch.unsqueeze(dts, dim=-1)
              
            self.model.zero_grad()

            #run the model for each input in the batch
            prediction = self.model(b_input,tgt, dts)

            #compute NLL for (assumed) Gaussian for the continuous variables
            #for MSE loss, NLL is the regular loss
            negative_log_likelihood = self.lossfcn(prediction, b_state)
            # print('negative_log_likelihood')
            # print(negative_log_likelihood)
            
            #for the neural net, the loss function is the NLL, so calculate the score 
            # by taking the derivative of the negative of the  NLL wrt the params (done via the backward pass)
            log_likelihood = -negative_log_likelihood
            # print('log_likelihood')
            # print(log_likelihood)
            # input()
            log_likelihood.backward()

            #compute the Fisher information matrix, which is just the
                #square of gradient of the negative_log_likelihood loss 
                # wrt the parameters
                #the weighted average across the batch is then taken
            for n, p in self.model.named_parameters():
                fisher_matrices[n].data += (p.grad.data ** 2) * ( b_input.shape[0] / N_total_samples)

        fisher_matrices = {n: p for n, p in fisher_matrices.items()}
        return fisher_matrices

    def penalty(self, new_model: nn.Module):
        """Compute the EWC penalty for the new model params relative to the old model task

        Args:
            new_model (nn.Module): the model being finetuned

        """        
        loss = 0

        for n, p in new_model.named_parameters():
            d_params = (p - self._baseline_model_params[n]) ** 2 #compute difference in parameters between the models
            _loss = self._fisher_matrices[n] * d_params
            loss += _loss.sum()
        return self.importance * loss

def enum_parameters(model):
    """Return the number of trainable parameters in a model
    """    
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    return total_params
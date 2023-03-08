import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import random

class WindowedGaitDataset(Dataset):
    """Windowed Gait dataset."""

    def __init__(self, gait_data, window_size=50, meas_scale=None, speed_scale=None, stair_height_scale=None, incline_scale=None, transform=None):
        """
        Args:
            gait_data (pandas data frame): a pandas data frame of the walking gait data
            window_size (integer): size of the window to apply to the data
            meas_scale (np.array, optional): an Nx2 array of the kinematics scaling factors, where N is the number of measurements
            speed_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the speed
            incline_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the incline
            stair_height_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the stair heights
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gait_data = gait_data
        self.window_size = window_size
        self.meas_scale = meas_scale
        self.speed_scale = speed_scale
        self.incline_scale = incline_scale
        self.stair_height_scale = stair_height_scale
        self.transform = transform

    def __len__(self):
        return len(self.gait_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #handle indexing dataframe rows within the window size
        #if the index is smaller than the window size, select another index outside that range randomly
        if idx < self.window_size and idx >= 0:
          idx = random.randint(self.window_size, len(self.gait_data)-1)


        #specify the indexes that contain the measured kinematics
        meas_idxs = [0,1,2,3,4,5,14]

        #specify the indexes that contain the gait states
        #phase, speed, incline, is_stairs
        gait_state_idxs = [6,7,8,9]    

        measurements = self.gait_data.iloc[idx-self.window_size+1:idx+1,meas_idxs].to_numpy()
        gait_states = self.gait_data.iloc[idx,gait_state_idxs].to_numpy()
        
        #round the stair height to one of three labels
        # - -1 if descending stairs
        # = 0 if not on stairs
        # - 1 if ascending stairs
        if gait_states[3] < -0.05:
            gait_states[3] = -1
        elif gait_states[3] > 0.05:
            gait_states[3] = 1
        else:
            gait_states[3] = 0 

        #apply scaling
        if self.meas_scale is not None:
            for i in range(len(meas_idxs)):
                lb = self.meas_scale[i,0]
                ub = self.meas_scale[i,1]
                measurements[:,i] = ((1 - 0)/(ub - lb)) * (measurements[:,i] - lb)

        if self.speed_scale:
            lb = self.speed_scale[0]
            ub = self.speed_scale[1]
            gait_states[1] = ((1 - 0)/(ub - lb)) * (gait_states[1] - lb)

        if self.incline_scale:
            lb = self.incline_scale[0]
            ub = self.incline_scale[1]
            gait_states[2] = ((1 - 0)/(ub - lb)) * (gait_states[2] - lb)
            
        if self.stair_height_scale:
            lb = self.stair_height_scale[0]
            ub = self.stair_height_scale[1]
            gait_states[3] = ((1 - 0)/(ub - lb)) * (gait_states[3] - lb)

        #apply trig to the phase variable to handle the discontinuity at 1
        #this encodes the phase as a sine and cosine 
        phase_as_angle = 2*np.pi*(gait_states[0]-0.5)
        cp = np.cos(phase_as_angle)
        sp = np.sin(phase_as_angle)
        gait_states_new = np.zeros(gait_states.shape[0]+1,)
        gait_states_new[0] = cp
        gait_states_new[1] = sp
        gait_states_new[2:] = gait_states[1:]
        gait_states = gait_states_new

        #compose the sample on the dataset
        sample = {'meas': measurements, 'state': gait_states}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ExobootDataset(Dataset):
    """ExobootDataset dataset."""

    def __init__(self, gait_data, window_size=50, meas_scale=None, speed_scale=None, incline_scale=None, stair_height_scale=None,transform=None):
        """
        Args:
            gait_data (pandas data frame): a pandas data frame of the walking gait data
            window_size (integer): size of the window to apply to the data
            meas_scale (np.array, optional): an Nx2 array of the kinematics scaling factors, where N is the number of measurements
            speed_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the speed
            incline_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the incline
            stair_height_scale (tuple, optional): A 2-tuple of the lower and upper bounds to scale the stair heights
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gait_data = gait_data
        self.window_size = window_size
        self.meas_scale = meas_scale
        self.speed_scale = speed_scale
        self.incline_scale = incline_scale
        self.stair_height_scale = stair_height_scale
        self.transform = transform

    def __len__(self):
        return len(self.gait_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #handle indexing dataframe rows within the window size
        #if the index is smaller than the window size, select another index outside that range randomly
        if idx < self.window_size and idx >= 0:
          # idx = len(self.gait_data)-1
          idx = random.randint(self.window_size, len(self.gait_data)-1)
          # print('saturating')

        #specify the indexes that contain the measured kinematics
        meas_idxs = [0,1,2,3,4,5,6]

        #specify the indexes that contain the gait states
        #phase, speed, incline, is_stairs
        gait_state_idxs = [7,8,9,10]  
        measurements = self.gait_data.iloc[idx-self.window_size+1:idx+1,meas_idxs].to_numpy()
        gait_states = self.gait_data.iloc[idx,gait_state_idxs].to_numpy()

        
        #round the stair height to one of three labels
        # - -1 if descending stairs
        # = 0 if not on stairs
        # - 1 if ascending stairs
        if gait_states[3] < -0.05:
            gait_states[3] = -1
        elif gait_states[3] > 0.05:
            gait_states[3] = 1
        else:
            gait_states[3] = 0 
        
        #apply scaling 
        if self.meas_scale is not None:
            for i in range(len(meas_idxs)):
                lb = self.meas_scale[i,0]
                ub = self.meas_scale[i,1]
                measurements[:,i] = ((1 - 0)/(ub - lb)) * (measurements[:,i] - lb)

        if self.speed_scale:
            lb = self.speed_scale[0]
            ub = self.speed_scale[1]
            gait_states[1] = ((1 - 0)/(ub - lb)) * (gait_states[1] - lb)

        if self.incline_scale:
            lb = self.incline_scale[0]
            ub = self.incline_scale[1]
            gait_states[2] = ((1 - 0)/(ub - lb)) * (gait_states[2] - lb)
        
        if self.stair_height_scale:
            lb = self.stair_height_scale[0]
            ub = self.stair_height_scale[1]
            gait_states[3] = ((1 - 0)/(ub - lb)) * (gait_states[3] - lb)
        
        #apply trig to the phase variable to handle the discontinuity at 1
        #this encodes the phase as a sine and cosine 
        phase_as_angle = 2*np.pi*(gait_states[0]-0.5)
        cp = np.cos(phase_as_angle)
        sp = np.sin(phase_as_angle)
        gait_states_new = np.zeros(gait_states.shape[0]+1,)
        gait_states_new[0] = cp
        gait_states_new[1] = sp
        gait_states_new[2:] = gait_states[1:]
        gait_states = gait_states_new

        #compose the sample on the dataset
        sample = {'meas': measurements, 'state': gait_states}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors.

    Returns:
        a dict that contains Tensors of both the measured kinematics and the state
    """    

    def __call__(self, sample):
        meas, state = sample['meas'], sample['state']
       
        meas = torch.from_numpy(meas).float()
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0)

        return {'meas': meas, 
                'state': state}

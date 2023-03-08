import torch
from torch import nn, Tensor
import numpy as np
from training_utils import generate_unit_vector

class GaitModel(nn.Module):
    """This class represents a continuous gait model using a multilayer perceptron
        that maps gait state inputs to kinematics
    As implemented, this class expects a normalized gait state input of: 
        [cos_phase, 
        sin_phase, 
        speed, 
        incline, 
        is_stairs]
    This class outputs a kinematics vector of: 
        [foot angle, 
        foot angle velocity, 
        shank angle, 
        shank angle velocity, 
        heel forward acceleration, 
        heel upward acceleration] 
    """
    def __init__(self, 
        input_size: int,
        num_predicted_features: int=4,
        batch_first: bool=True,
        dim_val: int=512,  
        n_hidden_layers: int=2,
        ): 
        """

        Args:
            input_size (int): the number of gait state inputs
            num_predicted_features (int, optional): the number of kinematics outputs. Defaults to 4.
            batch_first (bool, optional): whether to process/train the network using batch first inputs. Defaults to True.
            dim_val (int, optional): the dimension of the latent space. Defaults to 512.
            n_hidden_layers (int, optional): The number of hidden layers in the network. Defaults to 2.
        """
        super().__init__() 

        self.num_predicted_features = num_predicted_features
        self.input_size = input_size
        self.hidden_layers = []

        for i in range(n_hidden_layers):
            self.hidden_layers.extend([nn.Linear(in_features=dim_val, out_features=dim_val), nn.GELU()])
            
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        
        self.embedding_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.output_layer = nn.Linear(in_features=dim_val, out_features=num_predicted_features)        
          

    def forward(self, x):        
        x = self.embedding_layer(x) #embed the input into the latent space
        
        #run the input through the hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
        
        x = self.output_layer(x) #transform the input back into the output space
        return x
    
    def compute_numerical_gradient(self, lin_point, predicted_kinematics_tensor, device):
        """This function computes the numerical gradient (Jacobian if you're fancy) of the output
        kinematics with respect to the input gait state. This can be used in cases/contexts where 
        torch isn't computing gradients usign autograd since it's pretty slow

        Args:
            lin_point (Tensor): The linearization point about which to compute the gradient
            predicted_kinematics_tensor (Tensor): The predicted kinematics at the linearization point
            device: the hardware device on which the network is located

        Returns:
            Tensor: the Jacobian/gradient matrix that relates changes in input to changes in output
        """        
        dx = 1e-6 #set the delta/perturbation
        gradient_num = np.zeros((self.num_predicted_features, self.input_size)) #preallocate the Jacobian

        for jj in range(self.input_size):
            e_vec = generate_unit_vector(self.input_size,jj) #compute unit vector along the dimension of state jj
            e_vec_torch = torch.from_numpy(e_vec).float().to(device).unsqueeze(dim=0).unsqueeze(dim=0)

            predicted_tensor_shifted = lin_point + dx*e_vec_torch #shift the linearization point along dimension jj
            predicted_kinematics_tensor_shifted = self.forward(predicted_tensor_shifted) #predict the kinematics at the perturbed point

            #compute gradient for element
            dkinematicsdx = (predicted_kinematics_tensor_shifted - predicted_kinematics_tensor).squeeze().cpu().numpy()/dx
            gradient_num[:,jj] = dkinematicsdx
            
        return gradient_num

    
class HeteroscedasticModel(nn.Module):
    """This class represents a heteroscedastic model that encodes a changing level of trust/variance
    in the kinematic measurements. This model returns a Tensor of the unique elements of the covariance matrix
    """
    def __init__(self, 
        input_size=6,
        num_predicted_features=6,
        batch_first=True,
        dim_val=512,  
        n_hidden_layers=2,
        log_covar_element_scale = np.array(
            [[-4.60517019,  6.59583506],
             [-4.60517019,  8.00316999],
             [-4.60517019,  6.23502258],
             [-4.60517019,  7.62349018],
             [-4.60517019,  5.32510388],
             [-4.60517019,  5.17364564],
             [-2.21500356, 11.25173026],
             [-4.60517019,  7.68459809],
             [-4.60517019, 10.49797692],
             [-4.60517019,  7.5181146 ],
             [-4.60517019,  7.29806282],
             [-4.60517019,  6.26124854],
             [-4.60517019,  7.33407139],
             [-4.60517019,  4.83793644],
             [-4.60517019,  4.87319887],
             [-1.83459664, 10.27920013],
             [-4.60517019,  7.15667353],
             [-4.60517019,  6.91650956],
             [-4.60517019,  5.75253599],
             [-4.60517019,  4.40803868],
             [-4.60517019,  5.3568463 ]])
        ): 
        """
        Args:
            input_size (int, optional): the number of gait state inputs. Defaults to 6.
            num_predicted_features (int, optional): the number of covariance outputs. Defaults to 6.
            batch_first (bool, optional): whether to process/train the network using batch first inputs. Defaults to True.
            dim_val (int, optional): the dimension of the latent space. Defaults to 512.
            n_hidden_layers (int, optional): The number of hidden layers in the network. Defaults to 2.
            log_covar_element_scale (np array, optional): an Nx2 array of scaling factors that scale the _log_ of the covariances.
        """        

        super().__init__() 

        self.num_predicted_features = num_predicted_features
        self.input_size = input_size
        self.hidden_layers = []

        for i in range(n_hidden_layers):
            self.hidden_layers.extend([nn.Linear(in_features=dim_val, out_features=dim_val), nn.GELU()])
            
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        
        self.embedding_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.output_layer = nn.Linear(in_features=dim_val, out_features=num_predicted_features)        
          
        self.log_covar_element_scale = log_covar_element_scale

    def forward(self, x):        
        x = self.embedding_layer(x)
        
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
        
        x = self.output_layer(x)
        return x
    
    def scale_unique_log_covar_elements(self, unique_covar_elements):
        """This helper function scales the log of the unique covariance elements to [0,1]
        """        
        rows, cols = unique_covar_elements.shape
        scaled_covar_mat = np.zeros((rows,cols))

        for i in range(cols):
            scale_lb = self.log_covar_element_scale[i,0]
            scale_ub = self.log_covar_element_scale[i,1]
            # print(scale_lb, scale_ub)
            scaled_covar_mat[:,i] = (unique_covar_elements[:,i] - scale_lb) * (1/(scale_ub - scale_lb))
            # input()
        return scaled_covar_mat
    
    def unscale_unique_log_covar_elements(self, unique_covar_elements):
        """This helper function unscales the log of the unique covariance elements from [0,1]
        """        
        rows, cols = unique_covar_elements.shape
        unscaled_covar_mat = np.zeros((rows,cols))

        for i in range(cols):
            scale_lb = self.log_covar_element_scale[i,0]
            scale_ub = self.log_covar_element_scale[i,1]
            unscaled_covar_mat[:,i] = (unique_covar_elements[:,i] * (scale_ub - scale_lb)) + scale_lb
        return unscaled_covar_mat
            
    def process_unique_covar_elements(self, unique_covar_elements):
        """This helper function both unscales the log of the unique covariance elements from [0,1]
        and undoes the log operation on the elements
        """        
        #need unique_covar_elements to be N x 21
        processed_unique_covar_elements = self.unscale_unique_log_covar_elements(unique_covar_elements)
        processed_unique_covar_elements = np.exp(processed_unique_covar_elements)
        return processed_unique_covar_elements

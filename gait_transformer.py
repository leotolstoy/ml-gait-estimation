# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
# tf.test.is_built_with_cuda()
import torch
import torch.nn as nn 
from torch import nn, Tensor
import numpy as np


import random

class PositionalEncoder(nn.Module):
    """
    This class encodes the temporal position of the kinematic measurements input to the GaitTransformer. This
    is needed to distinguish similar measurements across time, as Transformers do not have recurrence or state to 
    keep track of time. The temporal positional encoding uses a vector of delta time steps to encode how far back
    these measurements are in time.
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=100, 
        d_model: int=512,
        batch_first: bool=True,
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first
        self.max_seq_len = max_seq_len
        self.x_dim = 1 if batch_first else 0
        pe = torch.zeros((1,max_seq_len,d_model)) #the position encoding Tensor

        #don't keep track of the position Tensor when saving the model
        self.register_buffer('pe', pe, persistent=False)

        
    def forward(self, x: Tensor, dts: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): the input to positionally encode
            dts (Tensor): a vector of delta time steps that are used to encode how far back elements
            of x are in time

        Returns:
            Tensor: The positionally encoded input
        """        

        #cumulative integrate the time steps, starting backwards at the last row since the last row
        #contains the most recent measurement
        t_rel = torch.cumsum(torch.flip(dts, [1]), dim=1)

        #flip order so most recent integrated is last
        t_rel = torch.flip(t_rel, [1])
        
        #repeat integrated time across the embedding dimension
        t_rel = t_rel.repeat(1, 1, self.d_model)

        #define the most recent temporal encoding as zero
        self.pe[0,:-1,:] = t_rel[0,1:,:]
        
        x = x + self.pe

        return self.dropout(x)

class GaitTransformer(nn.Module):
    """This class encodes the primary Transformer that translates a measurement buffer of kinematics
    to a succinct gait state at the present time
    """    
    def __init__(self, 
        input_size: int,
        num_predicted_features: int=4,
        batch_first: bool=True,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        enc_seq_len: int=50,
        dropout_encoder: float=0, 
        dropout_decoder: float=0,
        dropout_pos_enc: float=0,
        dropout_regression: float = 0,
        dim_feedforward_encoder: int=128,
        dim_feedforward_decoder: int=128,
        ): 
        """_summary_

        Args:
            input_size (int): number of input variables/kinematics.
            num_predicted_features (int, optional): The number of gait states to predict. Defaults to 4.
            batch_first (bool, optional): whether to process/train the network using batch first inputs. Defaults to True.
            dim_val (int, optional): The dimension of the Transformer latent space. Defaults to 512.
            n_encoder_layers (int, optional): number of encoder layers. Defaults to 4.
            n_decoder_layers (int, optional): number of decoder layers. Defaults to 4.
            n_heads (int, optional): The number of parallel attention heads. Defaults to 8.
            enc_seq_len (int, optional): The maximum encoder sequence length. Defaults to 50.
            dropout_encoder (float, optional): the dropout rate of the encoder. Defaults to 0.
            dropout_decoder (float, optional): the dropout rate of the decoder Defaults to 0.
            dropout_pos_enc (float, optional): the dropout rate of the positional encoder. Defaults to 0.
            dropout_regression (float, optional): the dropout rate of the regression head. Defaults to 0.
            dim_feedforward_encoder (int, optional): number of neurons in the linear layer of the encoder. Defaults to 128.
            dim_feedforward_decoder (int, optional): number of neurons in the linear layer of the decoder. Defaults to 128.
        """    
    
        super().__init__() 


        # The input linear layer needed for the encoder
        self.embedding_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )


        # The encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation='gelu',
            batch_first=batch_first
            )
        
        # The input linear layer needed for the decoder
        self.decoder_embedding_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
            )
        
        #the decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation='gelu',
            batch_first=batch_first
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=enc_seq_len
            )
        
        # Stack the encoder layers in nn.TransformerEncoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )
        
        # Stack the decoder layers in nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )

        # Create the regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dim_val),
            nn.Dropout(dropout_regression),
            nn.Linear(dim_val,  num_predicted_features)
        )


    def forward(self, src: Tensor, tgt: Tensor, dts : Tensor) -> Tensor:
        """
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
           
           tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            
            dts: the vector of time steps used to encode the positions of the input
            
        """

        # Pass throguh the input layer right before the encoder
        src = self.embedding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features

        #encoder the positions
        src = self.positional_encoding_layer(src, dts) # src shape: [batch_size, src length, dim_val] regardless of number of input features

        ## ENCODER
        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
            )
       
       ## DECODER
        decoder_output = self.decoder_embedding_layer(tgt) # tgt shape: [batch_size, target sequence length, dim_val] regardless of number of input features
        
        # Pass through decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src
            )
        
        # Pass through regression head
        output = self.regression_head(decoder_output)
        
        # force the outputs of is_stairs to be in [0,1] via sigmoid
        output[:,:,4] = torch.sigmoid(output[:,:,4])

        return output

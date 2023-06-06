# Transformers-Based Deep Learning Gait State Estimator

This project develops a neural network that learns the characteristics of human gaits using measurements from the [Dephy](https://www.dephy.com) ExoBoot. Gait is parametrized by the gait state (phase, speed, incline, is_stairs, is_moving), which encodes where a person is in the gait cycle, what their speed is, what grade the terrain is, and if they are walking on stairs.


## Prerequisites

Before we dive into the details, you need to have the following prerequisites:

- A basic understanding of Python
- Familiarity with Git
- Knowledge of PyTorch and its implementation of neural networks

## Getting Started
To get started, clone the Git repo onto your local machine. The repo contains various training and testing Jupyter notebooks in a folder called `training-testing-notebooks`, the primary neural network definitions are in `gait_transformer.py`, which contains the actual PyTorch implementation of the architecture used. Training of the various component neural networks is handled via a series of Jupyter notebooks in /training-eval-notebooks, which should run on a sufficiently powerful Linux computer or on a cloud platform (e.g. Vertex AI). Required libraries are pytorch, numpy, scipy, matplotlib The primary training script is called `train_gait_state_predictor.ipynb`, and generally, all the training scripts follow the same structure as this class: 1) loading the data 2) Instantiating the model 3) training the model 

## Data Loading
Data loading in the training process is handled by a custom `WindowedGaitDataset` class in the `datasets.py` file, which expects as input a Pandas dataframe containing kinematics data and gait state data in ordered columns. Users will modify the `WindowedGaitDataset` class to contain whatever kinematics data they have. 

    window_size = 150
    gait_data = pd.read_csv('your_filename_here')
    dataset = WindowedGaitDataset(gait_data=gait_data,
                                                window_size=window_size,
                                                transform=ToTensor())


In this example, the DataFrame `gait_data` should contain columns corresponding to the different variables of interest, e.g. columns containing kinematics and gait states. The `WindowedGaitDataset` class contains a set of indices that denote which columns contain the kinematics and the gait state variables.


    #specify the indexes that contain the measured kinematics
    meas_idxs = [0,1,2,3,4,5,14]

    #specify the indexes that contain the gait states
    #phase, speed, incline, is_stairs
    gait_state_idxs = [6,7,8,9]    

    measurements = self.gait_data.iloc[idx-self.window_size+1:idx+1,meas_idxs].to_numpy()
    gait_states = self.gait_data.iloc[idx,gait_state_idxs].to_numpy()


The measurements will be input to the neural network in the order they are specified in, and the same goes for the gait state outputs; for real-time inference, the model will expect inputs in the same order and will produce outputs in the same order as during training, so make sure to be consistent.

By default, WindowedGaitDataset can be iterated on (like an array) to return a frame. This frame is a dict that contains kinematic measurements (frame['meas']) and the gait state itself (frame['state']). In the default implementation, the gait transformer expects in windowed buffer of kinematics data of `window_size`, such that at index i, WindowedGaitDataset returns kinematics measurements at indicies `i-window_size:i`, and the gait state at index i.

    i = 0
    dataset[i]['meas'] //first element of the dataset kinematics

The PyTorch DataLoader class used during training will gracefully handle sampling from the dataset using this indexing.


## Model Initialization

Model initialization involves specifying the model hyperparameters (e.g. number of layers) and instantiating the GaitTransformer object. 

    #Transformer parameters
    dim_val = 32 
    n_heads = 4
    n_encoder_layers = 4 
    n_decoder_layers = 4
    input_size = 7 # the number of input variables
    enc_seq_len = 150 
    dec_seq_len = 1 
    dim_feedforward_encoder = 512
    dim_feedforward_decoder = 512

    num_predicted_features = 5 # The number of output variables. 

    model = GaitTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        enc_seq_len=enc_seq_len,
        num_predicted_features=num_predicted_features,
        dim_feedforward_encoder=dim_feedforward_encoder,
        dim_feedforward_decoder=dim_feedforward_decoder,
    )


If necessary, move it to the GPU.

    device = torch.device("cuda")
    model.to(device)

Models should have a model nickname that makes it easier to differentiate the different saved networks. By default, the nickname specifies a folder which will contain model parameters/weights.

    model_nickname = 'apollyon-three-stairs'
    output_dir = f'../your_directory/{model_nickname}/model_save/'

This folder will contain the model checkpoints, the best model (i.e. the model with the lowest val loss), and the model at the final iteration. 

### Model Saving
Model saving is handled by the training script itself. Within the training script, a class called `SaveBestModel` handles the automatic saving of the best model thus far in training. Models are saved in a folder named after the model nickname. 

    best_model_name = best_model.tar' //name of the archive containing weights, etc.
    save_best_model = SaveBestModel(output_dir+best_model_name)


`SaveBestModel` wraps PyTorch's default save functionality. This saves a `tar` archive file containing keyed dictionary containing the model weights, the model checkpoint number, and the lowest validation loss. After every epoch, the script saves the model with the lowest overall validation error.

## Training
The training script contains by default the parameters used to train the `gait_transformer` in the paper. The training scheme is to use AdamW for 10 epochs. 

Here's a code snippet that shows how to train the neural network using the default settings. Training scripts generally follow this workflow.

    import torch

    # Create the data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create the neural network
    model = GaitTransformer(...)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train the neural network
    for epoch in range(10):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


## Evaluation:

The live controller is split across two separate scripts: run_gait_transformer and control_exo. This separation occurs as the neural networks are too computationally intensive to run on a Raspberry Pi in real time, and so must be executed on separate hardware, such as a Jetson Nano. This separate computer then communicates with the primary Pi via UDP `run_gait_transformer` runs the neural networks on a sufficiently powerful computer. control_exo runs on the primary Pi. Live evaluation of the scripts requires the [NeuroLocoMiddleware](https://pypi.org/project/NeuroLocoMiddleware/) package.






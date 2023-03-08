# Transformers-Based Deep Learning Gait State Esimator

This project develops a neural network that learns the characteristics of human gaits using measurements from the [Dephy](https://www.dephy.com) ExoBoot. Gait is parametrized by the gait state (phase, speed, incline, is_stairs), which encodes where a person is in the gait cycle, what their speed is, what grade the terrain is, and if they are walking on stairs.

Training: 

Training of the various component neural networks is handled via a series of Jupyter notebooks in /training-eval-notebooks, which should run on a sufficiently powerful Linux computer or on a cloud platform (e.g. Vertex AI). Required libraries are pytorch, numpy, scipy, matplotlib

Evaluation:

The live controller is split across two separate scripts: run_gait_transformer and control_exo. This separation occurs as the neural networks are too computationally intensive to run on a Raspberry Pi in real time, and so must be executed on separate hardware, such as a Jetson Nano. This separate computer then communicates with the primary Pi via UDP run_gait_transformer runs the neural networks on a sufficiently powerful computer. control_exo runs on the primary Pi. Live evaluation of the scripts requires the [NeuroLocoMiddleware](https://pypi.org/project/NeuroLocoMiddleware/) package.

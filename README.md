# This is the code behind the analysis for my Master's Thesis
*Neural Networks for Option Pricing: An empirical analysis of two design choices*

-----

File | Intent
---- | -------
data_Preprocessing.py | Main entry point and loop of the program
config.py | Configuration parameters such as local file paths and hyperparameters
actions.py | Most functions are defined here, including model fitting
models.py | ANN model architecture is created here
data_Preprocessing.py | Downloads and transforms Option Data into format convenient for training
data.py | Loads the preprocessed data and splits it into train/validation/test sets
plotting_actions.py | Anything to do with plotting during training process

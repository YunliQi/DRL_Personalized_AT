'''
Script to train model with only the current tumor size as input.
Run from the demo/ directory from virtual environment with drl_env.yml installed. (see [SetUp](../README.md#setup))
'''
import os
import sys
from time import sleep, asctime
import pandas as pd
import numpy as np
import random

sys.path.append("../utils")
from myUtils import convert_ode_parameters
from drlUtils import run_training

if __name__ == '__main__':
    # Environment parameters
    # DRL parameters
    gamma = 0.9999  # Discount rate for advantage estimation and reward discounting
    n_doseOptions = 2  # Agent can choose 2 different intensities of chemotherapy
    base = 0.1  # Base reward given
    hol = 0.05  # Addition for a treatment holiday
    punish = -0.1  # Punishment when tumor progresses
    day_interval = 7  # Interval between decision points (time step)
    learning_rate = 1e-4  # Learning rate
    num_workers = 8  #multiprocessing.cpu_count() # Set workers to number of available CPU threads
    max_epochs = 100000
    model_path = "../models"  # Path to save model
    model_name = "spatial_p125_weekly_spatial_dose_2_longrun"
    load_model = False
    logging_interval = 1000  # Will save the state of the network every logging_interval patients
    model_loaded_name = None
    verbose = 2
    
    # Tumour model parameters
    l = 100
    initialResistDen = 0.1
    initialCellDen = 0.5

    # initial_state = posCell = np.random.randint(0, l ** 2, size=round(initialCellDen * l ** 2))
    # posResistCell = random.sample(list(posCell), round(len(posCell) * initialResistDen))
    # initial_state = [0] * l ** 2
    # for pos in posCell:
    #     initial_state[pos] = 1
    # for pos in posResistCell:
    #     initial_state[pos] = 2
    # initial_state = np.array(initial_state).reshape(l, l)
    initial_state = np.zeros((100, 100))
    pos = []
    for i in range(100):
        for j in range(100):
            pos.append((i, j))
    square_pos = []
    for i in range(25, 75):
        for j in range(25, 75):
            square_pos.append((i, j))
    for i in square_pos:
        pos.remove(i)
    selected_pos = random.sample(pos, 2500)
    for i in selected_pos:
        initial_state[i] = 1
    for i in square_pos:
        initial_state[i] = 1
    selected_pos = random.sample(square_pos, 1000)
    for i in selected_pos:
        initial_state[i] = 2

    trainingDataDf = pd.DataFrame(
        {
            'PatientId': [125],
            'DMax': [2],
            'l': [l],
            'rS': [0.027],
            'cR': [0.25],
            'dCell': [0.25],
            'dDrug': [0.75],
            'initialCellDen': [initialCellDen],
            'initialResistDen': [initialResistDen],
            'initialDist': [initial_state]
        }
    )

    # Run training
    run_training(training_patients_df=trainingDataDf,
                architecture_kws={'n_values_size':1, 'n_values_delta':0, 'architecture':[128, 64, 32, 16, 10], 'n_doseOptions':n_doseOptions},
                reward_kws={'gamma':gamma, 'base':base, 'hol':hol, 'punish':punish},
                learning_rate=learning_rate, updating_interval=day_interval, num_workers=num_workers, max_epochs=max_epochs, model_name=model_name,
                load_model=load_model, model_loaded_name=model_loaded_name, model_path=model_path, logging_interval=logging_interval, verbose=verbose)

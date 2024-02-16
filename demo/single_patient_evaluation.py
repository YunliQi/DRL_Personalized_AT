import os
import sys
from time import sleep, asctime
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import PillowWriter, FuncAnimation

sys.path.append("../utils")
from drlUtils import run_evaluation

if __name__ == '__main__':
    # Environment parameters
    # DRL parameters
    n_doseOptions = 2
    day_interval = 7  # Interval between decision points (time step)
    model_path = "../models/test_currSizeOnly_p25_step1/100000_patients_test_currSizeOnly_p25_step1"  # Path to save model
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
    # trainingDataDf = pd.read_csv("../models/trainingPatientsDf_bruchovsky.csv", index_col=0)
    # trainingDataDf = trainingDataDf[trainingDataDf.PatientId==25]


    # Run training
    plot_data = run_evaluation(model_path=model_path, patients_to_evaluate=trainingDataDf, 
                architecture_kws={'n_values_size':1, 'n_values_delta':0, 'architecture':[128, 64, 32, 16, 10], 'n_doseOptions':n_doseOptions},
                n_replicates=1, updating_interval=day_interval, 
                verbose=verbose)
    # test = []
    # for i in plot_data:
    #     test.append(np.sum(i == 10) + np.sum(i == 100))
    # print(np.sum(np.array(test)))

    # Number of frames in the GIF
    num_frames = len(plot_data)

    def update(frame_number):
        data = plot_data[frame_number]
        ax.clear()
        
        # Define a custom colormap
        cmap = mcolors.ListedColormap(['white', 'green', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create the plot
        ax.imshow(data, cmap=cmap, norm=norm)
        ax.axis('off')
        
        # You might need to adjust the legend for each frame or create it once outside the update function
        red_patch = mpatches.Patch(color='red', label='Sensitive')
        green_patch = mpatches.Patch(color='green', label='Resistant')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')

    # Create the initial figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames)

    # Save the animation as a GIF
    ani.save('transfer_learning.gif', writer=PillowWriter(fps=7))

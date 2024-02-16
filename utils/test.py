# import numpy as np
# from scipy.integrate import solve_ivp
# def exponential_decay(t, y): return -0.5 * y
# sol = solve_ivp(exponential_decay, [0, 10], [2, 5, 10], t_eval=[0, 10])
# stateVars = ['S', 'R']
# print(dict(zip(stateVars, sol.y)))

# import pandas as pd
# import numpy as np
# import random

# l = 100
# initialResistDen = 0.05
# initialCellDen = 0.5

# initial_state = posCell = np.random.randint(0, l ** 2, size=round(initialCellDen * l ** 2))
# posResistCell = random.sample(list(posCell), round(len(posCell) * initialResistDen))
# initial_state = [0] * l ** 2
# for pos in posCell:
#     initial_state[pos] = 1
# for pos in posResistCell:
#     initial_state[pos] = 2
# initial_state = np.array(initial_state).reshape(l, l)

# df = pd.DataFrame(
#     {
#         'PatientId': [125],
#         'DMax': [1],
#         'l': [l],
#         'rS': [0.027],
#         'cR': [0.25],
#         'dCell': [0.25],
#         'dDrug': [0.75],
#         'initialCellDen': [initialCellDen],
#         'initialResistDen': [initialResistDen],
#         'initialDist': [initial_state]
#     }
# )
# df.to_csv('./models/trainingSpatialPatients.csv')

# import tensorflow as tf
# print(tf.test.is_gpu_available())
# print(tf.__version__)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import random

# Set the size of the grid
grid_size = 100

# Generate grid data with values 0, 1, or 2
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

# Define a custom colormap
cmap = mcolors.ListedColormap(['white', 'green', 'red'])  # red for 0, green for 1, blue for 2
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the plot
plt.figure(figsize=(10, 10))
plt.imshow(data, cmap=cmap, norm=norm)

# Customize the plot
# plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
plt.axis('off')

# Create custom patches for the legend
red_patch = mpatches.Patch(color='red', label='Sensitive')
green_patch = mpatches.Patch(color='green', label='Resistant')

# Add the legend to the plot
plt.legend(handles=[red_patch, green_patch])

# Adjust layout to make room for the legend
# plt.tight_layout()

# Display the plot
plt.savefig('./test.png')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors as mcolors
# import matplotlib.patches as mpatches
# from matplotlib.animation import FuncAnimation, PillowWriter

# # Assume plot_data is a list of numpy arrays
# plot_data = [np.random.choice([0, 1, 2], size=(100, 100)) for _ in range(10)]  # Example data
# num_frames = len(plot_data)

# def update(frame_number):
#     data = plot_data[frame_number]
#     ax.clear()
    
#     # Define a custom colormap
#     cmap = mcolors.ListedColormap(['white', 'green', 'red'])
#     bounds = [-0.5, 0.5, 1.5, 2.5]
#     norm = mcolors.BoundaryNorm(bounds, cmap.N)

#     # Create the plot
#     ax.imshow(data, cmap=cmap, norm=norm)
#     ax.axis('off')
    
#     # You might need to adjust the legend for each frame or create it once outside the update function
#     red_patch = mpatches.Patch(color='red', label='Sensitive')
#     green_patch = mpatches.Patch(color='green', label='Resistant')
#     ax.legend(handles=[red_patch, green_patch], loc='upper right')

# # Create the initial figure and axes
# fig, ax = plt.subplots(figsize=(10, 10))

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames)

# # Save the animation as a GIF
# ani.save('gif_test.gif', writer=PillowWriter(fps=7))

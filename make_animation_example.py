import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Time setup
dt = 0.01
t = np.arange(0, 5 + dt, dt)  # include endpoint

# Phi calculation
phi = np.cumsum(np.ones_like(t) * np.pi / 180) % (2 * np.pi)

# Positions
pos_x = np.cos(phi)
pos_y = np.sin(phi)
pos_z = np.zeros_like(phi) + 2
pos = np.column_stack((pos_x, pos_y, pos_z))

# Reference vectors
ref_X = np.array([1, 0, 0])
ref_Y = np.array([0, 1, 0])
ref_Z = np.array([0, 0, 1])

# Set up the figure and axis
fig = plt.figure('Quiver moving')
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(pos_x, pos_y, pos_z, 'k', linewidth=1)  # trajectory

# Create quiver objects
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)  # X-axis
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)  # Y-axis
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)  # Z-axis

targ_X_hlr = ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)
targ_Y_hlr = ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)
targ_Z_hlr = ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)

# Axis settings
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])
ax.grid(True)

def update_quivers(num):
    """Update the quivers in animation"""
    theta = (num * np.pi / 180) % (2 * np.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    targ_X = R @ ref_X
    targ_Y = R @ ref_Y
    targ_Z = R @ ref_Z

    targ_X_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_X[0], pos_y[num] + targ_X[1], pos_z[num] + targ_X[2])]])
    targ_Y_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Y[0], pos_y[num] + targ_Y[1], pos_z[num] + targ_Y[2])]])
    targ_Z_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Z[0], pos_y[num] + targ_Z[1], pos_z[num] + targ_Z[2])]])
    ax.set_title('Quiver Simulation Time : {:.2f}s'.format(t[num]))

# Creating animation
anim = FuncAnimation(fig, update_quivers, frames=len(t), interval=0.1)
# fig : figure object
# func : function that will be iterated
# frames : number of iteration
# interval : delay time between frames in millisecond
# blit : option for optimizing drawing (don't use right now)

# To save the animation, uncomment the following line:
# anim.save('quiver_moving.gif', writer='pillow')

# To save the animation as mp4, uncomment the following line:
# https://blog.naver.com/ahn_ss75/222671709830

plt.show()

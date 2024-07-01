import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
TIMETICK =0.1
BOBSIZE = 15
WINDOWSIZE = 250
PIVOT = (WINDOWSIZE / 2, WINDOWSIZE / 10)
SWINGLENGTH = PIVOT[1] * 4
TIME_DURATION = 10  # Total time duration of the simulation

class BobMass:
    def __init__(self):
        self.theta = 45  # Starting angle in degrees
        self.dtheta = 0  # Starting angular velocity

    def recompute_angle(self):
        scaling = 3000.0 / (SWINGLENGTH**2)
        first_ddtheta = -math.sin(math.radians(self.theta)) * scaling
        mid_dtheta = self.dtheta + first_ddtheta
        mid_theta = self.theta + (self.dtheta + mid_dtheta) / 2.0

        mid_ddtheta = -math.sin(math.radians(mid_theta)) * scaling
        mid_dtheta = self.dtheta + (first_ddtheta + mid_ddtheta) / 2
        mid_theta = self.theta + (self.dtheta + mid_dtheta) / 2

        mid_ddtheta = -math.sin(math.radians(mid_theta)) * scaling
        last_dtheta = mid_dtheta + mid_ddtheta
        last_theta = mid_theta + (mid_dtheta + last_dtheta) / 2.0

        last_ddtheta = -math.sin(math.radians(last_theta)) * scaling
        last_dtheta = mid_dtheta + (mid_ddtheta + last_ddtheta) / 2.0
        last_theta = mid_theta + (mid_dtheta + last_dtheta) / 2.0

        self.dtheta = last_dtheta
        self.theta = last_theta

    def update(self):
        self.recompute_angle()
    

bob = BobMass()

# Prepare the plot
fig, ax = plt.subplots()
ax.set_xlim(0, WINDOWSIZE)
ax.set_ylim(0, WINDOWSIZE)
ax.set_aspect('equal', adjustable='box')
line, = ax.plot([], [], 'k-', lw=2)
point, = ax.plot([], [], 'bo', markersize=BOBSIZE)

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    bob.update()
    x = PIVOT[0] - SWINGLENGTH * math.sin(math.radians(bob.theta))
    y = PIVOT[1] + SWINGLENGTH * math.cos(math.radians(bob.theta))
    line.set_data([PIVOT[0], x], [PIVOT[1], y])
    point.set_data([x], [y])  # Modified to pass x, y as lists
    return line, point

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=int(TIME_DURATION / TIMETICK), interval=TIMETICK * 1000, blit=True)

# Display
plt.show()
# Save as GIF
# anim.save('GIF/pendulum_test.gif', writer='pillow')  # Changed to 'pillow' for wider compatibility
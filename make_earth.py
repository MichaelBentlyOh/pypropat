import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# load bluemarble with PIL
bm = PIL.Image.open('earth.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
bm = np.array(bm.resize([int(d/2) for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
R = 6371
x = np.outer(np.sqrt(R)*np.cos(lons), np.sqrt(R)*np.cos(lats)).T
y = np.outer(np.sqrt(R)*np.sin(lons), np.sqrt(R)*np.cos(lats)).T
z = np.outer(np.sqrt(R)*np.ones(np.size(lons)), np.sqrt(R)*np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

plt.show()

#reference : https://stackoverflow.com/questions/30269099/how-to-plot-a-rotating-3d-earth
import numpy as np
import matplotlib.pyplot as plt

#Example for 3D plotting with matplotlib

# example 1 : random point plotting
dim3D = plt.axes(projection="3d")
xData = np.random.randint(0,100,(500,))
yData = np.random.randint(0,100,(500,))
zData = np.random.randint(0,100,(500,))

dim3D.scatter(xData,yData,zData,marker="o",alpha=0.5)
plt.show()

# example 2 : plotting graph with labeling
# dim3D = plt.axes(projection="3d")
# xData = np.arange(0,50,0.1)
# yData = np.arange(0,50,0.1)
# zData = np.sin(xData) * np.cos(yData)

# dim3D.set_xlabel("X")
# dim3D.set_ylabel("Y")
# dim3D.set_zlabel("Z") # there exists set_zlabel, just not showing in shortcut option
# dim3D.set_title("Example 2")

# dim3D.plot(xData,yData,zData)
# plt.show()

# example 3 : subplotting
# fig = plt.figure(figsize=(10, 3),dpi=100) # window size | graph size

# dim3D_0 = fig.add_subplot(121, projection="3d") # 1 raw 2 column's 1 element 
# dim3D_1 = fig.add_subplot(122, projection="3d")

# dim3D_0_xData = np.random.randint(0,100,(500,))
# dim3D_0_yData = np.random.randint(0,100,(500,))
# dim3D_0_zData = np.random.randint(0,100,(500,))
# dim3D_0.scatter(dim3D_0_xData,dim3D_0_yData,dim3D_0_zData,marker="o",alpha=0.5)

# dim3D_1_xData = np.arange(0,50,0.1)
# dim3D_1_yData = np.arange(0,50,0.1)
# dim3D_1_zData = np.sin(dim3D_1_xData) * np.cos(dim3D_1_yData)

# dim3D_1.set_xlabel("X")
# dim3D_1.set_ylabel("Y")
# dim3D_1.set_zlabel("Z")
# dim3D_1.set_title("Example 2")

# dim3D_1.plot(dim3D_1_xData,dim3D_1_yData,dim3D_1_zData)

# plt.show()

# example 4 : drawing a origin quiver3d(arrow3d)
# fig = plt.figure()
# ax = plt.axes(projection="3d")

# ax.quiver(0, 0, 0, 1, 0, 0,color='r')
# ax.quiver(0, 0, 0, 0, 1, 0,color='g')
# ax.quiver(0, 0, 0, 0, 0, 1,color='b')

# ax.set_xlim3d([-2.0, 2.0])
# ax.set_ylim3d([-2.0, 2.0])
# ax.set_zlim3d([-2.0, 2.0])

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# example 5 : drawing multi quiver objects
# fig = plt.figure('Quiver Basic 2')
# ax = fig.add_subplot(111, projection='3d')

# # Origin arrows
# ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)  # X-axis
# ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)  # Y-axis
# ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)  # Z-axis

# # Arrows from point (1, 2, 3)
# ax.quiver(1, 2, 3, 1, 0, 0, color='r', linewidth=1)  # X-axis
# ax.quiver(1, 2, 3, 0, 1, 0, color='g', linewidth=1)  # Y-axis
# ax.quiver(1, 2, 3, 0, 0, 1, color='b', linewidth=1)  # Z-axis

# ax.set_xlabel('Pos X [m]')
# ax.set_ylabel('Pos Y [m]')
# ax.set_zlabel('Pos Z [m]')

# # Setting the aspect ratio for the axes
# ax.set_xlim([-4, 4])
# ax.set_ylim([-4, 4])
# ax.set_zlim([-4, 4])

# # Adding grid
# ax.grid(True)

# plt.show()

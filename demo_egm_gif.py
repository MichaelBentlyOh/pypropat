import numpy as np
import kinematics
import orbit
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import PIL
from matplotlib.animation import FuncAnimation

# egm_demo
# Program to demonstrate how to use the Earth Gravity Model functions
# and the numeric orbit propagator

deg2rad = np.pi/180

### Initial condition (meters, radians) ###
kepel = np.array([7000000, 0.01, 98*deg2rad, 0, 35*deg2rad, 0])

stat = orbit.kepel_statvec(kepel)

delk = orbit.delkep(kepel)

year = 2017
mjd = orbit.djm(17, 7, year)

dfra = orbit.time_to_dayf(23,0,0)

tstart = 0
tstep = 50
tend = 6000
n = int(np.fix(tend/tstep))+1

orbit.egm_read_data('egm_10.dat')

# data storage
z1      	= np.zeros((1,n))
z3      	= np.concatenate([z1, z1, z1], 0)
z1 = z1.squeeze(0)
r_time      = copy.deepcopy(z1)
r_xo        = copy.deepcopy(z3)
r_vo        = copy.deepcopy(z3)
r_sma       = copy.deepcopy(z1)
r_ecc       = copy.deepcopy(z1)
r_inc       = copy.deepcopy(z1)
r_raan      = copy.deepcopy(z1)
r_par       = copy.deepcopy(z1)
r_ma        = copy.deepcopy(z1)
r_dist      = copy.deepcopy(z1)
r_rx        = copy.deepcopy(z1)
r_ry        = copy.deepcopy(z1)
r_rz        = copy.deepcopy(z1)

dist_acc = np.zeros(3)
cont_acc = np.zeros(3)

ic = 0
len = 3
# Orbit propagation
for t in np.arange(tstart, tend+tstep, tstep):
	print(t)
	# Analytical orbit propagation
	kp_an = kepel + delk*t

	# Convert from keplerian elements to state vector
	sv_an = orbit.kepel_statvec(kp_an).squeeze(0)

	xi_an = sv_an[0:3]
	vi_an = sv_an[3:6]

	# Orbit reference frame rotation matrix
	c_i_o = orbit.orbital_to_inertial_matrix(kp_an)

	# tspan = np.array([t,t+tstep/2,t+tstep])
	tspan = np.linspace(t, t+tstep, 50)

	ext_acc = dist_acc + cont_acc

	def func(t, x, mjd=mjd, dsec=dfra, ext_acc=ext_acc):
		return orbit.egm_difeq(t, x, mjd, dsec, ext_acc)

	sol = solve_ivp(func,(t, t+tstep), tuple(stat.squeeze(0)),rtol=1e-12, atol=1e-12)
	Y = sol.y

	sv_nm = Y[:,-1]	# propagated state vector
	# print(sv_nm)
	xi_nm = sv_nm[0:3]				# propagated inertial posititon vector
	vi_nm = sv_nm[3:6]				# propagated inertial velocity vector
	stat = sv_nm.reshape((1,6))		# state vector update

	# numerically propagated keplerian elements
	kp_nm = orbit.statvec_kepel(np.transpose(sv_nm))

	# eccentric anomaly
	ea_nm = orbit.kepler(kp_nm[5], kp_nm[1])

	# geocentric distance
	dist = kp_nm[0]*(1-kp_nm[1]*np.cos(ea_nm))

	# orbit control acceleration (if any)
	cont_acc = np.array([0,0,0])

	# disturbance specific forces (if any)
	dist_acc = np.array([0,0,0])

	# Store data to be plotted
	r_time[ic] = t
	r_xo[:,ic] = np.dot(np.transpose(c_i_o),(xi_nm - xi_an))/1000
	r_vo[:,ic] = np.dot(np.transpose(c_i_o),(vi_nm - vi_an))
	r_dist[ic] = dist/1000
	r_rx[ic] = sv_nm[0]
	r_ry[ic] = sv_nm[1]
	r_rz[ic] = sv_nm[2]
	# r_rx[ic] = kp_nm[0]*(np.cos(ea_nm)-kp_nm[1])/1000
	# r_ry[ic] = kp_nm[0]*np.sin(ea_nm)*np.sqrt(1-np.power(kp_nm[1],2))/1000
	r_sma[ic] = kp_nm[0] - kp_an[0]/1000
	r_ecc[ic] = kp_nm[1] - kp_an[1]
	r_inc[ic] = kp_nm[2] - kp_an[2]
	r_raan[ic] = kp_nm[3] - kp_an[3]
	r_par[ic] = kp_nm[4] - kp_an[4]
	r_ma[ic] = orbit.proximus(kp_nm[5], kp_an[5]) - kp_an[5]
	ic = ic+1

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
R = 3000
x = np.outer(np.sqrt(R)*np.cos(lons), np.sqrt(R)*np.cos(lats)).T
y = np.outer(np.sqrt(R)*np.sin(lons), np.sqrt(R)*np.cos(lats)).T
z = np.outer(np.sqrt(R)*np.ones(np.size(lons)), np.sqrt(R)*np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

# Time setup
t = np.arange(tstart, tend+tstep, tstep)  # include endpoint

# Reference vectors
ref_X = np.array([1, 0, 0])
ref_Y = np.array([0, 1, 0])
ref_Z = np.array([0, 0, 1])

# Set up the figure and axis
# ax = fig.add_subplot(111, projection='3d')
ax.plot3D(r_rx/1000, r_ry/1000, r_rz/1000, 'k', linewidth=1)  # trajectory

# Create quiver objects
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)  # X-axis
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)  # Y-axis
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)  # Z-axis

targ_X_hlr = ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)
targ_Y_hlr = ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)
targ_Z_hlr = ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)

# Axis settings
ax.set_xlim([-8000, 8000])
ax.set_ylim([-8000, 8000])
ax.set_zlim([-8000, 8000])
ax.grid(True)

# Positions
pos_x = r_rx/1000
pos_y = r_ry/1000
pos_z = r_rz/1000

def update_quivers(num):
    """Update the quivers in animation"""
    theta = tend/2*np.pi*tstep*num
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    targ_X = R @ ref_X*1000
    targ_Y = R @ ref_Y*1000
    targ_Z = R @ ref_Z*1000

    targ_X_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_X[0], pos_y[num] + targ_X[1], pos_z[num] + targ_X[2])]])
    targ_Y_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Y[0], pos_y[num] + targ_Y[1], pos_z[num] + targ_Y[2])]])
    targ_Z_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Z[0], pos_y[num] + targ_Z[1], pos_z[num] + targ_Z[2])]])
    ax.set_title('Propagation Simulation Time : {:.2f}s'.format(t[num]))

# Creating animation
anim = FuncAnimation(fig, update_quivers, frames=n, interval=0.001)

plt.show()








plt.show()









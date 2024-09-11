import numpy as np
from orbit import *
from ADCS import *
from kinematics import *
from scipy import integrate
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

# demo)propat
# Program to show how to use PROPAT
# Type demo_propat from the Matlab prompt

# Orbit keplerian elements:
kepel = np.array([7000000, 0.01, 98, 0, 0, 0])

# Orbit state vector:
stat = kepel_statvec(kepel)

# Attitude elements in Euler angles of a 3-1-3 (z-x-z) rotation
eulzxz = np.array([30, 50, 20])*np.pi/180

# Attitude in quaternions
quat = ezxzquat(eulzxz)

# Angular velocity vector in body frame:
w_ang = np.array([0.1, 0, 0.5])           # in radians/sec

# Compute the variations in keplerian elements due to the Earth oblateness
delk = delkep(kepel)

year = 2009
mjd = djm(13, 4, year)     # Format (day, month, year)
mjdo = djm(1, 1, year)     # modified julian date of 1/1/year
mjd1 = djm(1, 1, year+1)   # modified julian date of 1/1/(year+1)
year_frac = year + (mjd - mjdo)/(mjd1 - mjdo)  # year and fraction

# Ephemerides time:
dfra = time_to_dayf(10, 20, 0)    # UTC time in (hour, minute, sec)

# Propagation time in seconds:
tstart = 0     # initial time (sec)
tstep = 0.5    # step time (sec)
tend = 600     # end time (10 minutes)

# Inertia matrix of axis-symetric rigid body:
iner = np.array([[8, 0, 0,], [0, 8, 0,], [0, 0, 12]])      # in kg*m*m

# Inverse inertia matrix:
invin = np.linalg.inv(iner)

# Initial control torque:
contq = np.array([0, 0, 0])

# Magnetic moment torque flag and moment:
flag_mag = 0   # 1=compute magnetic moment / 0=discard magnetic moment
mag_mom = np.array([0, 0, 0.1])     # in A.m

# Initial vectors
time = np.array([tstart])			# to store time
euler = np.transpose(np.array([eulzxz*180/np.pi]))  # Euler angles
omeg = np.transpose(np.array([w_ang]))              # Angular velocity
orbit = np.transpose(stat)              # Orbit elements (state vector)
keorb = np.transpose([kepel])             # Orbit elements (keplerian)

# Attitude and orbit propagation
for t in np.arange(tstart, tend+tstep, tstep):
	print(t)
	# Orbit	propagation
	kep2 = kepel + delk * t

	# To	convert from keplerian elements to state vector( if needed)
	stat = kepel_statvec(kep2)

	# Perturbation torques:
	ambt = np.array([0, 0, 0])

	# External torques (perturbation + control)
	ext_torq = ambt + contq

	# Initial attitude vector:
	att_vec = np.concatenate([quat, w_ang], 0)         # Transposed

	# ODE Solver parameters
	# tspan = np.array([t,t+tstep/2,t+tstep])
	tspan = np.linspace(t, t + tstep, 20)
	# Numeric integration (ODE45)
	if flag_mag == 0:
		def func(t, x, ext_torque=ext_torq, tensin=iner, teninv=invin):
			return rigbody(t, x, ext_torque, tensin, teninv)
		# print(func(t, att_vec))
		sol = solve_ivp(func, (t, t+tstep), tuple(att_vec), rtol=1e-12, atol=1e-12)
		Y = sol.y

	else:
		# To convert from inertial state vector to terrestrial vector
		geoc = inertial_to_terrestrial(gst(mjd, dfra + t), stat.squeeze(0))
		# Earth's magnetic field
		sphe = rectangular_to_spherical(geoc)
		alt = sphe[2] / 1000
		elong = sphe[0]
		colat = np.pi/2 - sphe[1]
		earth_field = 1E-9*igrf_field(year_frac, alt, colat, elong)
		def func(t, x, ext_torq, iner, invin, mag_mom, earth_field):
			return rigbody(t, x, ext_torq, iner, invin,	mag_mom, earth_field)
		sol = solve_ivp(func, (t, t + tstep), tuple(att_vec.squeeze(0)), rtol=1e-12, atol=1e-12)
		Y = sol.y

	att_vec = np.array(Y[:, -1])			# propagated attitude vector
	quat = np.array(att_vec[:4])		# propagated quaternion
	w_ang = att_vec[4:7]		# propagated angular velocity

	eulzxz = quatezxz(quat)		# euler angles

	# attitude control torque logic (if any)
	cont_torq = np.array([0, 0, 0])

	# Store data to be plotted
	time = np.concatenate([time, [t]], 0)
	euler = np.concatenate([euler, eulzxz*180/np.pi], 1)
	omeg = np.concatenate([omeg, np.transpose([w_ang])], 1)
	orbit = np.concatenate([orbit, np.transpose(stat)], 1)
	keorb = np.concatenate([keorb, np.transpose([kep2])], 1)

# Output visualization
plt.figure()
plt.plot(time, euler[0, :], 'r', label='phi')
plt.plot(time, euler[1, :], 'g', label='theta')
plt.plot(time, euler[2, :], 'b', label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Euler angles (3-1-3) (deg)')
plt.title('Attitude in Euler angles')
plt.legend()

plt.figure()
plt.plot(time, omeg[0, :], 'r', label='w_bx')
plt.plot(time, omeg[1, :], 'g', label='w_by')
plt.plot(time, omeg[2, :], 'b', label='w_bz')
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Attitude angular velocity')
plt.legend()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, keorb[3, :], 'b',)
plt.xlabel('Time (s)')
plt.ylabel('Ascencion node (deg)')
plt.title('Right ascencion of the ascending node')

plt.subplot(3, 1, 2)
plt.plot(time, keorb[4, :], 'b')
plt.xlabel('Time (s)')
plt.ylabel('Perigee argument (deg)')
plt.title('Perigee argument')

plt.subplot(3, 1, 3)
plt.plot(time, keorb[5, :], 'b')
plt.xlabel('Time (s)')
plt.ylabel('Mean anomaly (deg)')
plt.title('Mean anomaly')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, orbit[0, :]/1000, 'r', label='x')
plt.plot(time, orbit[1, :]/1000, 'g', label='y')
plt.plot(time, orbit[2, :]/1000, 'b', label='z')
plt.xlabel('Time (s)')
plt.ylabel('Position (km)')
plt.title('Satellite inertial position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, orbit[3, :]/1000, 'r', label='vx')
plt.plot(time, orbit[4, :]/1000, 'g', label='vy')
plt.plot(time, orbit[5, :]/1000, 'b', label='vz')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.title('Satellite velocity')
plt.legend()

plt.show()









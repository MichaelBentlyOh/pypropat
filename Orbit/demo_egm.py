import numpy as np
import kinematics
import orbit
from scipy import integrate
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

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
tstep = 1
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

dist_acc = np.zeros(3)
cont_acc = np.zeros(3)

ic = 0
len = 3
# Orbit propagation
for t in np.arange(tstart, tend+tstep, tstep):
	print(ic)
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

	sol = solve_ivp(func,(t, t+tstep),tuple(stat.squeeze(0)),rtol=1e-12, atol=1e-12)
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
	# r_rx[ic] = kp_nm[0]*(np.cos(ea_nm)-kp_nm[1])/1000
	# r_ry[ic] = kp_nm[0]*np.sin(ea_nm)*np.sqrt(1-np.power(kp_nm[1],2))/1000
	r_sma[ic] = kp_nm[0] - kp_an[0]/1000
	r_ecc[ic] = kp_nm[1] - kp_an[1]
	r_inc[ic] = kp_nm[2] - kp_an[2]
	r_raan[ic] = kp_nm[3] - kp_an[3]
	r_par[ic] = kp_nm[4] - kp_an[4]
	r_ma[ic] = orbit.proximus(kp_nm[5], kp_an[5]) - kp_an[5]
	ic = ic+1


# Plotting satellite position in orbit frame
plt.figure()
plt.plot(r_time, r_xo[0, :], 'r', label='X position')
plt.plot(r_time, r_xo[1, :], 'g', label='Y position')
plt.plot(r_time, r_xo[2, :], 'b', label='Z position')
plt.xlabel('Time (s)')
plt.ylabel('Satellite position (km)')
plt.title('Satellite position in orbit frame')
plt.legend()

# Plotting satellite velocity in orbit frame
plt.figure()
plt.plot(r_time, r_vo[0, :], 'r', label='X velocity')
plt.plot(r_time, r_vo[1, :], 'g', label='Y velocity')
plt.plot(r_time, r_vo[2, :], 'b', label='Z velocity')
plt.xlabel('Time (s)')
plt.ylabel('Satellite velocity (m/s)')
plt.title('Satellite velocity in orbit frame')
plt.legend()

# Plotting geocentric distance
plt.figure()
plt.plot(r_time, r_dist)
plt.xlabel('Time (s)')
plt.ylabel('Distance (km)')
plt.title('Geocentric distance')

# Plotting orbit in the orbit plane
plt.figure()
plt.plot(r_rx, r_ry)
plt.xlabel('Orbit plane - x (km)')
plt.ylabel('Orbit plane - y (km)')
plt.title('Orbit')

# Plotting satellite position in orbit plane (along track vs zenith)
plt.figure()
plt.plot(r_xo[1, :], r_xo[0, :])
plt.xlabel('Along track position (km)')
plt.ylabel('Zenith position (km)')
plt.title('Satellite position in orbit plane')

# Plotting cross track satellite position (cross track vs zenith)
plt.figure()
plt.plot(r_xo[2, :], r_xo[0, :])
plt.xlabel('Cross track position (km)')
plt.ylabel('Zenith position (km)')
plt.title('Cross track satellite position')

# Plotting semi major axis variation
plt.figure()
plt.plot(r_time, r_sma)
plt.xlabel('Time (s)')
plt.ylabel('Relative semi major axis (km)')
plt.title('Semi major axis variation')

# Plotting eccentricity variation
plt.figure()
plt.plot(r_time, r_ecc)
plt.xlabel('Time (s)')
plt.ylabel('Relative eccentricity')
plt.title('Eccentricity variation')

# Plotting orbit inclination variation
plt.figure()
plt.plot(r_time, r_inc / deg2rad)
plt.xlabel('Time (s)')
plt.ylabel('Relative inclination (deg)')
plt.title('Orbit inclination variation')

# Plotting right ascension of ascending node variation
plt.figure()
plt.plot(r_time, r_raan / deg2rad)
plt.xlabel('Time (s)')
plt.ylabel('Relative right ascension (deg)')
plt.title('Right ascension of ascending node variation')

# Plotting perigee argument variation
plt.figure()
plt.plot(r_time, r_par / deg2rad)
plt.xlabel('Time (s)')
plt.ylabel('Relative perigee argument (deg)')
plt.title('Perigee argument variation')

# Plotting mean anomaly variation
plt.figure()
plt.plot(r_time, r_ma / deg2rad)
plt.xlabel('Time (s)')
plt.ylabel('Relative mean anomaly (deg)')
plt.title('Mean anomaly variation')

plt.show()









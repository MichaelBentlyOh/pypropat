import numpy as np
import Kinematics
import Orbit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rad = np.pi / 180
deg = 1 / rad

fid = open("propov.txt", 'wt')

# Orbit keplerian elements:
kepel = np.array([7000000, 0.01, 98 * rad, 0, 0, 0])
# Orbit state vector:
stat = Orbit.kepel_statvec(kepel)

# Compute the variations in keplerian elements due to the Earth oblateness
delk = Orbit.delkep(kepel)

# Ephemerides date in Modified Julian date:
year = 2009
mjd = Orbit.djm(13, 4, year)  # Format (day, month, year)
mjdo = Orbit.djm(1, 1, year)  # modified julian date of 1/1/year
mjd1 = Orbit.djm(1, 1, year + 1)  # modified julian date of 1/1/(year+1)
year_frac = year + (mjd - mjdo) / (mjd1 - mjdo)  # year and fraction

# Ephemerides time:
dfra = Orbit.time_to_dayf(10, 20, 0)  # UTC time in (hour, minute, sec)

tini = 0
tstep = 1 / 15
tend = 60.

massa = 1  
lx = 0.1  
tensin = massa * lx * lx / 6 * np.array([[1, 0.0, 0.0],
                                         [0.0, 0.95, 0.0],
                                         [0.0, 0.0, 1.05]])


teninv = np.linalg.inv(tensin)


ext_torq = np.array([0, 0, 0]) 

torq_att = np.array([0, 0, 0])
torq_dis = np.array([0, 0, 0])
time_att = np.array([tini])

angular_velocity = np.array([2, 1, 3]) * np.pi / 30
euler_angles = np.array([60, 30, 40]) * rad 

ele = tensin @ angular_velocity

euler_att = euler_angles
euler_ant = euler_att

state_vector_in = np.concatenate((Kinematics.exyzquat(euler_angles), angular_velocity))

fid.write(f'{tstep:.11f}, {tend:.5f},\n')

fid.write(f'{tini:.3f}, <{euler_angles[0]*deg:.3f}, {euler_angles[1]*deg:.3f}, {euler_angles[2]*deg:.3f}>,\n')
# =============================================================================================================
def rigidbody(t, x, ext_torque, tensin, teninv,
              mag_moment=np.array([0, 0, 0]),
              magnetic_field=np.array([0, 0, 0])):
    q = x[0:4]
    w = x[4:7]  # Ensure w has the correct dimension
    xp = 0.5 * Kinematics.sangvel(w) @ q
    torque = ext_torque + np.cross(mag_moment, Kinematics.quatrmx(q) @ magnetic_field)
    wp = teninv @ (np.cross(tensin @ w, w) + torque)
    return np.concatenate((xp, wp))

def integrate_rigidbody(state_vector_in, tini, tend, tstep, ext_torq, tensin, teninv):
    t_span = np.arange(tini, tend, tstep)
    result = solve_ivp(rigidbody, [tini, tend], state_vector_in, t_eval=t_span,
                       args=(ext_torq, tensin, teninv))
    return result.t, result.y.T
# =============================================================================================================

# Perform the integration
times, states = integrate_rigidbody(state_vector_in, tini, tend, tstep, ext_torq, tensin, teninv)

# Process the result states here
for t, state in zip(times, states):
    # To convert from quaternions to Euler angles
    rot_mat = Kinematics.quatrmx(state[:4])
    euler_att = Kinematics.rmxexyz(rot_mat)
    
    for ind in range(3):
        if euler_att[ind] > np.pi:
            euler_att[ind] -= 2 * np.pi
        if euler_att[ind] < -np.pi:
            euler_att[ind] += 2 * np.pi
    
    euler_att = Kinematics.proximus(euler_att, euler_ant)
    euler_ant = euler_att
    euler_deg = euler_att * deg
    time_att = np.append(time_att, t)

    # Earth's magnetic field
    tsgr = Orbit.gst(mjd, dfra + t)
    geoc = Orbit.inertial_to_terrestrial(tsgr, state)
    sphe = Orbit.rectangular_to_spherical(geoc)
    alt = sphe[2] / 1000
    elong = sphe[0]
    colat = np.pi / 2 - sphe[1]
    earth_field = 1.e-9 * Orbit.igrf_field(year_frac,alt, colat, elong)
    earth_mag = Kinematics.rotmay(sphe[1] + np.pi / 2) @ Kinematics.rotmaz(-elong) @ earth_field
    earth_mag = np.append(earth_mag, [0, 0, 0])
    earth_iner = Orbit.terrestrial_to_inertial(tsgr, earth_mag)
    b_sat = rot_mat @ earth_iner[:3]
    
    # Sun position
    sun_sat = Orbit.sun(mjd, dfra + t)
    sun_sat = rot_mat @ sun_sat[:3]

    # Write the new attitude to file
    fid.write(f'{t + tstep:.3f}, <{euler_deg[0]:.3f}, {euler_deg[1]:.3f}, {euler_deg[2]:.3f}>,\n')

fid.close()

# Example plots (similar to the MATLAB plotting section)
time_att = np.array(time_att)
euler_angles_array = np.array([Kinematics.rmxexyz(Kinematics.quatrmx(state[:4])) * deg for state in states]).T
angular_velocity_array = np.array([state[4:7] * 30 / np.pi for state in states]).T

time_att = time_att[:len(euler_angles_array[0])]

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time_att, euler_angles_array[0], 'r', label='Euler Angle X')
plt.plot(time_att, euler_angles_array[1], 'g', label='Euler Angle Y')
plt.plot(time_att, euler_angles_array[2], 'b', label='Euler Angle Z')
plt.legend()
plt.title('Euler Angles Over Time')

plt.subplot(3, 1, 2)
plt.plot(time_att, angular_velocity_array[0], 'r', label='Angular Velocity X')
plt.plot(time_att, angular_velocity_array[1], 'g', label='Angular Velocity Y')
plt.plot(time_att, angular_velocity_array[2], 'b', label='Angular Velocity Z')
plt.legend()
plt.title('Angular Velocities Over Time')

plt.subplot(3, 1, 3)
plt.plot(time_att, np.tile(ext_torq[0], len(time_att)), 'r', label='Torque X')
plt.plot(time_att, np.tile(ext_torq[1], len(time_att)), 'g', label='Torque Y')
plt.plot(time_att, np.tile(ext_torq[2], len(time_att)), 'b', label='Torque Z')
plt.legend()
plt.title('Torques Over Time')

plt.show()
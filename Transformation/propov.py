import numpy as np
import Kinematics
import orbit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rad = np.pi / 180
deg = 1 / rad

fid = open("propov.txt", 'wt')

# Orbit keplerian elements:
kepel = np.array([7000000, 0.01, 98 * rad, 0, 0, 0])
# Orbit state vector:
stat = orbit.kepel_statvec(kepel)

# Compute the variations in keplerian elements due to the Earth oblateness
delk = orbit.delkep(kepel)

# Ephemerides date in Modified Julian date:
year = 2009
mjd = orbit.djm(13, 4, year)  # Format (day, month, year)
mjdo = orbit.djm(1, 1, year)  # modified julian date of 1/1/year
mjd1 = orbit.djm(1, 1, year + 1)  # modified julian date of 1/1/(year+1)
year_frac = year + (mjd - mjdo) / (mjd1 - mjdo)  # year and fraction

# Ephemerides time:
dfra = orbit.time_to_dayf(10, 20, 0)  # UTC time in (hour, minute, sec)

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

def sun(djm, ts):
    """
    Calculate the position vector of the Sun in the geocentric inertial system referred to the J2000 equator and equinox.

    Parameters:
    djm : float
        Modified Julian Date in days, referred to 1950.0.
    ts : float
        Fraction of the day in seconds.

    Returns:
    sunpos : numpy array
        Position vector of the Sun:
        [0] - First component of Earth-Sun position vector in meters.
        [1] - Second component of Earth-Sun position vector in meters.
        [2] - Third component of Earth-Sun position vector in meters.
        [3] - Right ascension in radians.
        [4] - Declination in radians.
        [5] - Radius vector (distance) in meters.
    """
    rad = np.pi / 180
    ASTRONOMICAL_UNIT = 149.60e+09  # Astronomical unit (meters)

    t = djm - 18262.5 + ts / 86400.0

    # Mean longitude of the Sun, corrected
    alom_ab = np.mod((280.460 + 0.9856474 * t) * rad, 2 * np.pi)
    if alom_ab < 0:
        alom_ab += 2 * np.pi

    # Mean anomaly
    an_mean = np.mod((357.528 + 0.9856003 * t) * rad, 2 * np.pi)
    if an_mean < 0:
        an_mean += 2 * np.pi

    an_mean_2 = an_mean + an_mean
    if an_mean_2 > 2 * np.pi:
        an_mean_2 = np.mod(an_mean_2, 2 * np.pi)

    ecli_lo = alom_ab + (1.915 * np.sin(an_mean) + 0.02 * np.sin(an_mean_2)) * rad
    sin_ecli_lo = np.sin(ecli_lo)
    cos_ecli_lo = np.cos(ecli_lo)

    # Ecliptic latitude
    obl_ecli = (23.439 - 4.e-7 * t) * rad
    sin_obl_ecli = np.sin(obl_ecli)
    cos_obl_ecli = np.cos(obl_ecli)

    sunpos = np.zeros(6)
    sunpos[3] = np.arctan2(cos_obl_ecli * sin_ecli_lo, cos_ecli_lo)
    if sunpos[3] < 0:
        sunpos[3] += 2 * np.pi

    sunpos[4] = np.arcsin(sin_obl_ecli * sin_ecli_lo)
    sunpos[5] = (1.00014 - 0.01671 * np.cos(an_mean) - 1.4e-4 * np.cos(an_mean_2)) * ASTRONOMICAL_UNIT

    sunpos[0] = sunpos[5] * cos_ecli_lo
    sunpos[1] = sunpos[5] * cos_obl_ecli * sin_ecli_lo
    sunpos[2] = sunpos[5] * sin_obl_ecli * sin_ecli_lo

    return sunpos

def igrf_field(filename, idx):
    # Open and read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Extract the 'idx'th row and split it into columns by comma
    row_data = lines[idx].strip().split(',')
    
    # Convert the necessary columns to floats and return them
    x, y, z = float(row_data[0]), float(row_data[1]), float(row_data[2])
    return np.array([x, y, z])

# Perform the integration
times, states = integrate_rigidbody(state_vector_in, tini, tend, tstep, ext_torq, tensin, teninv)

# Process the result states here
i = 0
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
    tsgr = orbit.gst(mjd, dfra + t)
    geoc = orbit.inertial_to_terrestrial(tsgr, state)
    sphe = orbit.rectangular_to_spherical(geoc)
    alt = sphe[2] / 1000
    elong = sphe[0]
    colat = np.pi / 2 - sphe[1]
    earth_field = 1.e-9 * igrf_field('igrf11.dat',i)
    i = i + 1
    earth_mag = Kinematics.rotmay(sphe[1] + np.pi / 2) @ Kinematics.rotmaz(-elong) @ earth_field
    earth_mag = np.append(earth_mag, [0, 0, 0])
    earth_iner = orbit.terrestrial_to_inertial(tsgr, earth_mag)
    b_sat = rot_mat @ earth_iner[:3]
    
    # Sun position
    sun_sat = sun(mjd, dfra + t)
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
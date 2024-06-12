import numpy as np
import kinematics
import copy
global egm_order, egm_length, egm_conv_f, egm_cc, egm_sc
global egm_ae, egm_gm, egm_pn, egm_qn, egm_ip, egm_nmax


def delkep(kep_el):
	# calculates the rate of variation of keplerian elements, considering only J2 and J4.
	### Input ###
	# kep_el
	# vector with the keplerian elements:
	# (0) - semimajor axis of the orbit in meters.
	# (1) - eccentricity.
	# (2) - inclination in radians.
	# (3) - right ascension of ascending node in radians.
	# (4) - argument of perigee in radians.
	# (5) - mean anomaly in radians.
	# Obs: 3 ,4 and 5 are not used
	#############

	EARTH_RADIUS = 6378139. 		# Earth's radius in meters
	EARTH_GRAVITY = 3.9860064e14	# Earth's gravitational constant [m3/s2]
	J_2 = 1.0826268362e-3			# = 484.16544e-6 * SQRT(5.e0)
	J_4 = -1.62336e-6				# = -0.54112e-6 * 3e0

	seix = kep_el[0]
	exce = kep_el[1]
	exc2 = np.power(exce,2)
	eta2 = 1. - exc2
	eta1 = np.sqrt(eta2)
	teta = np.cos(kep_el[2])
	tet2 = np.power(teta,2)
	tet4 = np.power(teta,4)
	aux0 = np.sqrt(EARTH_GRAVITY / (seix * seix * seix))
	plar = EARTH_RADIUS / (seix * eta2)
	gam2 = 0.5 * J_2 * np.power(plar,2)
	gam4 = -0.375 * J_4 * np.power(plar,4)

	deltakep = np.zeros(6)

	deltakep[3] = aux0*teta*(3.*gam2*(-1. + 0.125*gam2*(9.*eta2 + 12.*eta1 - \
					5. - (5.*eta2 + 36.*eta1 + 35.)*tet2)) + \
					1.25*gam4*(5. - 3.*eta2)*(3. - 7.*tet2));
	deltakep[4] = aux0*(1.5*gam2*((5.*tet2 - 1.) + \
	   				0.0625*gam2*(25.*eta2 + 24.*eta1 - \
	   				35. + (90. - 192.*eta1 - 126.*eta2)*tet2 + \
	   				(385. + 360.*eta1 + 45.*eta2)*tet4)) + \
	   				0.3125*gam4*(21. - 9.*eta2 + (-270. + 126.*eta2)*tet2 + \
	   				(385. - 189.*eta2)*tet4))
	deltakep[5] = aux0*(1. + eta1*(1.5*gam2*((3.*tet2 - 1.) + \
				    0.0625*gam2*(16.*eta1 + 25.*eta2 - 15. + \
				    (30. - 96.*eta1 - 90.*eta2)*tet2 + \
				    (105. + 144.*eta1 + 25.*eta2)*tet4)) + \
				    0.9375*gam4*exc2*(3. - 30.*tet2 + 35.*tet4)))

	return deltakep

def djm(day, month, year):
	# furnish the Modified Julian Date with reference to the day, month, and year at zero hours of the day
	diju = 367*year + day - 712269 + np.fix(275*month/9)- np.fix(7*(year+np.fix((month+9)/12))/4)
	return diju

def egm_acc(x):
	global egm_order, egm_length, egm_conv_f, egm_cc, egm_sc
	global egm_ae, egm_gm, egm_pn, egm_qn, egm_ip, egm_nmax
	r = np.linalg.norm(x)
	q = egm_ae/r
	t = x[2]/r
	u = np.sqrt(1-t*t)
	tf = t/u
	sc = np.sqrt(x[0]*x[0]+x[1]*x[1])
	if sc==0:
		sl = 0
		cl = 1
	else:
		sl = x[1]/sc
		cl = x[0]/sc
	gmr = egm_gm/r

	vl = 0.0
	vf = 0.0
	vr = 0.0

	egm_pn[0] = 1.0
	egm_pn[1] = 1.73205080756887730*u
	egm_qn[0] = 1.0
	egm_qn[1] = q

	for m in range(2, egm_nmax + 1):
		egm_pn[m] = u * np.sqrt(1.0 + 0.50 / m) * egm_pn[m - 1]
		egm_qn[m] = q * egm_qn[m - 1]

	# Initialize sin and cos recursions
	sm = 0.0
	cm = 1.0

	# Outer n loop
	for m in range(0, egm_nmax + 1):
		# Init
		pnm = egm_pn[m]  # m=n sectoral
		dpnm = -m * pnm * tf
		pnm1m = pnm
		pnm2m = 0.0

		# Init Horner's scheme
		qc = egm_qn[m] * egm_cc[int(egm_ip[m] + m)]
		qs = egm_qn[m] * egm_sc[int(egm_ip[m] + m)]
		#print(egm_ip[m])
		#print(m)
		#print(round(egm_ip[m]+m))
		#print(egm_ip)
		xc = qc * pnm
		xs = qs * pnm
		xcf = qc * dpnm
		xsf = qs * dpnm
		xcr = m * qc * pnm
		xsr = m * qs * pnm
		mm = m

		# Inner m loop
		for n in range(m + 1, egm_nmax + 1):
			nn = n
			anm = np.sqrt(((nn + nn - 1.0) * (nn + nn + 1.0)) /
						  ((nn - mm) * (nn + mm)))
			bnm = np.sqrt(((nn + nn + 1.0) * (nn + mm - 1.0) *
						   (nn - mm - 1.0)) / ((nn - mm) * (nn + mm) * (nn + nn - 3.0)))
			fnm = np.sqrt(((nn * nn - mm * mm) * (nn + nn + 1.0)) / (nn + nn - 1.0))

			# Recursion p and dp
			pnm = anm * t * pnm1m - bnm * pnm2m
			dpnm = -nn * pnm * tf + fnm * pnm1m / u  # Signal opposite to paper

			# Store
			pnm2m = pnm1m
			pnm1m = pnm

			# Inner sum
			if nn >= 2:
				qc = egm_qn[n] * egm_cc[int(egm_ip[n] + m)]
				qs = egm_qn[n] * egm_sc[int(egm_ip[n] + m)]
				xc = (xc + qc * pnm)
				xs = (xs + qs * pnm)
				xcf = (xcf + qc * dpnm)
				xsf = (xsf + qs * dpnm)
				xcr = (xcr + (nn + 1.0) * qc * pnm)
				xsr = (xsr + (nn + 1.0) * qs * pnm)

		# Outer sum
		vl = vl + mm * (xc * sm - xs * cm)
		vf = vf + (xcf * cm + xsf * sm)
		vr = vr + (xcr * cm + xsr * sm)
		# Sin and cos recursions to next m
		cml = cl * cm - sm * sl
		sml = cl * sm + cm * sl
		cm = cml  # Save to next m
		sm = sml  # Save to next m

	# Finalization, include n=0 (p00=1)
	# For n=1 all terms are zero: c,s(1,1), c,s(1,0) = 0

	# Gradient
	vl = -gmr * egm_conv_f * vl
	vf = gmr * egm_conv_f * vf
	vr = -(gmr / r) * (1.0 + egm_conv_f * vr)

	# Body x, y, z accelerations
	ac = np.array([
		u * cl * vr - t * cl * vf / r - sl * vl / (u * r),
		u * sl * vr - t * sl * vf / r + cl * vl / (u * r),
		t * vr + u * vf / r
	])

	return ac

def egm_difeq (t, x, mjd, dsec, ext_acc):
	xip = x[3:6]
	gwst = gst(mjd, dsec + t)
	se = inertial_to_terrestial(gwst, x)
	xe = se[0:3]
	ae = np.concatenate([egm_acc(xe),[0,0,0]],0)
	ai = terrestial_to_inertial(gwst,np.transpose(ae))
	vip = ext_acc + np.transpose(ai[0:3])
	dxdt = np.concatenate([xip,vip],0)
	return dxdt

def egm_read_data(egm_data_file, nmax=0):
	global egm_order, egm_length, egm_conv_f, egm_cc, egm_sc
	global egm_ae, egm_gm, egm_pn, egm_qn, egm_ip, egm_nmax
	funit = open(egm_data_file, mode = 'r')
	data = funit.readline().split()
	num_data = [float(num) for num in data]
	cf = funit.readlines()
	cf_data = np.zeros((np.shape(cf)[0], np.shape(cf[1].split())[0]))
	for i in range(0, np.shape(cf)[0]):
		# print([float(num) for num in cf[i].split()])
		cf_data[i] = [float(num) for num in cf[i].split()]
	cf_data = np.transpose(cf_data)

	egm_order = int(num_data[0])
	egm_length = copy.deepcopy(num_data[1])
	egm_length = (egm_order+2)*(egm_order+1)/2-3
	egm_conv_f = copy.deepcopy(num_data[2])

	egm_cc = np.concatenate([[0, 0, 0], cf_data[2]],0)
	egm_sc = np.concatenate([[0, 0, 0], cf_data[3]],0)

	egm_ae = 6378136.3
	egm_gm = 3986004.415e8
	egm_pn = np.zeros(egm_order+1)
	egm_qn = np.zeros(egm_order+1)
	egm_ip = np.zeros(egm_order+1)
	egm_ip[0] = 0

	for n in range(1,egm_order+1):
		egm_ip[n] = egm_ip[n-1] + n

	if nmax != 0:
		if nmax < egm_order:
			egm_nmax = nmax
		else:
			egm_nmax = copy.deepcopy(egm_order)
	else:
		egm_nmax = copy.deepcopy(egm_order)
	return

def gst(diju, time):
	tsj = (diju - 18262.5) / 36525
	tsgo = (24110.54841 + (8640184.812866 + 9.3104e-2 * tsj - 6.2e-6 * tsj * tsj) * tsj) * np.pi / 43200
	tetp = 7.292116e-5 	# velocidade angular da Terra(rd / s)
	gwst = np.mod(tsgo + time * tetp, 2 * np.pi)
	return gwst

def inertial_to_terrestial(tesig, xi):
	xterrestial = np.matmul(np.concatenate([[xi[0:3]],[xi[3:6]]],0),np.transpose(kinematics.rotmaz(tesig)))
	xterrestial = np.concatenate([xterrestial[0],xterrestial[1]],0)
	return xterrestial

def kepel_statvec(kepel):
	# transform the keplerian elements kepel into the corresponding state vector in the same reference system
	### Input ###
	# kep_el
	# vector with the keplerian elements:
	# (0) - semimajor axis of the orbit in meters.
	# (1) - eccentricity.
	# (2) - inclination in radians.
	# (3) - right ascension of ascending node in radians.
	# (4) - argument of perigee in radians.
	# (5) - mean anomaly in radians.
	# Obs: 3 ,4 and 5 are not used
	#############
	EARTH_GRAVITY = 3.9860064e14  # Earth's gravitational constant [m3/s2]

	a = kepel[0]	# semi-major axis
	exc = kepel[1]	# eccentricity

	c1 = np.sqrt(1 - exc*exc)

	orb2iner = np.matmul(kinematics.rotmaz(-kepel[3]),np.matmul(kinematics.rotmax(-kepel[2]),kinematics.rotmaz(-kepel[4])))

	E = kepler(kepel[5], exc)

	sE = np.sin(E)
	cE = np.cos(E)
	c3 = np.sqrt(EARTH_GRAVITY/a)/(1.-exc*cE)

	statevec1 = np.dot(np.array([[a*(cE-exc), a*c1*sE, 0]]),np.transpose(orb2iner))
	statevec2 = np.dot(np.array([[-c3*sE, c1*c3*cE, 0]]), np.transpose(orb2iner))
	statevec = np.concatenate([statevec1, statevec2],1)

	return statevec

def kepler(mean_anomaly, eccentricity):
	# Find a solution to the kepler's equation
	### Input ###
	# mean_anomaly : in radians
	# eccentricity
	#############
	exc2 = np.power(eccentricity,2)
	am = np.mod(mean_anomaly, 2*np.pi)

	shoot = am + eccentricity*(1. - 0.125*exc2)*np.sin(am) + \
			0.5*exc2*(np.sin(am+am) + 0.75*eccentricity*np.sin(am+am+am))
	shoot = np.mod(shoot, 2*np.pi)

	e1 = 1.0
	ic = 0

	while (np.abs(e1) > 1.e-12) and (ic <= 10):
		e1 = (shoot - am - eccentricity*np.sin(shoot))/(1.0-eccentricity*np.cos(shoot))
		shoot = shoot - e1
		ic = ic+1

	if ic >= 10:
		print('warning ** subroutine kepler did not converge in 10 iterations')

	return shoot

def orbital_to_inertial_matrix(kepel):
	# computes the rotation matrix from orbital frame to inertial frame
	exc = kepel[1]
	c1 = np.sqrt(1.-np.power(exc,2))
	orb2iner = np.matmul(kinematics.rotmaz(-kepel[3]),np.matmul(kinematics.rotmax(-kepel[2]),kinematics.rotmaz(-kepel[4])))
	E = kepler(kepel[5], exc)
	sE = np.sin(E)
	cE = np.cos(E)

	r_ov_a = 1 - exc*cE
	cf = (cE-exc)/r_ov_a
	sf = c1*sE/r_ov_a

	rmx_i_o = np.dot(orb2iner,np.array([[cf,-sf,0],[sf,cf,0],[0,0,1]]))
	return rmx_i_o

def proximus(angleinp, angleprox):
	test = 2*np.pi
	angle = angleprox + np.mod((angleinp-angleprox+test/2),test)-test/2
	return angle

def statvec_kepel(statv):
	# transform the state vetor statv into the corresponding keplerian elements in the same reference system
	### Input ###
	# state vector in meters and meters/second
	#############
	EARTH_GRAVITY = 3.9860064e14  # Earth's gravitational constant [m3/s2]
	xp = statv[0:3]
	xv = statv[3:6]
	r = np.linalg.norm(xp)
	vq = np.power(np.linalg.norm(xv),2)
	ainv = 2.0/r - vq/EARTH_GRAVITY
	h = np.cross(xp, xv)
	# print("Angular momentum")
	# print(h)
	hm = np.linalg.norm(h)
	if hm < 1.e-10:
		print(' *** Messange from function statvec_kepel: ***')
		print(' There are no keplerian elements corresponding to this state vector')
		kepel = np.zeros(6)
	else:
		h = h/hm
		incl = np.arccos(h[2])
		raan = np.arctan2(h[0], -h[1])
		# print("RAAN")
		# print(raan)
		d = np.dot(xp, xv)/EARTH_GRAVITY
		esene = d * np.sqrt(EARTH_GRAVITY*ainv)
		ecose = 1 - r*ainv
		exc = np.sqrt(np.power(esene,2) + np.power(ecose,2))
		E = np.arctan2(esene, ecose)
		mean = np.mod(E-esene, 2*np.pi)
		if mean < 0:
			mean = mean + 2*np.pi
		if exc < 1.e-10:
			arpe = 0
		else:
			dp = 1./r - ainv
			ev = dp*xp - d*xv
			abev = np.linalg.norm(ev)
			ev = ev/abev
			an = np.zeros(3)
			an[0] = np.cos(raan)
			an[1] = np.sin(raan)
			fi = np.dot(ev, np.cross(h, an))
			arpe = np.arccos(np.dot(ev,an))
			if fi < 0:
				arpe = -arpe + 2*np.pi
	kepel = np.array([1./ainv, exc, incl, raan, arpe, mean])
	return kepel

def terrestial_to_inertial(tesig, xt):
	xinert = np.matmul(np.concatenate([[xt[0:3]],[xt[3:6]]],0),np.transpose(kinematics.rotmaz(-tesig)))
	xinert = np.concatenate([xinert[0],xinert[1]],0)
	return xinert

def time_to_dayf(hours, minutes, seconds):
	# return with the day elapesed time in seconds
	dayf = seconds + 60*(minutes + 60*hours)
	return dayf
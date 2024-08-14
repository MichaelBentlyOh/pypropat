import numpy as np
import kinematics
import copy
global egm_order, egm_length, egm_conv_f, egm_cc, egm_sc
global egm_ae, egm_gm, egm_pn, egm_qn, egm_ip, egm_nmax

def dayf_to_time(dayf):
	day1 = np.fix(dayf/3600)
	day2 = np.fix(dayf/60) - 60*day1
	day3 = dayf - 3600*day1 - 60*day2
	day_time = np.array([day1, day2, day3])
	return day_time

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

def djm_inv(mjd):
	d1 = np.array([0, 31, 61, 92, 122, 153, 184, 214, 245, 275, 306, 337, 366])

	y4 = 0;
	y1 = 0;
	d = np.fix(mjd + 127775)
	y400 = np.fix(d/146097)
	d = d - y400*146097
	y100 = np.fix(d/36524)
	d = d - y100*36524

	if y100 > 3:
		dat1 = 29
		dat2 = 2
		dat3 = 1600 + y400*400 + y100*100 + y4*4 + y1
	else:
		y4 = np.fix(d/1461)
		d = d - y4*1461
		y1 = np.fix(d/365)
		if y1 > 3:
			dat1 = 29
			dat2 = 2
			dat3 = 1600 + y400*400 + y100*100 + y4*4 + y1
		else:
			d = d - y1*365
			i = np.fix(d/32 + 2)
			d = d + 1
			while d1[int(i-1)] < d:
				i = i + 1
			dat2 = i + 1
			dat1 = d - d1[int(i-2)]
			dat3 = 1600 + y400*400 + y100*100 + y4*4 + y1
			if dat2 > 12:
				dat2 = dat2 - 12
				dat3 = dat3 + 1
	date = np.array([dat1, dat2, dat3])
	return date

def earth_shadow(sat_pos, sun_pos):
	EARTH_RADIUS = 6378139.
	SUN_RADIUS = 0.6953e9

	dsun = np.linalg.norm(sun_pos[0:2])
	if dsun <= 0:
		shadow = -1
	else:
		vecsun = sun_pos[0:2]/dsun
		rcob = np.dot(sat_pos[0:2], vecsun)
		if rcob < 0:
			radi = SUN_RADIUS/dsun
			auxi = np.cross(sat_pos[0:2],vecsun)
			auxi = np.linalg.norm(auxi)
			psvs = (auxi - EARTH_RADIUS)/rcob/radi
			if np.abs(psvs) < 1:
				shadow = (np.arccos(psvs) - psvs*np.sqrt(1. - psvs*psvs))/np.pi
			else:
				if psvs >= 0:
					shadow = 0
				else:
					shadow = 1
		else:
			shadow = 1
	return shadow

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
	se = inertial_to_terrestrial(gwst, x)
	xe = se[0:3]
	ae = np.concatenate([egm_acc(xe),[0,0,0]],0)
	ai = terrestrial_to_inertial(gwst,np.transpose(ae))
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

def geocentric_to_sph_geodetic(geoc):
	EARTH_FLATNESS = 0.0033528131778969144
	EARTH_RADIUS = 6378139.

	px = geoc[0]
	py = geoc[1]
	pz = geoc[2]
	gama = (1. - EARTH_FLATNESS)
	gama = gama*gama
	eps = 1. - gama
	as_ = EARTH_RADIUS*EARTH_RADIUS
	ws = px*px + py*py
	zs = pz*pz
	zs1 = gama*zs
	e = 1.

	det = 0.01*np.sqrt((2/3)/EARTH_RADIUS)
	de = 2*det

	while (de > det):
		alf = e/(e-eps)
		zs2 = zs1*alf*alf
		de = 0.5*(ws + zs2 - as_*e*e)/((ws + zs2*alf)/e)
		e = e + de

	ss = e - eps
	ss = eps*zs/as_/ss/ss
	ro = EARTH_RADIUS*((1. + ss)/(2. + ss) + 0.25*(2. + ss))
	rw = e*ro

	arl = np.arctan2(py, px)
	sf = pz/(rw - eps*ro)
	cf = np.sqrt(ws)/rw
	anorma = np.sqrt(sf*sf + cf*cf)
	arf = np.arcsin(sf/anorma)
	geodetic = np.array([arl, arf, rw-ro])

	return geodetic

def gst(diju, time):
	tsj = (diju - 18262.5) / 36525
	tsgo = (24110.54841 + (8640184.812866 + 9.3104e-2 * tsj - 6.2e-6 * tsj * tsj) * tsj) * np.pi / 43200
	tetp = 7.292116e-5 	# velocidade angular da Terra(rd / s)
	gwst = np.mod(tsgo + time * tetp, 2 * np.pi)
	return gwst

def igrf_field(date, alt, colat, elong):
	funit = open('igrf11.dat', mode='r')
	data = funit.readline().split()
	n_data = int(data[0])
	other = funit.readlines()
	other_data = np.zeros(0)
	for i in range(0, np.shape(other)[0]):
		other_temp = copy.deepcopy(other[i])
		w = 0
		j = 0
		for num in other[i]:
			if num == '-':
				other_temp = other_temp + ' '
				other_temp = other_temp[0:(j + w)] + ' ' + other_temp[(j + w):-1]
				w = w + 1
			j = j + 1
		other[i] = copy.deepcopy(other_temp)
		other_data = np.append(other_data, np.array([float(num) for num in other[i].split()]))
	n_year = other_data[0:n_data]
	order = other_data[n_data:(2 * n_data)]
	stll = other_data[(2 * n_data):(3 * n_data)]
	gh = other_data[(3 * n_data):-1]

	cl = np.zeros(14)
	sl = np.zeros(14)
	p = np.zeros(106)

	q = np.zeros(106)

	x = np.zeros(3)

	if date < n_year[0] or date > (n_year[n_data - 1] + 5):
		print('igrf_field error')
		print('Date must be in the range:')
		print(n_year[0])
		print(n_year[n_data - 1] + 5)
		field = np.zeros(3,dtype=float)
		return field

	t = 0.2*(date - 1900.0)
	i = int(np.fix(t))
	t = t - i

	if date < n_year[n_data - 2]:
		tc = 1.0 - t
	else:
		t = date - n_year[n_data - 2]
		tc = 1.0

	l1 = stll[i]
	nmx = order[i-1]
	if order[i] < nmx:
		nmx = order[i]

	nc = int(nmx*(nmx + 2))
	kmx = (nmx + 1)*(nmx + 2)/2

	r = alt
	ct = np.cos(colat)
	st = np.sin(colat)

	cl[0] = np.cos(elong)
	sl[0] = np.sin(elong)
	l = 0
	m = 1
	n = 0

	ratio = 6371.2/r
	rr = ratio*ratio

	p[0] = 1.0
	p[2] = st
	q[0] = 0.0
	q[2] = ct

	for k in range(2, int(kmx+1)):
		if n < m:
			m = 0
			n = n + 1
			rr = rr * ratio
			fn = n
			gn = n - 1
		fm = m
		if m != n:
			gmm = m*m
			one = np.sqrt(fn*fn - gmm)
			two = np.sqrt(gn*gn - gmm)/one
			three = (fn + gn)/one
			i = k - n
			j = i - n + 1
			p[k-1] = three*ct*p[i-1] - two*p[j-1]
			q[k-1] = three*(ct*q[i-1] - st*p[i-1]) - two*q[j-1]
		else:
			if k != 3:
				one = np.sqrt(1 - 0.5/fm)
				j = k - n - 1
				p[k-1] = one*st*p[j-1]
				q[k-1] = one*(st*q[j-1] + ct*p[j-1])
				cl[m-1] = cl[m-2]*cl[0] - sl[m-2]*sl[0]
				sl[m-1] = sl[m-2]*cl[0] + cl[m-2]*sl[0]

		lm = int(l1 + l + 1)
		one = (tc*gh[lm-1] + t*gh[lm+nc-1])*rr

		if m != 0:
			two = (tc*gh[lm] + t*gh[lm+nc])*rr
			three = one*cl[m-1] + two*sl[m-1]
			if st != 0.0:
				y = (one*sl[m-1] - two*cl[m-1])*fm*p[k-1]/st
			else:
				y = (one*sl[m-1] - two*cl[m-1])*q[k-1]*ct
			x = x + np.array([three*q[k-1], y, -(fn + 1.0)*three*p[k-1]])
			l = l + 2
		else:
			x = x + np.array([one*q[k-1], 0, -(fn + 1.0)*one*p[k-1]])
			l = l + 1
		m = m + 1
	field = x
	return field

def inertial_to_terrestrial(tesig, xi):
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

def sph_geodetic_to_geocentric(spgd):
	EARTH_FLATNESS = 0.0033528131778969144
	EARTH_RADIUS	= 6378139.

	al = spgd[0]
	h = spgd[2]

	sf = np.sin(spgd[1])
	cf = np.cos(spgd[1])
	gama = (1. - EARTH_FLATNESS)
	gama = gama*gama
	s = EARTH_RADIUS / np.sqrt(1. - (1. - gama)*sf*sf)
	g1 = (s + h)*cf
	geoc = np.array([g1*np.cos(al), g1*np.sin(al), (s*gama + h)*sf])

	return geoc
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

def sun(djm, ts):
	# The subroutine sun calculates the position vector of the Sun in ECI system refered to J2000
	rad = np.pi/180
	ASTRONOMICAL_UNIT = 149.60e9

	t = djm - 18262.5 + ts/86400.

	alom_ab = np.mod((280.460 + 0.9856474*t)*rad, 2*np.pi)

	if alom_ab < 0:
		alom_ab = alom_ab + 2*np.pi

	an_mean = np.mod((357.528 + 0.9856003*t)*rad, 2*np.pi)
	if an_mean < 0:
		an_mean = an_mean + 2*np.pi

	an_mean_2 = an_mean + an_mean

	if an_mean_2 > (2*np.pi):
		an_mean_2 = np.mod(an_mean_2, 2*np.pi)

	ecli_lo = alom_ab + (1.915*np.sin(an_mean) + 0.02*np.sin(an_mean_2))*rad
	sin_ecli_lo = np.sin(ecli_lo)
	cos_ecli_lo = np.cos(ecli_lo)

	obl_ecli = (23.439 - 4e-7*t)*rad
	sin_obl_ecli = np.sin(obl_ecli)
	cos_obl_ecli = np.cos(obl_ecli)

	sunpos = np.zeros(6)
	sunpos[3] = np.arctan2(cos_obl_ecli*sin_ecli_lo, cos_ecli_lo)
	if sunpos[3] < 0:
		sunpos[3] = sunpos[3] + 2*np.pi

	sunpos[4] = np.arcsin(sin_obl_ecli*sin_ecli_lo)
	sunpos[5] = (1.00014 - 0.01671*np.cos(an_mean) - 1.4e-4*np.cos(an_mean_2))*ASTRONOMICAL_UNIT

	sunpos[0] = sunpos[5]*cos_ecli_lo
	sunpos[1] = sunpos[5]*cos_obl_ecli*sin_ecli_lo
	sunpos[2] = sunpos[5]*sin_obl_ecli*sin_ecli_lo

	return sunpos

def sun_dir(djm, ts):
	idays = djm - 18261
	tttt = idays + ts/86400

	w = 4.9382416 + 8.21936631e-7*tttt
	m = 6.2141924 + 0.01720197*tttt
	m = np.mod(m, 2*np.pi)
	ecc = 0.016709 - 1.151e-9*tttt

	u = m + 2.*ecc*np.sin(m) + w + 1.25*ecc*ecc*np.cos(m)
	ret = u
	su = np.sin(u)

	eps = 0.409093 - 6.2186081e-9*tttt

	sunpos = np.array([[np.cos(u)],[su*np.cos(eps)],[su*np.sin(eps)]])
	return sunpos, ret

def sunsync_inc(sma, exc):
	earth_gravity = 3.9860064e14
	tropic_year = 365.24219879
	earth_radius = 6378139.
	arg = 1.72
	j_2 = 1.0826268362e-3
	el = np.array([sma, exc, arg, 0, 0, 0])

	omegap = 2*np.pi/tropic_year/86400
	amm = np.sqrt(earth_gravity/(sma*sma*sma))
	con = -1.5*j_2*amm*earth_radius*earth_radius/(sma*sma)
	delta = 1
	ic = 0

	while (np.abs(delta) > 1e-6) & (ic < 20):
		delk = delkep(el)
		chu = np.cos(arg)
		delta = (omegap - delk[3])/con
		chu = chu + delta
		arg = np.arccos(chu)
		el[2] = arg
		ic = ic + 1
	sunsync_inclination = arg

	if ic > 20:
		print('Error in function sunsync_inclination:. Interaction did not converge')

	return sunsync_inclination

def sunsync_raan(eq_cross_time, gst0):
	raan = eq_cross_time + gst0
	return raan

def sunsync_sma(exc, inc, q):
	earth_gravity = 3.9860064e14
	earth_rate = 7.2921158546819492e-5

	el = np.array([6878000, exc, inc, 0, 0, 0])
	ant = el[0]

	epx = -1.5*np.sqrt(earth_gravity/(ant**5))
	delta = 1000000
	ic = 0

	while (np.abs(delta/ant) > 1e-9) & (ic < 20):
		delk = delkep(el)
		fact = delk[4] + delk[5] - q*(earth_rate - delk[3])
		delta = fact/epx
		sma = ant -	delta
		ant = sma
		el[0] = sma
		ic = ic + 1;

	smaxis = sma

	if ic > 30:
		print(' Error in routine sunsync_recf. Interaction did not converge')

	return smaxis

def terrestrial_to_inertial(tesig, xt):
	xinert = np.matmul(np.concatenate([[xt[0:3]],[xt[3:6]]],0),np.transpose(kinematics.rotmaz(-tesig)))
	xinert = np.concatenate([xinert[0],xinert[1]],0)
	return xinert

def time_to_dayf(hours, minutes, seconds):
	# return with the day elapesed time in seconds
	dayf = seconds + 60*(minutes + 60*hours)
	return dayf

def visviva(a,r):
	mu = 3.986e14
	vi = np.sqrt(mu*(2/r - 1/a))
	return vi
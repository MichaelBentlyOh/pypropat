import numpy as np

'''
Rotation method follows DCM for frame change.
Original propat follows DCM; a transpose of Rotation Matrix.
Normally, and without any comments, vectors are almost row vectors(1*3).
'''


def cross_matrix(w):
    """
    skew matrix for cross product
    input
    w : 1*3 element radian array
    output
    cross_mat : 3*3 numpy array matrix
    """
    cross_mat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
    return cross_mat


def rotmax(angle):
    """
    rotation matrix for x-axis rotation DCM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[1, 0, 0],
                        [0, cosine, sine],
                        [0, -sine, cosine]])
    return rot_mat


def rotmay(angle):
    """
    rotation matrix for y-axis rotation DCM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[cosine, 0, -sine],
                        [0, 1, 0],
                        [sine, 0, cosine]])
    return rot_mat


def rotmaz(angle):
    """
    rotation matrix for y-axis rotation DCM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[cosine, sine, 0],
                        [-sine, cosine, 0],
                        [0, 0, 1]])
    return rot_mat


def rotmax_rx(angle):
    """
    rotation matrix for x-axis rotation RM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[1, 0, 0],
                        [0, cosine, -sine],
                        [0, sine, cosine]])
    return rot_mat


def rotmay_rx(angle):
    """
    rotation matrix for y-axis rotation RM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[cosine, 0, sine],
                        [0, 1, 0],
                        [-sine, 0, cosine]])
    return rot_mat


def rotmaz_rx(angle):
    """
    rotation matrix for y-axis rotation RM
    input
    angle : radian angle
    output
    rot_mat : 3*3 numpy array matrix
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rot_mat = np.array([[cosine, -sine, 0],
                        [sine, cosine, 0],
                        [0, 0, 1]])
    return rot_mat


# =======================================================================
def eulerrmx(euler_angle, euler_vector):
    """
    rodrigues rotation
    input
    euler_angle  : radian angle
    euler_vector : 1*3 element radian array
    output
    rot_mat      : 3*3 numpy array matrix
    """
    cosine_0 = np.cos(euler_angle)
    sine = np.sin(euler_angle)
    cosine_1 = 1 - cosine_0
    euler_vector = np.reshape(euler_vector, (3, 1))

    rot_mat = cosine_0 * np.eye(3) + cosine_1 * np.dot(euler_vector, euler_vector.T) + \
              sine * cross_matrix(euler_vector.flatten())

    return rot_mat


def rmxeuler(rot_mat):
    """
    abstract Rodrigues rotation elements
    input
    rot_mat : 3*3 radian angle matrix
    output
    euler_angle  : scalar radian angle
    euler_vector : 1*3 numpy array
    """
    trace = np.trace(rot_mat)
    if trace == 3:
        euler_angle = 0
        euler_vector = np.array([1, 0, 0])
    elif trace < -0.99999:
        euler_angle = np.pi
        w = np.diagonal(rot_mat)
        euler_vector = np.sqrt((1 + w) / 2)
        if euler_vector[0] > 0.5:
            euler_vector[1] = np.sign(rot_mat[0, 1]) * euler_vector[1]
            euler_vector[2] = np.sign(rot_mat[2, 0]) * euler_vector[2]
        elif euler_vector[1] > 0.5:
            euler_vector[0] = np.sign(rot_mat[0, 1]) * euler_vector[0]
            euler_vector[2] = np.sign(rot_mat[1, 2]) * euler_vector[2]
        else:
            euler_vector[0] = np.sign(rot_mat[2, 0]) * euler_vector[0]
            euler_vector[1] = np.sign(rot_mat[1, 2]) * euler_vector[1]
    else:
        euler_angle = np.arccos((trace - 1) / 2)
        sine = np.sin(euler_angle)
        euler_vector = np.array([
            rot_mat[1, 2] - rot_mat[2, 1],
            rot_mat[2, 0] - rot_mat[0, 2],
            rot_mat[0, 1] - rot_mat[1, 0]
        ]) / (2 * sine)

    return euler_angle, euler_vector


# =======================================================================

def rmxexyz(rot_mat):
    """
    abstract X-Y-Z rotation angles from rotation matrix
    input
    rot_mat : 3*3 numpy array
    output
    euler_angles : 1*3 numpy array
    """
    a11, a12, a21, a22, a31, a32, a33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[1, 0], rot_mat[1, 1], rot_mat[2, 0], \
    rot_mat[2, 1], rot_mat[2, 2]

    if abs(a31) <= 1:
        eul2 = np.arcsin(a31)
    elif a31 < 0:
        eul2 = -np.pi / 2
    else:
        eul2 = np.pi / 2

    if abs(a31) <= 0.99999:
        if a33 != 0:
            eul1 = np.arctan2(-a32, a33)
            if eul1 > np.pi:
                eul1 = eul1 - 2 * np.pi
        else:
            eul1 = np.pi / 2 * np.sign(-a32)

        if a11 != 0:
            eul3 = np.arctan2(-a21, a11)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(-a21)
    else:
        eul1 = 0
        if a22 != 0:
            eul3 = np.arctan2(a12, a22)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(a12)

    euler_angles = np.array([eul1, eul2, eul3])

    return euler_angles


def rmxezxy(rot_mat):
    """
    abstract Z-X-Y rotation angles from rotation matrix
    input
    rot_mat : 3*3 numpy array
    output
    euler_angles : 1*3 numpy array
    """
    spct = -rot_mat[0, 2]
    ctsf = -rot_mat[1, 0]
    ctcf = rot_mat[1, 1]
    stet = rot_mat[1, 2]
    cpct = rot_mat[2, 2]

    if abs(stet) <= 1:
        eul2 = np.arcsin(stet)
    else:
        eul2 = np.pi / 2 * np.sign(stet)

    if abs(eul2) <= np.pi / 2 - 1e-5:
        if abs(ctcf) != 0:
            eul1 = np.arctan2(ctsf, ctcf)
            if eul1 > np.pi:
                eul1 = eul1 - 2 * np.pi
        else:
            eul1 = np.pi / 2 * np.sign(ctsf)

        if abs(cpct) != 0:
            eul3 = np.arctan2(spct, cpct)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(spct)
    else:
        capb = rot_mat[0, 0]
        sapb = rot_mat[0, 1]
        eul1 = 0.
        if abs(capb) != 0:
            eul3 = np.arctan2(sapb, capb)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = 0.

    euler_angles = np.array([eul1, eul2, eul3])

    return euler_angles


def rmxezxz(rot_mat):
    """
    abstract Z-X-Z rotation angles from rotation matrix
    input
    rot_mat : 3*3 numpy array
    output
    euler_angles : 1*3 numpy array
    """
    a11, a12, a13, a23, a31, a32, a33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 0], \
    rot_mat[2, 1], rot_mat[2, 2]

    if abs(a33) <= 1:
        eul2 = np.arccos(a33)
    elif a33 < 0:
        eul2 = np.pi
    else:
        eul2 = 0

    if abs(eul2) >= 0.00001:
        if a32 != 0:
            eul1 = np.arctan2(a31, -a32)
        else:
            eul1 = np.pi / 2 * np.sign(a31)
            if eul1 > np.pi:
                eul1 = eul1 - 2 * np.pi

        if a23 != 0:
            eul3 = np.arctan2(a13, a23)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(a13)
    else:
        eul1 = 0
        if a11 != 0:
            eul3 = np.arctan2(a12, a11)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(a12)

    euler_angles = np.array([eul1, eul2, eul3])

    return euler_angles


def rmxezyx(rot_mat):
    """
    abstract Z-Y-X rotation angles from rotation matrix
    input
    rot_mat : 3*3 numpy array
    output
    euler_angles : 1*3 numpy array
    """
    stet = -rot_mat[0, 2]
    ctsf = rot_mat[0, 1]
    ctcf = rot_mat[0, 0]
    spct = rot_mat[1, 2]
    cpct = rot_mat[2, 2]

    if abs(stet) <= 1.:
        eul2 = np.arcsin(stet)
    else:
        eul2 = np.pi / 2 * np.sign(stet)

    if abs(eul2) <= np.pi / 2 - 1e-5:
        if abs(ctcf) != 0:
            eul1 = np.arctan2(ctsf, ctcf)
            if eul1 > np.pi:
                eul1 = eul1 - 2 * np.pi
        else:
            eul1 = np.pi / 2 * np.sign(ctsf)

        if abs(cpct) != 0:
            eul3 = np.arctan2(spct, cpct)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = np.pi / 2 * np.sign(spct)
    else:
        capb = rot_mat[1, 1]
        sapb = rot_mat[1, 0]
        eul1 = 0.

        if abs(capb) != 0:
            eul3 = np.arctan2(sapb, capb)
            if eul3 > np.pi:
                eul3 = eul3 - 2 * np.pi
        else:
            eul3 = 0.

    euler_angles = np.array([eul1, eul2, eul3])

    return euler_angles


def rmxquat(rot_mat):
    """
    Obtain the attitude quaternions from the attitude rotation matrix.

    Parameters:
    rot_mat : np.array
        Rotation matrix (3x3).

    Returns:
    quaternions : np.array
        Attitude quaternions.
    """
    matra = np.trace(rot_mat)
    auxi = 1 - matra
    selec = np.array([1 + matra, auxi + 2 * rot_mat[0, 0], auxi + 2 * rot_mat[1, 1], auxi + 2 * rot_mat[2, 2]])
    ites = np.argmax(selec)
    auxi = 0.5 * np.sqrt(selec[ites])

    if ites == 0:
        quaternions = np.array([
            (rot_mat[1, 2] - rot_mat[2, 1]) / (4 * auxi),
            (rot_mat[2, 0] - rot_mat[0, 2]) / (4 * auxi),
            (rot_mat[0, 1] - rot_mat[1, 0]) / (4 * auxi),
            auxi
        ])
    elif ites == 1:
        quaternions = np.array([
            auxi,
            (rot_mat[0, 1] + rot_mat[1, 0]) / (4 * auxi),
            (rot_mat[0, 2] + rot_mat[2, 0]) / (4 * auxi),
            (rot_mat[1, 2] - rot_mat[2, 1]) / (4 * auxi)
        ])
    elif ites == 2:
        quaternions = np.array([
            (rot_mat[0, 1] + rot_mat[1, 0]) / (4 * auxi),
            auxi,
            (rot_mat[1, 2] + rot_mat[2, 1]) / (4 * auxi),
            (rot_mat[2, 0] - rot_mat[0, 2]) / (4 * auxi)
        ])
    else:  # ites == 3
        quaternions = np.array([
            (rot_mat[0, 2] + rot_mat[2, 0]) / (4 * auxi),
            (rot_mat[1, 2] + rot_mat[2, 1]) / (4 * auxi),
            auxi,
            (rot_mat[0, 1] - rot_mat[1, 0]) / (4 * auxi)
        ])

    return quaternions


def quat_matrix(q):
    """
    q : quaternion [q_vec,q_meg]^T
    Return : 4*4 quaternion matrix
    """
    q_mat = [[[q[3], -q[2], q[1], q[0]],
              [q[2], q[3], -q[0], q[1]],
              [-q[1], q[0], q[3], q[2]],
              [-q[0], -q[1], -q[2], q[3]]]]
    return q_mat


def quat_inv(q):
    """
    q : quaternion
    Return : q_conj = [-q_vec,q_meg]^T
    """
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    return q_conj


def quat_norm(q):
    """
    Normalize the quaternion to have a unit norm.

    q : np.array
        Input quaternion [qx, qy, qz, qw]

    Returns:
    np.array
        Normalized quaternion [qx, qy, qz, qw]
    """
    v = q[:3]
    e = q[3]
    e = min(max(e, -1), 1)
    vnorm = np.dot(v, v)
    enorm = e ** 2

    if vnorm != 0:
        enorm = np.sqrt((1 - enorm) / vnorm)
        q_norm = np.concatenate((enorm * v, [e]))
    else:
        q_norm = np.array([0, 0, 0, 1])

    return q_norm


def quat_prod(quat1, quat2):
    """
    Compute the product of two quaternions.

    quat1 : np.array
        First quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4.
    quat2 : np.array
        Second quaternion [p1, p2, p3, p4] where P = p1 i + p2 j + p3 k + p4.

    Returns:
    np.array
        Quaternion product [r1, r2, r3, r4] where R = Q X P.
    """
    v1 = quat1[:3]
    v2 = quat2[:3]
    q4 = quat1[3]
    p4 = quat2[3]

    # Quaternion product formula
    real_part = q4 * p4 - np.dot(v1, v2)
    imaginary_part = q4 * v2 + p4 * v1 + np.cross(v1, v2)
    quat = np.concatenate((imaginary_part, [real_part]))

    return quat


def quat_unity(q):
    """
    q : np.array
        Input quaternion [qx, qy, qz, qw]
    Returns:
        np.array
        Normalized quaternion [qx, qy, qz, qw], such that the square of its norm equals 1.
    """
    qnorm = np.sqrt(np.dot(q, q))
    if (qnorm != 0):
        q_unit = q / qnorm
    else:
        q_unit = np.array([0, 0, 0, 1])

    return q_unit


def quatrmx(quaternion):
    """
    Compute the rotation matrix from the quaternion.

    Parameters:
    quaternion : np.array
        Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4

    Returns:
    rot_mat : np.array
        Rotation matrix (3, 3) as euler rotation matrix (DCM)
    """
    q1q, q2q, q3q, q4q = quaternion[0] ** 2, quaternion[1] ** 2, quaternion[2] ** 2, quaternion[3] ** 2
    q12, q13, q14 = 2 * quaternion[0] * quaternion[1], 2 * quaternion[0] * quaternion[2], 2 * quaternion[0] * \
                    quaternion[3]
    q23, q24, q34 = 2 * quaternion[1] * quaternion[2], 2 * quaternion[1] * quaternion[3], 2 * quaternion[2] * \
                    quaternion[3]

    rot_mat = np.array([
        [q1q - q2q - q3q + q4q, q12 + q34, q13 - q24],
        [q12 - q34, q2q - q1q + q4q - q3q, q23 + q14],
        [q13 + q24, q23 - q14, q3q - q1q - q2q + q4q]
    ])
    return rot_mat


def quatexyz(q):
    """
    Parameters:
    quaternion : np.array
        Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
    Returns:
    rot_mat : np.array
        Rotation matrix (3, 3) as a xyz sequence rotation
    """
    if q.ndim == 1:
        q = q.reshape(4, 1)

    _, n = np.size(q)
    euler_angle = []

    for i in range(n):
        rot_mat = quatrmx(q[:, i])
        angle = rmxexyz(rot_mat)
        euler_angle.append(angle)
    euler_angle = np.array(euler_angle).T

    return euler_angle


def quatezxz(q):
    """
    Parameters:
    quaternion : np.array
        Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
    Returns:
    rot_mat : np.array
        Rotation matrix (3, 3) as a zyz sequence rotation
    """
    if q.ndim == 1:
        q = q.reshape(4,1)

    _, n = np.shape(q)
    euler_angle = []

    for i in range(n):
        rot_mat = quatrmx(q[:, i])
        angle = rmxezxz(rot_mat)
        euler_angle.append(angle)
    euler_angle = np.array(euler_angle).T

    return euler_angle


def exyzrmx(euler_angles):
    rot_mat = rotmaz(euler_angles[2]) @ rotmay(euler_angles[1]) @ rotmax(euler_angles[0])
    return rot_mat


def ezxyrmx(euler_angles):
    rot_mat = rotmay(euler_angles[2]) @ rotmax(euler_angles[1]) @ rotmaz(euler_angles[0])
    return rot_mat


def ezxzrmx(euler_angles):
    rot_mat = rotmaz(euler_angles[2]) @ rotmax(euler_angles[1]) @ rotmaz(euler_angles[0])
    return rot_mat


def ezyxrmx(euler_angles):
    rot_mat = rotmax(euler_angles[2]) @ rotmay(euler_angles[1]) @ rotmaz(euler_angles[0])
    return rot_mat


def ezxzquat(euler_angles):
    rot_mat = ezxzrmx(euler_angles)
    quat = rmxquat(rot_mat)
    return quat


def exyzquat(euler_angles):
    rot_mat = exyzrmx(euler_angles)
    quat = rmxquat(rot_mat)
    return quat


def sangvel(w):
    skew_ang_vel = np.array([
        [0, w[2], -w[1], w[0]],
        [-w[2], 0, w[0], w[1]],
        [w[1], -w[0], 0, w[2]],
        [-w[0], -w[1], -w[2], 0]
    ])
    return skew_ang_vel


def proximus(angleinp, angleprox):
    test = 2 * np.pi
    angle = angleprox + np.mod((angleinp - angleprox + test / 2), test) - test / 2
    return angle

def rectangular_to_spherical(geoc):
    px = geoc[0]
    py = geoc[1]
    pz = geoc[2]
    ws = px * px + py * py
    rw = np.sqrt(ws + pz * pz)
    lg = np.arctan2(py, px)
    lt = np.arctan2(pz, np.sqrt(ws))
    spherical = np.array([lg, lt, rw])
    return spherical

import numpy as np
import matplotlib.pyplot as plt


# Plot 2D points
def displaypoints2d(points):
    plt.figure(0)
    plt.plot(points[0, :], points[1, :], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

  Args:
    points: a np array of points in cartesian coordinates

  Returns:
    points_hom: a np array of points in homogeneous coordinates
  """

    #
    # You code here
    #
    points_homo = np.ones((points.shape[0] + 1, points.shape[1]))
    points_homo[:-1, :] = points
    return points_homo


# print(cart2hom(np.arange(12).reshape((4, 3))))

def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

  Args:
    points: a np array of points in homogenous coordinates

  Returns:
    points_hom: a np array of points in cartesian coordinates
  """

    #
    # You code here
    #
    points_cart = points[:-1, :]
    return points_cart


# print(hom2cart(cart2hom(np.arange(12).reshape((4, 3)))))

def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates for translation by v.

  Args:
    v: 3d translation vector

  Returns:
    T: translation matrix in homogeneous coordinates
  """

    #
    # You code here
    #
    mx_T = np.diag(np.ones(4))
    mx_T[:-1, -1] = v
    return mx_T


# print(gettranslation(np.arange(3)))

def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rx: rotation matrix
  """

    #
    # You code here
    #
    dg = np.radians(d)
    c, s = np.cos(dg), np.sin(dg)
    rx = np.diag(np.ones(4))
    rx[1:-1, 1:-1] = np.array(((c, -s), (s, c)))
    return rx


# print(getxrotation(30))

def getyrotation(d):
    """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

  Args:
    d: degrees of the rotation

  Returns:
    Ry: rotation matrix
  """

    #
    # You code here
    #
    dg = np.radians(d)
    c, s = np.cos(dg), np.sin(dg)
    ry = np.diag(np.ones(4))
    ry[:-1, :-1] = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    return ry


# print(getyrotation(30))

def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rz: rotation matrix
  """

    #
    # You code here
    #
    dg = np.radians(d)
    c, s = np.cos(dg), np.sin(dg)
    rz = np.diag(np.ones(4))
    rz[:-2, :-2] = np.array(((c, -s), (s, c)))
    return rz


# print(getzrotation(30))


def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
  image coordinates depending on the principal point and focal length.
  
  Args:
    principal: the principal point, 2d vector
    focal: focal length

  Returns:
    L: central projection matrix
  """

    #
    # You code here
    #
    mx_k = np.zeros((3, 4))
    np.fill_diagonal(mx_k, [focal, focal, 1])
    mx_k[:-1, -1] = principal
    return mx_k


# print(getcentralprojection([1, 2], -1))

def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic transformation matrix M.

  Args:
    T: translation matrix
    Rx: rotation matrix for rotation around the x-axis
    Ry: rotation matrix for rotation around the y-axis
    Rz: rotation matrix for rotation around the z-axis
    L: central projection matrix

  Returns:
    P: projection matrix
    M: matrix that summarizes extrinsic transformations
  """

    #
    # You code here
    #

    #Matrix-Mul-Assoziativítät, hier: (Rz x (Rx x (Ry x T)))
    m = np.matmul(Ry, T)
    m = np.matmul(Rx, m)
    m = np.matmul(Rz, m)

    p = np.matmul(L, m)

    return p, m


'''rx, ry, rz = getxrotation(0), getyrotation(0), getzrotation(90)
mx_L = getcentralprojection(np.array([0, 0]), 1)
mx_T = gettranslation(np.array([0, 0, 1]))
mx_P = getfullprojection(mx_T, rx, ry, rz, mx_L)
print("mx_T: \n", mx_T)
print("mx_L: \n", mx_L)
print("before: \n", np.ones((4, 1)))
print("mx_P: \n", mx_P)
print("after: \n", np.matmul(mx_P, np.ones(4)))'''


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

  Args:
    P: projection matrix
    X: 3d points in cartesian coordinates

  Returns:
    x: 2d points in cartesian coordinates
  """

    #
    # You code here
    #
    homo_3d = np.vstack((X, np.ones((1, X.shape[1]))))
    homo_2d = np.matmul(P, homo_3d)
    return homo_2d[:, :-1]


def loadpoints():
    """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

    #
    # You code here
    #
    npy = np.load('data/obj2d.npy')
    return npy


# show plot with 2d points
# 1npy = loadpoints()
# plt.plot(npy)
# plt.show()

def loadz():
    """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

    #
    # You code here
    #
    npy = np.load('data/zs.npy')  # .transpose()
    return npy


# loadz()

def invertprojection(L, P2d, z):
    """
  Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

  Args:
    L: central projection matrix
    P2d: 2d image coordinates of the projected points
    z: z-components of the homogeneous image coordinates

  Returns:
    P3d: 3d cartesian camera coordinates of the points
  """

    #
    # You code here
    #
    m = np.matrix(L).I
    xyz = np.vstack((P2d, z))
    new_xy = np.matmul(m, xyz)

    return new_xy[:-1, :]


#P3d = invertprojection(getcentralprojection([0, 0], 1), loadpoints(), loadz())


def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous coordinates
  for the 3D points P3d in cartesian coordinates.

  Args:
    M: matrix summarizing the extrinsic transformations
    P3d: 3d points in cartesian coordinates

  Returns:
    X: 3d points after the extrinsic transformations have been reverted
  """

    #
    # You code here
    #
    reversed_M = np.matrix(M).I
    homo_3d = np.vstack((P3d, np.ones((1, P3d.shape[1]))))
    reversed_3d = np.matmul(reversed_M, homo_3d)
    return reversed_3d


#print(inverttransformation(gettranslation([0, 0, 1]), P3d)[0])


def p3multiplecoice():
    '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''
    '''change transformation-order'''
    # Translation Matrix
    mx_T = gettranslation(np.array([-27.1, -2.9, -3.2]))
    
    # order before: translation -> rotationY -> rotationX -> rotationZ
    # order after:  rotationY -> translation -> rotationX -> rotationZ
    rx, ry, rz = getxrotation(-30), getyrotation(135), getzrotation(90)

    # Calibration Matrix
    mx_L = getcentralprojection(np.array([8, -10]), 8)

    # Projection Matrix use original order
    _, pre_mx_M = getfullprojection(mx_T, rx, ry, rz, mx_L)

    # Projection Matrix use changed order
    _, suf_mx_M = getfullprojection(rx, mx_T, ry, rz, mx_L)

    # load 3d Points
    points_3d = np.arange(30).reshape((10, 3))
    homo_points_3d = np.hstack((points_3d, np.ones((10, 1))))

    # Projected Points using original order
    pre_2d = np.matmul(homo_points_3d, pre_mx_M.transpose())[:, :-1]

    # Projected Points using changed order
    suf_2d_t = np.matmul(homo_points_3d, suf_mx_M.transpose())[:, :-1]

    '''change rotations-order'''
    #order before: translation -> rotationY -> rotationX -> rotationZ
    #order after:  translation -> rotationX -> rotationY -> rotationZ

    # Projection Matrix use changed order
    _, suf_mx_M = getfullprojection(mx_T, ry, rx, rz, mx_L)


    # Projected Points using changed order
    suf_2d_r = np.matmul(homo_points_3d, suf_mx_M.transpose())[:, :-1]

    '''change rotations-order and transformation-order'''
    # order before: translation -> rotationY -> rotationX -> rotationZ
    # order after:  rotationX -> rotationY -> rotationZ -> translation

    # Projection Matrix use changed order
    _, suf_mx_M = getfullprojection(rx, rz, ry, mx_T, mx_L)

    # Projected Points using changed order
    suf_2d_rt = np.matmul(homo_points_3d, suf_mx_M.transpose())[:, :-1]

    ret = -1
    if (pre_2d == suf_2d_rt).all():
      ret = 2
    elif (pre_2d == suf_2d_r).all() and (not (pre_2d == suf_2d_t).all()):
      ret = 1
    elif not (pre_2d == suf_2d_t).all():
      ret = 0
    return ret


#print(p3multiplecoice())
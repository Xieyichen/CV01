import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    #
    # You code here
    #
    bd_np = np.load(path)
    return bd_np

def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    #
    # You code here
    #
    g_channel = np.copy(bayerdata)
    g_channel[::2, 1::2] = 0
    g_channel[1::2, ::2] = 0

    r_channel = np.copy(bayerdata)
    r_channel[::, ::2] = 0
    r_channel[1::2, 1::2] = 0

    b_channel = np.copy(bayerdata)
    b_channel[::, 1::2] = 0
    b_channel[::2, ::2] = 0

    return r_channel, g_channel, b_channel

def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    c = np.dstack((r, g, b))
    return c



def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    #k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    kr = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    kg = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    kb = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]])
    r = convolve(r, kr, mode='constant', cval=0.0)
    g = convolve(g, kg, mode='constant', cval=0.0)
    b = convolve(b, kb, mode='constant', cval=0.0)
    c = np.dstack((r, g, b))
    return c



img = loaddata('data/bayerdata.npy')

r_channel, g_channel, b_channel = separatechannels(img)
c_img_np = assembleimage(r_channel, g_channel, b_channel)
#imgplot = plt.imshow(c_img_np)
#plt.show()

fin_img_np = interpolate(r_channel, g_channel, b_channel)
#imgplot = plt.imshow(fin_img_np)
#plt.show()

#tune weights
plt.rcParams["figure.figsize"] = [20.00, 10.00]
plt.rcParams["figure.autolayout"] = True
plt.subplot(1, 2, 1)
plt.imshow(c_img_np)
plt.subplot(1, 2, 2)
plt.imshow(fin_img_np)
plt.show()
 
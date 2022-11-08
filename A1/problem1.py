import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from main import *

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    imgplot = plt.imshow(img)
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    np.save(path, img)



def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    img = np.load(path)
    return img




def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #
    # You code here
    #
    mirrored_img = np.flip(img, 1)
    return mirrored_img

def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    #
    plt.rcParams["figure.figsize"] = [10.00, 5.00]
    plt.rcParams["figure.autolayout"] = True

    plt.subplot(1, 2, 1)
    plt.imshow(img1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()

'''#check reloaded img
img1 = load_image('data/a1p1.png')
save_as_npy('a1p1_np.npy', img1)
new_img1 = load_npy('a1p1_np.npy')
assert img1.shape == new_img1.shape
print('reloaded_imgs same?: ', (img1 == new_img1).all())

#mirrored img and display
img2 = mirror_horizontal(img1)
display_images(img1, img2)'''
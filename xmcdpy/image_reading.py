import numpy as np
import os
from skimage import io
from skimage.io import imsave


def read_image(file_path):
    """Reads the image from file_path.

    Args:
        file_path (str)

    Returns:
        array: Image in np array.
    """
    ext = os.path.splitext(file_path)[-1]
    if ext == '.dat':
        img = read_dat_image(file_path)
    else:
        img = io.imread(file_path, as_gray=True)
    return img


def read_dat_image(file_path):
    """Reads an image in Alba .dat format

    Args:
        file_path (str)

    Returns:
        array: Image in np array.
    """
    width = 1024
    height = 1024
    with open(file_path, 'rb') as f:
        read = f.read()
        datfile = np.frombuffer(read[-width * height * 2:], dtype=np.ushort)
#     width = int(datfile[20])
#     height = int(datfile[20])
    img = np.reshape(datfile.astype(int), (width, height))
    img = np.flipud(img)
    return img


def import_stack(folder, name_filter=''):
    """Imports a stack of images in the given folder

    Args:
        folder (str): Path to folder
        name_filter (str, optional): Substring that images need to contain. Defaults to ''.

    Returns:
        (n, m, k) array: Stack of images with the last axis being the different images.
    """
    files = [os.path.join(folder, f) for f in os.listdir(
        folder) if os.path.splitext(f)[1] in {'.dat', '.tiff', '.png', '.tif'} and name_filter in os.path.basename(f)]
    images = [read_image(f) for f in files]
    return np.moveaxis(np.array(images), 0, 2)


def import_single(folder, name_filter=''):
    """Imports the first image in the folder containing the name_filter substring.

    Args:
        folder (str): Path to folder
        name_filter (str, optional): Defaults to ''.

    Returns:
        (n, m) array:  Image in np array.
    """
    files = [os.path.join(folder, f) for f in os.listdir(
        folder) if os.path.splitext(f)[1] in {'.dat', '.tiff', '.png', '.tif'} and name_filter in os.path.basename(f)]
    images = [read_image(files[0]), ]
    return np.moveaxis(np.array(images), 0, 2)


def save_image(path, image):
    """Saves the image to path. First puts the values between 0 and 255 and transforms to uint8.

    Args:
        path (str)
        image (array)
    """
    # for saving the images
    mn, mx = np.min(image), np.max(image)
    image_tosave = 255 * ((image - mn) / (mx - mn)).astype(float)
    imsave(path, image_tosave.astype(np.uint8))

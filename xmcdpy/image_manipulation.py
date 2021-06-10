import warnings
import numpy as np
from PIL import Image


def select_rect_coord(img, rect_coord):
    """Selects the image cropped to rect_coord. 

    Args:
        img ((n, m), (n, m, k) array): Image numpy array.
        rect_coord ((4,) array, list): Rectangular coordinates defining corners of the rectangle.

    Returns:
        array: cropped image
    """
    if len(img.shape) == 3:
        return img[rect_coord[1]:rect_coord[3],
                   rect_coord[0]:rect_coord[2], :]
    else:
        return img[rect_coord[1]:rect_coord[3],
                   rect_coord[0]:rect_coord[2]]


def normalise_mnmx(img):
    """Sets the image values between 0 and 1 by substracting the minimum and dividing by the difference"""
    mn, mx = np.min(img), np.max(img)
    return (img.astype(float) - mn) / (mx - mn)


def normalise_image(img, norm_image):
    """Divides the image with normalisation image"""
    img_out = img / norm_image
    return img_out


def normalise_stack(stack, norm_image):
    """Runs normalise_image on every image in the stack and returns a stack of normalised images"""
    norm_stack = [normalise_image(stack[:, :, i], norm_image)
                  for i in range(stack.shape[2])]
    return np.moveaxis(np.array(norm_stack), 0, 2)


def normalise_stack_brightness(images, background_mask, threshold=0.15):
    """For every structure in the stack, gets the mean of the masked region and divides each image by the factor of difference from the first one.
    If the adjustment factor for brightness is larger than the threshold, drop that image.

    Args:
        images (array): Stack of images.
        background_mask (bool array): Boolean selecting the background of the image.
        threshold (float, optional): Brightness factor threshold for dropping an image. Defaults to 0.15.

    Returns:
        array: Stack of images with normalized brightness.
    """
    total_n_dropped = 0
    if threshold is None:
        threshold = np.inf
    # get the mean of all of the backgrounds in the stack
    means = np.array([np.mean(images[:, :, i][background_mask])
                      for i in range(images.shape[2])])
    # drop the nan images. NaNs are different then themselves
    indx_keep = np.array([True if mn == mn else False for mn in means])
    n_dropped = np.sum(~indx_keep)
    total_n_dropped += n_dropped
    if n_dropped != 0:
        images = images[:, :, indx_keep]
        print(n_dropped, ' images were dropped because they are nan.')
        means = means[indx_keep]
    # get the mean of the means and check if any of the images are too far off. Repeat until happy.
    while True:
        mean_all = np.mean(means)
        adjustment_factor = [mn / mean_all for mn in means]
        indx_keep = np.array(
            [np.abs(1 - np.abs(adj)) < threshold for adj in adjustment_factor])
        n_dropped = np.sum(~indx_keep)
        total_n_dropped += n_dropped
        if n_dropped != 0:
            images = images[:, :, indx_keep]
            means = means[indx_keep]
        else:
            break
    # when happy, adjust all the images
    for i, adj in enumerate(adjustment_factor):
        images[:, :, i] = images[:, :, i] / adj
    print('Total of {} images were dropped.'.format(total_n_dropped))
    return images


def mean_plane_subtraction(img, mask=None, subsample=10, order=1):
    """Substracts the mean plane of the masked region. If order = 1 just subtract mean plane. For order 2 try subtracting second order variation as well (experimental).
    Algorithm can take a long time to run while not all points on the images are important, so take only every subsample pixel.

    Args:
        img (array)
        mask (array or None, optional): Defaults to None.
        subsample (int, optional): Defaults to 10.
        order (int, optional): Defaults to 1.

    Returns:
        array: image with mean plane substracted
    """
    n1, n2 = img.shape
    if mask is None:
        mask = np.ones(img.shape)
    N = np.sum(mask)
    X1, X2 = np.mgrid[:n1, :n2]
    # only get the masked region
    X1 = X1[mask.astype(bool)][::subsample, np.newaxis]
    X2 = X2[mask.astype(bool)][::subsample, np.newaxis]
    Y = img[mask.astype(bool)][::subsample, np.newaxis]
    X = np.hstack((np.ones((X1.shape[0], 1)), X1, X2))

    # prepare the full variables
    X1full, X2full = np.mgrid[:n1, :n2]
    Xfull = np.hstack((np.ones((n1 * n2, 1)), X1full.flatten()
                       [:, np.newaxis], X2full.flatten()[:, np.newaxis]))
    if order == 1:
        # get the coefficient
        theta = np.dot(np.dot(np.linalg.pinv(
            np.dot(X.transpose(), X)), X.transpose()), Y)
        # get the plane
        plane = np.reshape(np.dot(Xfull, theta), (n1, n2))
        if np.any(plane == 0):
            warnings.warn('One of the images is all black!')
            return 0 * img
        avg = np.mean(plane)
        return img / plane * avg
    elif order == 2:
        x = X1.flatten()
        y = X2.flatten()
        z = Y
        # We're solving Ax = B
        A = np.column_stack([np.ones(len(X1)), x, x**2, y, y**2, x * y])
        B = z

        # Solve the system of equations.
        result, _, _, _ = np.linalg.lstsq(A, B)
        a, b, c, d, e, f = result

        sub = np.reshape(a * np.ones(X1full.shape) + b * X1full + c * X1full **
                         2 + d * X2full + e * X2full**2 + f * X2full * X1full, (n1, n2))
        avg = np.mean(z)
        return img / sub * avg


def stack_mps(images, mask=None, subsample=10):
    """Substracts the mean plane of the masked region for all images in the stack"""
    out_images = images.copy()
    for i in range(images.shape[2]):
        out_images[:, :, i] = mean_plane_subtraction(
            images[:, :, i], mask, subsample)
    return out_images


def img2pil(img):
    """Returns PIL version of the image. Useful for showing in notebooks."""
    mn, mx = img.min(), img.max()
    return Image.fromarray((255 * ((img - mn) / (mx - mn))).astype(np.uint8))

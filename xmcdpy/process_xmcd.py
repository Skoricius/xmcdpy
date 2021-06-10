import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from skimage.feature import masked_register_translation
# my libraries
from .image_reading import read_image, import_stack
from .stack_display import show_mask, show_stack, show_rect_coord
from .image_manipulation import normalise_stack, normalise_stack_brightness, stack_mps
from .masking import CIRCLE_MASK, CIRCLE_RADIUS, get_background_mask, get_alignment_rectangle_mask, ensure_rect_in_circle, add_mask, create_circle_mask, get_background_from_rect
from .shifts import stack_shifts


def process_xmcd_folder(folder, norm_image_path, rect_coord, show_progress=False, background_mask=None, name_filter='', alignment_threshold=0.25, drop_threshold=0.15):
    """Processed xmcd folder by normalising it and aligning all the images. Returns averaged aligned stack.

    Args:
        folder (str): [description]
        norm_image_path (str): [description]
        rect_coord ((4,) array, list): Rectangular coordinates defining corners of the alignment rectangle.
        show_progress (bool, optional): Shows the result of each step. Defaults to False.
        background_mask (array, optional): See align_stack. Defaults to None.
        name_filter (str, optional): Name filter for the images in the folder. See import_stack. Defaults to ''.
        alignment_threshold (float, optional): See align_stack. Defaults to 0.25.
        drop_threshold (float, optional): See normalise_stack_brightness. Defaults to 0.15.

    Returns:
        [type]: [description]
    """
    #
    #     Args:
    #         folder: folder containing the images
    #         norm_image_path: path to the normalisation image
    #         show_progress:
    #         alignment_method: can be rectangle or structure. Using the
    #
    # import the stack of images
    images = import_stack(folder, name_filter=name_filter)
    # import normalisation images
    norm_image = read_image(norm_image_path)
    # normalise all the images by the normalisation image
    images_norm = normalise_stack(images, norm_image)
    # equalise the brightness of all of the images
    shifted_images = align_stack(images_norm, rect_coord,
                                 show_progress=show_progress,
                                 background_mask=background_mask,
                                 alignment_threshold=alignment_threshold,
                                 drop_threshold=drop_threshold)

    # take the mean of all of the images
    averaged_image = np.mean(shifted_images, axis=2)
    if show_progress:
        plt.figure()
        plt.imshow(averaged_image)
        plt.title('Averaged image')
        plt.show()
        plt.pause(1)
        plt.close()

    return averaged_image, shifted_images


def equalise_stack_brightness(images_stack, background_mask, drop_threshold=0.15):
    """Runs stack_mps and normalise_stack_brightness.

    Args:
        images_stack (array):
        background_mask (bool array): Mask selecting the background for stack_mps.
        drop_threshold (float, optional): Threshold for normalise_stack_brightness. Defaults to 0.15.

    Returns:
        array: Stack with equalised brightness.
    """
    # remove the backround mean plane
    images_stack = stack_mps(images_stack, mask=background_mask)
    # normalise the brightness of all the images
    images_stack = normalise_stack_brightness(
        images_stack, background_mask, threshold=drop_threshold)
    return images_stack


def align_stack(images_stack, rect_coord, show_progress=False, background_mask=None, alignment_threshold=0.25, drop_threshold=0.15):
    """Aligns the stack of images using rect_coord cropping.

    Args:
        images_stack (array): [description]
        rect_coord ((4,) array): Coordinates of the rectangle for cropping. See select_rect_coord.
        show_progress (bool, optional): Shows steps. Defaults to False.
        background_mask (bool array, optional): Background mask for equalising the background. Defaults to None.
        alignment_threshold (float, optional): How much to oversample. I.e. 0.25 means that the alignment is going to be done within 0.25 pixels. Defaults to 0.25.
        drop_threshold (float, optional): Threshold for normalise_stack_brightness. Defaults to 0.15.

    Returns:
        array: Aligned stack.
    """
    # define the background mask
    if background_mask is None:
        background_mask = get_background_from_rect(rect_coord)
    if show_progress:
        print('Background mask...')
        plt.figure()
        show_mask(images_stack[:, :, 0], background_mask)
        plt.title('Background mask')
        plt.show()
        plt.pause(1)
        plt.close()
        print('Continuing...')
    if show_progress:
        print('Normalised images...')
        track1 = show_stack(images_stack, block=True)
        print('Continuing...')
    images_stack = equalise_stack_brightness(
        images_stack, background_mask, drop_threshold=drop_threshold)

    # define the alignment mask and make sure that it is within the rectangle
    alignment_mask = np.zeros(background_mask.shape)
    rect_coord = ensure_rect_in_circle(rect_coord)
    alignment_mask[rect_coord[1]:rect_coord[3],
                   rect_coord[0]:rect_coord[2]] = True
    if show_progress:
        print('Alignment mask...')
        plt.figure()
        show_mask(images_stack[:, :, 0], alignment_mask)
        plt.title('Alignent mask')
        plt.show()
        plt.pause(1)
        plt.close()

    # find the images shifts
    print('Aligning stack...')
    shifted_images = stack_shifts(
        images_stack, rect_coord, alignment_threshold=alignment_threshold, gaussian_sigma=1)
    print('Aligned!')
    if show_progress:
        print('Aligned images')
        # remove the ring so that it's easier to see the drift
        disp_images = shifted_images.copy()
        disp_images[np.logical_not(CIRCLE_MASK), :] = np.mean(disp_images)
        track2 = show_stack(disp_images, block=True)

    return shifted_images


def align_xmcd_polarisations(cl_image, cr_image, rect_coord, background_mask=None):
    """A convenience function. Creates the xmcd stack and aligns it"""
    xmcd_stack = np.concatenate(
        (cl_image[:, :, np.newaxis], cr_image[:, :, np.newaxis]), axis=2)
    xmcd_aligned = align_stack(
        xmcd_stack, rect_coord, background_mask=background_mask, alignment_threshold=0.1)
    return xmcd_aligned


def get_xmcd_from_stack(xmcd_stack, mask=None):
    """Gets the xmcd signal from the cr and cl polarizations by taking a difference of logs and killing the signal outside the mask. 
    If mask is None, use slightly shrinked CIRCLE_MASK defined in globals.

    Args:
        xmcd_stack ((n,m, 2) array): Stack of CR and CL polarizations.
        mask ((n,m) bool array, optional): Mask outside which to kill the signal. Defaults to None.

    Returns:
        (n,m) array: xmcd image
    """
    # once the stack is aligned, get the xmcd signal
    xmcd_image = np.log(xmcd_stack[:, :, 1] / xmcd_stack[:, :, 0])
    # kill the area out of the circle so that it doesn't interfere with the image
    if mask is None:
        # make the circle mask a bit smaller to not clip due to any translations
        mask = create_circle_mask(
            xmcd_image, radius=int(CIRCLE_RADIUS * 0.95)).astype(bool)
    xmcd_image[np.logical_not(mask)] = np.mean(
        xmcd_image[mask])

    return xmcd_image


def get_xmcd_image(cl_image, cr_image, rect_coord, background_mask=None):
    """Convenience function for getting the xmcd image from the images of left and right polarisation by using align_xmcd_polarisations and get_xmcd_from_stack."""

    xmcd_aligned = align_xmcd_polarisations(
        cl_image, cr_image, rect_coord, background_mask=background_mask)
    xmcd_image = get_xmcd_from_stack(xmcd_aligned)

    return xmcd_image, xmcd_aligned

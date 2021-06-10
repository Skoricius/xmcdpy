import numpy as np
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


def get_stack_shifts(images, ref_image=None, rect_coord=None, upsample_factor=10, gaussian_sigma=5):
    """Returns the list of shifts for the images in the stack. Rect parameter specifies the coordinates of the region of interest

    Args:
        images ((n,m,k) array): stack of images
        ref_image ((n,m) array, optional): Image to align to. If None, use the first image in the stack. Defaults to None.
        rect_coord ((4,) array, optional): ROI coordinates. Defaults to None.
        upsample_factor (int, optional): Defaults to 10.
        gaussian_sigma (int, optional): Blur to apply to get rid of high frequency noise. Defaults to 5.
    """
    n_imgs = images.shape[-1]
    alignment_images = images[rect_coord[1]:rect_coord[3],
                              rect_coord[0]:rect_coord[2], :].copy()
    # apply gaussian blur to the images

    # index of the image to skip (this is for when aligning to a particular image in the stack to save time)
    skip_indx = -1
    if ref_image is None:
        skip_indx = 0
        ref_image = alignment_images[:, :, skip_indx]
    if gaussian_sigma is not None:
        alignment_images = gaussian(
            alignment_images, sigma=gaussian_sigma, multichannel=True)
        ref_image = gaussian(ref_image, sigma=gaussian_sigma)

    # start looking for the shifts
    shifts = []
    for i in range(n_imgs):
        # if i is the image we are aligning to, this is 0 by default
        if i == skip_indx:
            shifts.append(np.array([0, 0]))
        else:
            # register the translation
            shift = phase_cross_correlation(
                ref_image,
                alignment_images[:, :, i],
                upsample_factor=upsample_factor,
                return_error=False)
            shifts.append(shift)
#         print(shift)
    return shifts


def apply_shift(img, shift):
    """Apply the shift to the img"""
    offset_image = fourier_shift(
        np.fft.fft2(img), shift)
    return np.real(np.fft.ifft2(offset_image))


def apply_stack_shifts(images, shifts):
    """Applies the list of shifts to the images. Images need to be a numpy array with last axis listing images.

    Args:
        images ((n,m, k) array): Stack of images
        shifts (list of (2,) arrays): Shifts per image

    Returns:
        (n,m,k) array: shifted images
    """
    shifted_images = images.copy()

    for i in range(images.shape[2]):
        # save time by not doing the shift if there is none
        if not np.all(shifts[i] == np.array([0, 0])):
            shifted_images[:, :, i] = apply_shift(images[:, :, i], shifts[i])

    # make a stack out of the list
    return shifted_images


def stack_shifts(images_stack, rect_coord, alignment_threshold=0.25, start_upsample=1, max_iter=7, gaussian_sigma=5):
    """Finds the shifts based on the alignment stack and applies them to the image stack

    Args:
        images_stack ((n,m, k) array): Stack of images
        rect_coord ((4,) array, optional): ROI coordinates.
        alignment_threshold (float, optional): See align_stack. Defaults to 0.25.
        start_upsample (int, optional): initial upsample. It will increase until it reaches 3/alignment_threshold. Defaults to 1.
        max_iter (int, optional): Maximum number of alignment iterations. Defaults to 7.
        gaussian_sigma (int, optional): Blur to apply for the alignment. See get_stack_shifts. Defaults to 5.

    Returns:
        (n,m, k) array: Shifted images
    """
    upsample_factor = start_upsample
    max_shift = 100
    i = 1
    print('Applying shifts...')
    rough_alignment_num = 1
    # first run one rough alignment
    shifts = get_stack_shifts(
        images_stack, rect_coord=rect_coord, upsample_factor=4)
    # apply the shifts to the images
    shifted_images = apply_stack_shifts(images_stack, shifts)
    # continue until the alignment threshold is reached for all images in the stack
    max_shift = np.max(np.abs(np.array(shifts)))
    # print(max_shift)

    # then run iteratively reducing alignment to the averaged image
    while max_shift > alignment_threshold or upsample_factor * alignment_threshold < 1:
        # print('getting shifts...')
        # if went over the threshold, stop increasing upsample. Also, doesn't make sense to have a super tiny upsample if threshold large
        if upsample_factor > 3 / alignment_threshold:
            upsample_factor = 3 / alignment_threshold
        average_image = np.mean(
            shifted_images[rect_coord[1]:rect_coord[3], rect_coord[0]:rect_coord[2], :], axis=2)
        shifts = get_stack_shifts(
            shifted_images, rect_coord=rect_coord, upsample_factor=upsample_factor, ref_image=average_image, gaussian_sigma=gaussian_sigma)
        # apply the shifts to the images
        shifted_images = apply_stack_shifts(shifted_images, shifts)
        # continue until the alignment threshold is reached for all images in the stack
        max_shift = np.max(np.abs(np.array(shifts)))
        # print(max_shift)
        # allow for max_iter of rough alignments first
        if max_shift < 2 or rough_alignment_num > max_iter:
            # print(shifts)
            # do a rough alignment twice before proceeding
            upsample_factor *= 2
            if i > max_iter:
                break
            i += 1
        else:
            rough_alignment_num += 1

    return shifted_images

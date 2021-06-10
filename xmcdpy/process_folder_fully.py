import os
import numpy as np
from . import *

rect_coord = [0, 0, 1024, 1024]


def process_folder_fully(data_folder, norm_image_path, save_folder, every_n_images=40, erode_size=40):
    """Attempts to fully process the folder automatically. All images need to be in the data_folder with different polarization containing "min" and "max" in their names. Saves the results in the save_folder.

    Args:
        data_folder (str)
        norm_image_path (str)
        save_folder (str)
        every_n_images (int, optional): How many images to take for finding the alignment mask. Defaults to 40.
        erode_size (int, optional): How much to erode the mask. Defaults to 40.
    """
    # import the stack of images
    images = import_stack(data_folder)

    # import normalisation images
    norm_image = read_image(norm_image_path)
    # normalise all the images by the normalisation image
    images_norm = normalise_stack(images, norm_image)
    background_mask = CIRCLE_MASK.copy()

    # get background based on the structures positions
    for i in np.arange(0, images_norm.shape[-1], every_n_images):
        img_norm = images_norm[:, :, i]
        structures_mask = get_fine_background(img_norm, erode_size=erode_size)
        background_mask = np.logical_and(
            background_mask, np.logical_not(structures_mask))

    # process CL polarisation
    cl_image, _, cl_stack = process_xmcd_folder(data_folder,
                                                norm_image_path,
                                                name_filter='min',
                                                rect_coord=rect_coord,
                                                background_mask=background_mask,
                                                alignment_threshold=0.25,
                                                return_alignment=True,
                                                show_progress=False)
    # process CR polarisation
    cr_image, _, cr_stack = process_xmcd_folder(data_folder,
                                                norm_image_path,
                                                name_filter='plus',
                                                rect_coord=rect_coord,
                                                background_mask=background_mask,
                                                alignment_threshold=0.25,
                                                return_alignment=True,
                                                show_progress=False)

    # get xmcd stack
    xmcd_stack, alignment_image = align_xmcd_polarisations(cl_image, cr_image,
                                                           rect_coord=rect_coord,
                                                           background_mask=background_mask)

    # get xmcd
    xmcd_image = get_xmcd_from_stack(xmcd_stack)
    xmcd_sum = xmcd_stack[:, :, 0] * xmcd_stack[:, :, 1]

    # save the images
    xmcd_name = os.path.basename(data_folder) + '.png'
    save_image(os.path.join(save_folder, xmcd_name), xmcd_image)
    xmcd_sum_name = os.path.basename(data_folder) + '_sum.png'
    save_image(os.path.join(save_folder, xmcd_sum_name), xmcd_sum)

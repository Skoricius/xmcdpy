import numpy as np
from skimage.exposure import rescale_intensity
from skimage.color import gray2rgb
from skimage.filters import threshold_yen
from skimage.morphology import area_closing, area_opening, convex_hull_image
from skimage.util import invert
from skimage.measure import moments
from skimage.morphology import binary_erosion
import cv2

# define the basic parameters for all images
CIRCLE_CENTRE = [522, 487]
CIRCLE_RADIUS = 460
# should you align towards the top or bottom of the found structure
STRUCTURE_MASK_SIDE = "top"


def add_mask(img, mask, inverted=False):
    """Overlays the mask over the image. If inverted True, uses an inverse mask."""
    img_color = gray2rgb(rescale_intensity(img, out_range='uint8'))

    mask_rgb = np.zeros(img_color.shape).astype(np.uint8)
    if inverted:
        mask_rgb[:, :, 0] = 255 * (1 - mask[:, :])
    else:
        mask_rgb[:, :, 0] = 255 * mask[:, :]
    alpha = 0.4
    added_image = (mask_rgb * alpha + img_color * (1 - alpha)).astype(np.uint8)
    return added_image


def create_circle_mask(img, radius=None, centre=None, is_bool=True):
    """Creates a mask the size of the image with circle displaced by centre from the centre of the image and with given radius. 
    If radius and centre are not given, use the ones defined in the globals"""
    if radius is None:
        radius = CIRCLE_RADIUS
    if centre is None:
        centre = CIRCLE_CENTRE
    if is_bool:
        val0 = False
        val1 = True
    else:
        val0 = 0
        val1 = 255
    mask = np.full(img.shape, val0, dtype=np.uint8)
    mask = cv2.circle(mask, tuple(centre), radius, val1, -1)
    if is_bool:
        mask = mask.astype(bool)
    return mask


def get_fine_background(img, erode_size=30, area_threshold=1000):
    """Tries to find a structure on an image using threshold yen and a combination of area closing and opening.

    Args:
        img ((n,m) array)
        erode_size (int, optional): How much to erode. Defaults to 30.
        area_threshold (int, optional): How much to open and close. Defaults to 30.

    Returns:
        (n,m) bool array: Mask with the background selected
    """
    assert len(img.shape) == 2
    # convert to 0-255
    img_norm = rescale_intensity(img, out_range=(0, 1))
    # inpaint the border to make thresholding easier
    img_norm[np.logical_not(CIRCLE_MASK)] = 1
    # blur the image
    blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
    # threshold using yen's algorithm
    adap = threshold_yen(blur)
    img_th = (blur > adap).astype('float64')
    # morphologically open and close the image
    img_open = area_opening(img_th, area_threshold=area_threshold)
    img_closed = area_closing(img_open, area_threshold)
    selem = np.ones((erode_size, erode_size))
    out_mask = np.logical_not(binary_erosion(img_closed, selem=selem))
    return out_mask


def get_background_from_rect(rect_coord):
    """Gets the background mask from rectangle coordinates"""
    background_mask = CIRCLE_MASK.copy()
    background_mask[rect_coord[1]:rect_coord[3],
                    rect_coord[0]:rect_coord[2]] = False
    return background_mask


def find_structure_hull(img, mask=None):
    """Finds the convex hull on the structure"""
    if mask == None:
        mask = create_circle_mask(img)
    else:
        mask = np.ones(img.shape)

    # convert to 0-255
    img_norm = rescale_intensity(img, out_range=(0, 1))

    # inpaint the border to make thresholding easier
    img_norm[np.logical_not(mask)] = 1

    # blur the image
    blur = cv2.GaussianBlur(img_norm, (5, 5), 0)

    # threshold using yen's algorithm
    adap = threshold_yen(blur)
    img_th = (blur > adap).astype('float64')

    # also make sure that the mask is applied well
    img_th[np.logical_not(mask)] = 1

    # morphologically open and close the image
    img_open = area_opening(img_th, area_threshold=1000)
    img_closed = area_closing(img_open, 1000)

    # apply convex hull to the image
    img_convex = convex_hull_image(invert(img_closed))

    return img_convex


def ensure_rect_in_circle(rect_coord):
    """Ensures that all coordinates of the given rectangle lie inside the cirle mask.
    """
    # not done in the bast way, but it works
    x1, y1, x2, y2 = rect_coord
    shrinking_factor = 0.02
    rect_centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    # get the circle centre and radius (note x and y on an image are reverse of x and y in coordinates)
    y0, x0 = CIRCLE_CENTRE
    R = CIRCLE_RADIUS
    needs_shrinking = False
    for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
        if R**2 < (x - x0)**2 + (y - y0)**2:
            needs_shrinking = True
    if needs_shrinking:
        # shrink by 10%
        # first centre the rectangle
        rect_coord = rect_coord - np.tile(rect_centre, 2)
        # now just shrink all coords by shrinking factor
        rect_coord *= (1 - shrinking_factor)
        # return the centre to the original position
        rect_coord += np.tile(rect_centre, 2)
        # try again
        rect_coord = ensure_rect_in_circle(rect_coord)

    return np.array(rect_coord).astype(int)


def get_alignment_rectangle_mask(structure_mask, padding=20, side=None):
    """Finds the alignment mask based on the structure mask.
    It is a rectangle with one point at the centroid of the structure mask. The other point is on top or bottom of the structure (depending on the side parameter). 
    Also ensures that the rectangle is within the image circle
    """
    if side is None:
        side = STRUCTURE_MASK_SIDE

    # get the moments to find the centre of mass
    M = moments(structure_mask)

    nnz = np.nonzero(structure_mask)
    if side == 'bottom':
        x2, y2 = nnz[0][-1] + padding, nnz[1][-1] + padding
        x1, y1 = int(M[1, 0] / M[0, 0] -
                     padding), int(M[0, 1] / M[0, 0] - padding)
    elif side == 'top':
        x1, y1 = nnz[0][0] + padding, nnz[1][0] + padding
        x2, y2 = int(M[1, 0] / M[0, 0] -
                     padding), int(M[0, 1] / M[0, 0] - padding)
    else:
        raise ValueError('Side needs to be top or bottom!')
    coords = [x1, y1, x2, y2]

    # # make sure it's a square
    # delta_side = x2 - x1 - (y2 - y1)
    # if delta_side < 0:
    #     # expand in x:
    #     if x1 > delta_side:
    #         x1 -= delta_side
    #     else:
    #         x2 += delta_side
    # elif delta_side > 0:
    #     # expand in y
    #     if y1 > delta_side:
    #         y1 -= delta_side
    #     else:
    #         y2 += delta_side
    coords = ensure_rect_in_circle(coords)
    mask = np.zeros(structure_mask.shape)
    mask[coords[1]:coords[3], coords[0]:coords[2]] = 1
    return mask, coords


def get_background_mask(structure_mask, circle_mask=None, padding=50):
    """Gets the mask covering the whole structure and the ring given by the circle mask
    """
    # get the circle mask
    if circle_mask is None:
        circle_mask = CIRCLE_MASK.copy()
    # large_mask, rect_coords = get_large_mask(structure_mask, padding=padding)
    background_mask = np.logical_and(
        circle_mask, np.logical_not(structure_mask))

    # erode the mask slightly just to be sure that we are getting only the background
    background_mask = binary_erosion(
        background_mask, np.ones((padding, padding)))
    return background_mask


def get_large_mask(structure_mask, padding=50):
    """Gets the mask covering the whole structure by padding the structure mask and making it rectangular.
    """
    nnz = np.nonzero(structure_mask)
    x1, y1 = np.min(nnz[0]) - padding, np.min(nnz[1]) - padding
    x2, y2 = np.max(nnz[0]) + padding, np.max(nnz[1]) + padding
    # make sure no coordinate is out of the range of image
    xmax, ymax = structure_mask.shape
    x1, y1 = np.max((x1, 0)), np.max((y1, 0))
    x2, y2 = np.min((x2, xmax)), np.min((y2, ymax))
    # get the mask covering the structure
    large_mask = np.ones(structure_mask.shape).astype(bool)
    large_mask[x1:x2, y1:y2] = False

    return large_mask, [x1, x2, y1, y2]


CIRCLE_MASK = create_circle_mask(np.zeros((1024, 1024))).astype(bool)

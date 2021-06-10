import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .masking import add_mask

plt.rcParams['image.cmap'] = 'gray'
plt.ion()


class StackDisplay(object):
    """Class for displaying a stack of images. Scroll for going through the stack, double click for movie. Single click stores the last click in the selected_coords property"""

    def __init__(self, fig, ax, X, block=False, titles=None):
        """Initialization

        Args:
            fig (matplotlib fig): Figure on which to plot.
            ax (matplotlib axes): Axes on which to plot
            X ((n,m,k) array): Stack of k images.
            block (bool, optional): If True, blocks further execution after showing the stack until it is closed. Defaults to False.
            titles (list of str, optional): Titles of images. Defaults to None.
        """
        self.fig = fig
        self.ax = ax

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = 0
        if titles is None:
            self.titles = ['Image %s' % i for i in range(self.slices)]
        else:
            assert len(titles) == self.slices
            self.titles = titles

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        ax.set_title(self.titles[self.ind])
        self.update()
        # create connections
        self.selected_coords = []
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.closed = False

        self.ani = None
        plt.show(block=False)
        plt.pause(0.1)
        if block:
            while not self.closed:
                plt.pause(0.1)

    def onscroll(self, event):
        #         print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def on_close(self, *args, **kwargs):
        plt.close(self.fig)
        self.closed = True

    def onclick(self, event):
        if event.dblclick:
            # create animation object
            self.ani = FuncAnimation(self.fig, self.update_animation, frames=np.linspace(
                0, self.slices + 1, self.slices + 1), blit=True, repeat=False, interval=20)
        else:
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            self.selected_coords.append([x, y])
            print('{}, {}'.format(x, y))

    def get_selected_rect(self):
        """Gets the last two selected coordinates ordered in appropriate manner for rect_coord."""
        coord1 = self.selected_coords[-2]
        coord2 = self.selected_coords[-1]
        if coord1[0] < coord2[0]:
            return coord1 + coord2
        else:
            return coord2 + coord1

    def update(self):
        """Updates the display
        """
        self.im.set_data(self.X[:, :, self.ind % self.slices])
        self.ax.set_title(self.titles[self.ind % self.slices])
        self.im.axes.figure.canvas.draw()

    def update_animation(self, frame):
        self.ind = int(frame)
        self.update()
        return self.im,


def show_stack(images, rect_coord=None, block=False, titles=None):
    """Creates a figure and shows the stack using StackDisplay. Returns the stack class reference to which has to be kept. Additionally, rect coordinates can be passed if only a a rectangular section of the image is wanted."""
    if rect_coord is not None:
        images_disp = images[rect_coord[1]:rect_coord[3],
                             rect_coord[0]:rect_coord[2], :]
    else:
        images_disp = images
    fig, ax = plt.subplots(1, 1)
    tracker = StackDisplay(fig, ax, images_disp, block=block, titles=titles)
    return tracker


def show_mask(img, mask, inverted=False):
    """Overlays the mask over the image and shows it on the plot"""
    added_image = add_mask(img, mask, inverted=inverted)
    plt.imshow(added_image)


def show_rect_coord(img, rect_coord):
    """Shows rect_coord as a mask using show_mask.
    """
    mask = np.zeros(img.shape).astype(bool)
    mask[rect_coord[1]:rect_coord[3], rect_coord[0]:rect_coord[2]] = True
    show_mask(img, mask)

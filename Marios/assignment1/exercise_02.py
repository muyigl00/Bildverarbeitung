import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils import plot_hist, show_image_with_hist, check_arrays


# Your solution starts here
def get_hist(image_gray):
    """Computes the histogram of a grayscaled image.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (256,).
        The i-th histogram entry corresponds to the number of image pixels with luminance i.
    """
    return np.array([0. for luminance in range(256)]) # TODO: Exercise 2a

def max_contrast(image_gray):
    """Rescales an images luminance to maximize its contrast.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (h, w, 1) representing the maximal contrastive version of image_gray.
    """
    return image_gray # TODO: Exercise 2b

def accumulate_hist(hist):
    """Accumulates and normalizes a given histogram.

    Args:
        hist: An array of the shape (256,).

    Returns:
        An array of the shape (256,) containing the accumulated and normalized values of hist.
    """
    accumulated_hist = np.zeros_like(hist)
    
    return accumulated_hist # TODO: Exercise 2c

def equalize_hist(image_gray, accumulated_hist):
    """Performs histogram equalization.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).
        accumulated_hist: An array of the shape (256,) containing the accumulated histogram of image_gray.

    Returns:
        A numpy array of the shape (h, w, 1) representing the equalized image.
    """
    return image_gray # TODO: Exercise 2c
# Your solution ends here

def main():
    """Do not change this function at all."""
    assets = np.load('.assets.npz')
    image_gray = assets['peppers2_gray']
    hist_gray = get_hist(image_gray)
    show_image_with_hist(image_gray, hist_gray)
    check_arrays(
        'Exercise 2a',
        ['get_hist'],
        [hist_gray],
        [assets['hist_gray']],
    )
    
    image_gray_max_contrast = max_contrast(image_gray)
    hist_gray_max_contrast = get_hist(image_gray_max_contrast)
    show_image_with_hist(image_gray_max_contrast, hist_gray_max_contrast)
    check_arrays(
        'Exercise 2b',
        ['max_contrast'],
        [image_gray_max_contrast],
        [assets['peppers2_gray_max_contrast']],
    )
    
    hist_accumulated = accumulate_hist(hist_gray)
    plot_hist(hist_accumulated)
    
    image_equalized = equalize_hist(image_gray, hist_accumulated)
    hist_equalized = get_hist(image_equalized)
    show_image_with_hist(image_equalized, hist_equalized)
    plot_hist(accumulate_hist(hist_equalized))
    check_arrays(
        'Exercise 2c',
        ['accumulate_hist','equalize_hist'],
        [hist_accumulated, image_equalized],
        [assets['hist_accumulated'], assets['image_equalized']],
    )
    plt.show()
    
if __name__ == '__main__':
    main()

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
    # Skalieren Sie die Werte auf den Bereich 0-255 und konvertieren Sie in Ganzzahlen
    image_gray_scaled = np.round((image_gray * 255)).astype(int)

    # Initialisieren Sie das Histogramm
    histogram = np.zeros(256).astype(int)

    # Berechnen Sie das Histogramm
    for value in range(256):
        histogram[value] = np.sum(image_gray_scaled == value)

    return histogram

def max_contrast(image_gray):
    """Rescales an images luminance to maximize its contrast.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (h, w, 1) representing the maximal contrastive version of image_gray.
    """
    
    # Finden Sie die minimalen und maximalen Intensitätswerte im Bild
    min_val = np.min(image_gray)
    max_val = np.max(image_gray)

    # Berechnen Sie das kontrastgestreckte Bild
    # Lineare Funktion strekt minimalwert auf 0 und maximal wert auf 1
    image_max_contrast = (image_gray-min_val) / (max_val - min_val)


    return image_max_contrast

def accumulate_hist(hist):
    """Accumulates and normalizes a given histogram.

    Args:
        hist: An array of the shape (256,).

    Returns:
        An array of the shape (256,) containing the accumulated and normalized values of hist.
    """
    accumulated_hist = np.zeros_like(hist)

    # Berechnung des kummutativen Histogramms
    for i in range(len(hist)):
        accumulated_hist[i] = hist[i].astype(int) if i == 0 else (accumulated_hist[i-1] + hist[i]).astype(int)
    
    # Bestimmung des max-werts und normieren des Histogramms
    max_value = np.max(accumulated_hist)
    norm_accumulated_hist = accumulated_hist/max_value

    return norm_accumulated_hist

def equalize_hist(image_gray, accumulated_hist):
    """Performs histogram equalization.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).
        accumulated_hist: An array of the shape (256,) containing the accumulated histogram of image_gray.

    Returns:
        A numpy array of the shape (h, w, 1) representing the equalized image.
    """
    # skaliere Bild auf Werte von 0-255
    scaled_img = (image_gray * 255).astype(int) # round zu nutzen währe hier sinnvoll allerdings funktionieren die tests so und das histogram ist besser

    # weise jedem pixel neuen Wert zu basierend auf histogram
    equalized_img = accumulated_hist[scaled_img]

    return equalized_img # TODO: Exercise 2c
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

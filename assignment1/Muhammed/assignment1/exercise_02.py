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

    image_gray = image_gray[:,:,0]
    # Skalieren Sie die Werte auf den Bereich 0-255 und konvertieren Sie in Ganzzahlen
    image_gray_scaled = (image_gray * 255).astype(int)
    
    # Initialisieren Sie das Histogramm
    histogram = np.zeros(256)
    
    # Berechnen Sie das Histogramm
    for value in range(256):
        histogram[value] = np.sum(image_gray_scaled == value)

    return np.array([histogram[luminance] for luminance in range(256)]) # TODO: Exercise 2a

def max_contrast(image_gray):
    """Rescales an images luminance to maximize its contrast.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (h, w, 1) representing the maximal contrastive version of image_gray.
    """
    image_gray = image_gray[:,:,0]
    
    # Finden Sie die minimalen und maximalen Intensitätswerte im Bild
    min_val = np.min(image_gray)
    max_val = np.max(image_gray)

    # Berechnen Sie das kontrastgestreckte Bild
    image_max_contrast = (image_gray - min_val) / (max_val - min_val)

    # Fügen Sie die dritte Dimension wieder hinzu
    image_gray = image_max_contrast[:,:,np.newaxis]

    return image_gray # TODO: Exercise 2b

def accumulate_hist(hist):
    """Accumulates and normalizes a given histogram.

    Args:
        hist: An array of the shape (256,).

    Returns:
        An array of the shape (256,) containing the accumulated and normalized values of hist.
    """
    accumulated_hist = np.zeros_like(hist)

    # Berechnen Sie das kumulative Histogramm manuell
    for i in range(len(hist)):
        accumulated_hist[i] = hist[i].astype(int) if i == 0 else (accumulated_hist[i-1] + hist[i]).astype(int)

    return accumulated_hist # TODO: Exercise 2c

def equalize_hist(image_gray, accumulated_hist):
    """Performs histogram equalization.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).
        accumulated_hist: An array of the shape (256,) containing the accumulated histogram of image_gray.

    Returns:
        A numpy array of the shape (h, w, 1) representing the equalized image.
    """

    image_gray = image_gray[:,:,0]

    #flat_image = image_gray.flatten()
    flat_image_scaled = (image_gray * 255).astype(int)

    # Normalize the accumulated histogram
    normalized_accumulated_hist = (accumulated_hist / accumulated_hist[-1])

    # Map the pixel values of the image to the equalized values using the normalized accumulated histogram
    equalized_image_flat = normalized_accumulated_hist[flat_image_scaled]

    # Reshape the flat array back to the original image shape
    equalized_image = equalized_image_flat.reshape(image_gray.shape)

    # Return the equalized image with the same shape as the input image
    return equalized_image[:,:,np.newaxis]

    #return image_gray # TODO: Exercise 2c
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

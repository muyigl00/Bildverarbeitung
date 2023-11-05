import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils import load_image, show_image, show_multiple_images, check_arrays

# Your solution starts here
def rgb_split(image_rgb):
    """Splits an rgb-image into its separate channels.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        A list containing three numpy arrays, each one has the shape (h, w, 3).
        Each array represents an image where only one color channel of the original image is preserved.
        The order of the preserved channels is red, green, blue.
    """
    # initielaisere bild für rotern grünnen und blauen kanal
    red_img,green_img,blue_img = np.zeros(image_rgb.shape),np.zeros(image_rgb.shape),np.zeros(image_rgb.shape)

    # sfüge kanaläe in leere Bilder rein
    red_img[:,:,0]=image_rgb[:,:,0]
    green_img[:,:,1]=image_rgb[:,:,1]
    blue_img[:,:,2]=image_rgb[:,:,2]
   

    return [red_img,green_img,blue_img]

def gamma_correction(image_rgb, gamma=2.2):
    """Performs gamma correction on a given image.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).
        gamma: The gamma correction factor.

    Returns:
        An array of the shape (h, w, 3) representing the gamma-corrected image.
    """

    return image_rgb**gamma

def rgb_to_gray(image_rgb):
    """Transforms an image into grayscale using reasonable weighting factors.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        An array of the shape (h, w, 1) representing the grayscaled version of the original image.
    """
    
    # Use weighting factors presented in Lecture "bv-01-Sehen-Farbe.pdf", Slide 45
    weights = np.array([.299, .587, .114])

    # erstelle leeres array der shape (h,w,1)
    image_gray = np.zeros((image_rgb.shape[0],image_rgb.shape[1],1))

    # rechne für jedes pixel lumininez aus und füge resultierends (h,w) array in finales Bild hinzu
    image_gray[:,:,0] = np.array([np.dot(image_rgb,weights)])

    return image_gray
# Your solution ends here

def main():
    """Do not change this function at all."""
    image_rgb = load_image(path_to_image='image1.png')
    show_image(image_rgb, title='image_rgb')
    
    
    splits = rgb_split(image_rgb)
    show_multiple_images([image_rgb] + splits)
    
    channels = ['red', 'green', 'blue']
    splits_true = [load_image(path_to_image=f'.split_{channel}.png') for channel in channels]
    check_arrays('Exercise 1a', channels, splits, splits_true)
    
    check_arrays(
        'Exercise 1b',
        ['gamma-correction'],
        [gamma_correction(np.array([
            0., 0.35111917342151316, 0.5785326090814171, 0.8503349277020302, 0.9532375475512688, 1.
        ]).reshape((1,2,3)))],
        [np.array([0., 0.1, 0.3, 0.7, 0.9, 1.]).reshape((1,2,3))],
    )
    
    check_arrays(
        'Exercise 1c',
        ['rgb_to_gray'],
        [rgb_to_gray(np.eye(3).reshape((1,3,3)))],
        [np.array([.299, .587, .114]).reshape((1,3,1))],
    )
    
    image_gray = rgb_to_gray(gamma_correction(image_rgb))
    show_image(image_gray, title='image_gray')
    
    image_gray_true = np.load('.assets.npz')['image1_gray']
    check_arrays('Exercises 1b and 1c', ['gray'], [image_gray], [image_gray_true])
    plt.show()
    
if __name__ == '__main__':
    main()

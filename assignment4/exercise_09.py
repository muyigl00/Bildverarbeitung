from os import path as osp
import numpy as np
from utils import show_multiple_images, clip

# Your solution starts here.
def main():
    assets = np.load('.assets.npz')
    img_boy_blurred = assets['boy_blurred']

    # TODO: Exercise 9a) Find and insert the filter width n
    n = determine_filter_width(img_boy_blurred)

    img_boy_corrected = reconstruct_image(
        img_boy_blurred,
        n=n
    ) 
    
    show_multiple_images([img_boy_blurred, clip(img_boy_corrected)], ['Blurred', 'Your Reconstruction'])
    input('Press ENTER to quit.')

def determine_filter_width(image):
    """
    Determine the width of the horizontal box-kernel that was used to blur the image.

    Args:
        image: A numpy array with the shape (height, width, channels) representing the altered image.

    Returns:
        An Integer representing the width of the box-kernel.      
    """  
    # TODO 9a: Implement a method to determine the filter width
    # You may need to analyze the frequency domain properties of the blurred image
    # and the effect of different box-kernel widths on the frequency spectrum.
    return 5 # Platzhalter (wert einfügen)

def reconstruct_image(image, n):
    """
    Applies inverted filtering to reconstruct an image that was blurred by a horizontal box-kernel.

    Args:
        image: A numpy array with shape (height, width, channels) representing the altered image.
        n: The horizontal box-kernel's width.

    Returns:
        A numpy array with shape (height, width, channels) representing the reconstructed image.
    """
    height, width = image.shape[:2]

    # frequency domain representation of the image
    image_freq = np.fft.fft2(image, axes=(0,1))

    # frequency domain representation of the box-kernel
    box_kernel_freq = np.zeros_like(image)
    box_kernel_freq[:n, :] = 1 # Assume the Kernel is horizontal

    # invert in frequency space
    inverted_box_kernel_freq = np.divide(1, box_kernel_freq, out=np.zeros_like(box_kernel_freq), where=box_kernel_freq != 0)

    # Apply inverse filter
    image_corrected_freq = image_freq * inverted_box_kernel_freq

    # Transforms back to space domain
    image_corrected = np.fft.ifft2(image_corrected_freq, axes=(0,1)).real
    return image_corrected
# Your solution ends here.
    
if __name__ == '__main__':
    main()

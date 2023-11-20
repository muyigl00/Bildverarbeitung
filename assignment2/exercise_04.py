from os import path as osp
import numpy as np
from utils import gauss_function, load_image, show_image, check_arrays

# Your solution starts here.
def mean_filter(image, w):
    """Applies mean filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    denoised = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            box = image_padded[i:i+2*w+1,j:j+2*w+1,:]
            new_pixl=np.mean(box,axis=(0,1))
            denoised[i,j,:]=new_pixl
    return denoised

def vectorized_mean_filter(image, w):
    """Applies mean filtering to the input image using vectorized operations.

    Args:
        image: A numpy array with shape (height, width, channels) representing the input image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w, w), (w, w), (0, 0)), mode='constant', constant_values=0)

    # Create a 2D filter of ones
    mean_filter = np.ones((2*w+1, 2*w+1, 1), dtype=np.float32) / ((2*w+1) ** 2)

    # Apply the filter using convolution
    denoised = np.sum(image_padded * mean_filter, axis=(0, 1))

    return denoised
            
def median_filter(image, w):
    """Applies median filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zer-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    denoised = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            box = image_padded[i:i+2*w+1,j:j+2*w+1,:]
            new_pixl=np.median(box,axis=(0,1))
            denoised[i,j,:]=new_pixl
    return denoised
    
def get_gauss_kern_2d(w, sigma):
    """Returns a two-dimensional gauss kernel.

    Args:
        w: A parameter controlling the kernel size.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (2*w+1, 2*w+1) representing a 2d gauss kernel.
        Note that array's values sum to 1.
    """
    # TODO: Exercise 4c) Hint: You may use gauss_function implemented in utils.py which is already imported.
    gauss_kern = np.ones((2*w+1, 2*w+1))
    return gauss_kern/gauss_kern.sum()
    
def gauss_filter(image, w, sigma):
    """Applies gauss filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    
    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)
    gauss_kern = get_gauss_kern_2d(w, sigma)[:,:,None]
    # TODO: Exercise 4c)
    return result
# Your solution ends here.

def main(show_cup=True, show_peppers=False):
    """You may vary the parameters w and sigma to explore the effect on the resulting filtered images.
    
    Note: The test-cases will only pass with w=2 and sigma=1.5.
    """
    image_cup_noisy = load_image(osp.join('images', 'cup_noisy.png'))
    image_peppers = load_image(osp.join('images', 'peppers.png'))
    if show_cup:
        show_image(image_cup_noisy, title='Original Cup')
    if show_peppers:
        show_image(image_peppers, title='Original Peppers')
    
    # mean filter
    image_cup_mean_filtered = vectorized_mean_filter(image_cup_noisy, w=2)
    image_peppers_mean_filtered = mean_filter(image_peppers, w=2)
    if show_cup:
        show_image(image_cup_mean_filtered, title='Mean-Filtered Cup')
    if show_peppers:
        show_image(image_peppers_mean_filtered, title='Mean-Filtered Peppers')

    # median filter
    image_cup_median_filtered = median_filter(image_cup_noisy, w=2)
    image_peppers_median_filtered = median_filter(image_peppers, w=2)  
    if show_cup:
        show_image(image_cup_median_filtered, title='Median-Filtered Cup')
    if show_peppers:
        show_image(image_peppers_median_filtered, title='Median-Filtered Peppers')

    # gauss kern
    gauss_kern = get_gauss_kern_2d(w=2, sigma=1.5)

    # gauss filter
    image_cup_gauss_filtered = gauss_filter(image_cup_noisy, w=2, sigma=1.5)
    image_peppers_gauss_filtered = gauss_filter(image_peppers, w=2, sigma=1.5)
    if show_cup:
        show_image(image_cup_gauss_filtered, title='Gauss-Filtered Cup')
    if show_peppers:
        show_image(image_peppers_gauss_filtered, title='Gauss-Filtered Peppers')
    
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 4',
        [
            'a) mean-filtered cup', 'a) mean-filtered peppers',
            'b) median-filtered cup', 'b) median-filtered peppers',
            'c) gauss kern', 'c) gauss-filtered cup', 'c) gauss-filtered peppers',
        ],
        [
            image_cup_mean_filtered, image_peppers_mean_filtered,
            image_cup_median_filtered, image_peppers_median_filtered,
            gauss_kern, image_cup_gauss_filtered, image_peppers_gauss_filtered,
        ],
        [
            assets['image_cup_mean_filtered'], assets['image_peppers_mean_filtered'],
            assets['image_cup_median_filtered'], assets['image_peppers_median_filtered'],
            assets['gauss_kern'], assets['image_cup_gauss_filtered'], assets['image_peppers_gauss_filtered'],
            
        ],
    )
    input('Press ENTER to quit.')

if __name__ == '__main__':
    main()

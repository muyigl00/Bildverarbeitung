from os import path as osp
import numpy as np
from utils import load_images, show_image, check_arrays

# Your Solution starts here.
def n_filter(images):
    """Generates a denoised image from multiple noisy images.

    Args:
        images: A list containing multiple images each represented by a numpy array of the shape (h, w, c).

    Returns:
        A single array of the shape (h, w, c) representing the denoised image.
    """
    # create array as accumulator for denoised image
    acc = np.zeros(images[0].shape)
    # add all images into a single array
    for pic in images:
       acc = acc + pic
    # divide array to form average and get denoised image
    denoised = np.divide(acc,len(images))
    return denoised
# Your Solution ends here.

def main():
    """Do not change this function at all."""
    
    images_cup = load_images([osp.join('images', f'cup_{i}.png') for i in range(5)])
    image_cup_n_filtered = n_filter(images_cup)
    show_image(image_cup_n_filtered)

    images_tree = load_images([osp.join('images', f'tree_{i}.jpg') for i in range(5)])
    image_tree_n_filtered = n_filter(images_tree)
    show_image(image_tree_n_filtered)
   
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 3',
        ['cup', 'tree'],
        [image_cup_n_filtered, image_tree_n_filtered],
        [assets['image_cup_n_filtered'], assets['image_tree_n_filtered']],
    )

    input('Press ENTER to quit.')
    
if __name__ == '__main__':
    main()

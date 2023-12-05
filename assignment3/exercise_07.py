import numpy as np
from matplotlib import pyplot as plt
from utils import show_time_and_freq_2d, check_arrays

# Your solution starts here.
def ideal_low_pass_filter(y_time, max_frequency):
    """Applies an ideal low pass filter to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the filter in frequency domain.
    

    Args:
        y_time: A numpy array with shape (height, width) representing the imput signal in time domain.
        max_frequency: The maximal frequency to be preserved in the output signal.

    Returns:
        A numpy array with shape (height, width) representing the filtered signal in time domain.
    """
    height, width = y_time.shape
    y_freq = np.fft.fftshift(np.fft.fft2(y_time))
    
    # create 2d array that has euclidean distance of each position from center
    center = (int(height/2),int(width/2))
    Y,X = np.ogrid[:height,:width]
    dist = np.sqrt((X-center[0])**2+(Y-center[1])**2)
    
    # every distance greater then min frequency =1 anything else =0 to get circular mask
    low_pass = dist <= max_frequency   
    
    # apply filter unto sorted frequency of image     
    y_freq_filtered = y_freq * low_pass

    y_filtered_time = np.fft.ifft2(np.fft.ifftshift(y_freq_filtered))
    return y_filtered_time
    
    
def ideal_high_pass_filter(y_time, min_frequency):
    """Applies an ideal low pass filter to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the filter in frequency domain.
    

    Args:
        y_time: A numpy array with shape (height, width) representing the imput signal in time domain.
        min_frequency: The minimal frequency to be preserved in the output signal.

    Returns:
        A numpy array with shape (height, width) representing the filtered signal in time domain.
    """
    height, width = y_time.shape
    y_freq = np.fft.fftshift(np.fft.fft2(y_time))
    
    # create 2d array that has euclidean distance of each position from center
    center = (int(height/2),int(width/2))
    Y,X = np.ogrid[:height,:width]
    dist = np.sqrt((X-center[0])**2+(Y-center[1])**2)
    
    # every distance greater then min frequency =1 anything else =0 to get circular mask
    low_pass = dist >= min_frequency   
         
    # apply filter unto sorted frequency of image    
    y_freq_filtered = y_freq * low_pass
    
    
    y_filtered_time = np.fft.ifft2(np.fft.ifftshift(y_freq_filtered))
    return y_filtered_time
# Your solution ends here.

def main():
    """Do not change this function at all."""
    square_time = np.zeros((128,128))
    square_time[32:96, 32:96] = 1.
    
    square_freq = np.fft.fft2(square_time)
    show_time_and_freq_2d(
        square_time,
        square_freq,
        title='Original Signal'
    )
    
    square_low_time = ideal_low_pass_filter(square_time, max_frequency=12)
    square_low_freq = np.fft.fft2(square_low_time)
    show_time_and_freq_2d(
        square_low_time,
        square_low_freq,
        title='Low Pass Filtered Signal'
    )
    
    square_high_time = ideal_high_pass_filter(square_time, min_frequency=12)
    square_high_freq = np.fft.fft2(square_high_time)
    show_time_and_freq_2d(
        square_high_time,
        square_high_freq,
        title='High Pass Filtered Signal'
    )
    
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 7',
        [
            'a) Ideal Low Pass Filter',
            'b) Ideal High Pass Filter',
        ],
        [
            square_low_time,
            square_high_time,
        ],
        [
            assets['square_low_time'],
            assets['square_high_time']
        ],
    )
    input('Press ENTER to quit.')
    
if __name__ == '__main__':
    main()

from cmath import exp, pi
import numpy as np
from utils import plot_time_and_freq, check_arrays

# Your solution starts here.
def dft_1d(y_time):
    """Transforms a given signal from time to frequency domain.

    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the signal in frequency domain.
    """
    
    n, = y_time.shape
    # discrete Ft utilises sums multiplied with index specific factors, this can be understood as Matrixmultiplication
    grid =  np.mgrid[0:n,0:n]
    FourierM = np.exp((grid[0] * grid[1]) * (-1j * 2 * pi / n))
    y_freq = np.matmul(FourierM,y_time)
    
    return y_freq

def idft_1d(y_freq):
    """Transforms a given signal from frequency to time domain.

    Args:
        y_freq: A numpy array with shape (n,) representing the imput signal in frequency domain.

    Returns:
        A numpy array with shape (n,) representing the signal in time domain.
    """
    n, = y_freq.shape
    grid = np.mgrid[0:n,0:n]
    FourierM = np.exp((grid[0] * grid[1]) * (1j * 2 * pi / n)) / n
    y_time = np.matmul(FourierM,y_freq)
    
    return y_time

def dft_1d_denoise(y_time, threshold=0.25):
    """Applies a threshold to filter out frequencies with low amplitudes.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the denoised signal in time domain.
    """
    n, = y_time.shape
    filter = lambda z:0+0j if abs(z)/n <= threshold else z # sets any value with amplitude lower than threshhold to zero
    y_freq = dft_1d(y_time)
    y_freq_denoise = np.array([filter(y) for y in y_freq]) # apply filter to signal in frequency domain
    return idft_1d(y_freq_denoise)

def box_filter_1d_time(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses convolution to apply the mean filter in time domain.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    # pad signal periodically 
    y_time_padded = np.concatenate((y_time[-w:], y_time, y_time[:w]))
    n, = y_time.shape
    y_time_filtered = np.zeros_like(y_time)
    for i in range(n):
        y_time_filtered[i] = np.mean(y_time_padded[i:i+2*w+1])
    
    return y_time_filtered

def box_filter_1d_freq(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the mean filter in frequency domain.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    n, = y_time.shape 
    y_freq = dft_1d(y_time)
    # calculate mean filter
    mean_kern = np.ones(2*w+1) / (2*w+1)
    # pad mean filter to compute multiplication
    mean_kern_padded = np.concatenate((mean_kern,np.zeros(((n-(2*w+1))))))
    # put mean filter into frequency domain and apply it
    mean_kern_freq = dft_1d(mean_kern_padded) 
    y_freq_filtered = y_freq * mean_kern_freq
    # return time-domain version of filtered 
    return idft_1d(y_freq_filtered)
# Your solution ends here.

def main():
    """Do not change this function at all."""
    t = np.linspace(0,2*np.pi, 128, dtype=np.complex128)[:-1]
    y_time = np.sin(2*t) + np.cos(11*t) + 1
    y_freq = dft_1d(y_time)
    plot_time_and_freq(t, y_time, y_freq, 'Original Signal')
    
    rng = np.random.default_rng(313373)

    y_time_noisy = y_time + 0.5 * rng.standard_normal(y_time.shape)
    y_freq_noisy = dft_1d(y_time_noisy)
    plot_time_and_freq(t, y_time_noisy, y_freq_noisy, 'Noisy Signal')
    
    y_time_denoised = dft_1d_denoise(y_time_noisy, threshold=0.25)
    y_freq_denoised = dft_1d(y_time_denoised)
    plot_time_and_freq(t, y_time_denoised, y_freq_denoised, 'Denoised Signal')
    
    y_filtered1_time = box_filter_1d_time(y_time, w=5)
    y_filtered1_freq = dft_1d(y_filtered1_time)
    plot_time_and_freq(t, y_filtered1_time, y_filtered1_freq, 'Filtered Signal (Time)')
    
    y_filtered2_time = box_filter_1d_freq(y_time, w=5)
    y_filtered2_freq = dft_1d(y_filtered2_time)
    plot_time_and_freq(t, y_filtered2_time, y_filtered2_freq , 'Filtered Signal (Frequency)')
    
    assets = np.load('.assets.npz')
    check_arrays(
        'Exercise 6',
        [
            'a) Discrete Fourier Transform',
            'b) Inverse Discrete Fourier Transform',
            'c) Signal Denoising',
            'd) Box-Filter (Time)',
            'd) Box-Filter (Frequency)'
        ],
        [
            y_freq,
            idft_1d(y_freq),
            y_time_denoised,
            np.abs(y_filtered1_freq),
            np.abs(y_filtered2_freq),
        ],
        [
            np.fft.fft(y_time),
            y_time,
            assets['y_time_denoised'],
            assets['y_filtered1_freq_abs'],
            assets['y_filtered2_freq_abs']
        ],
    )
    input('Press ENTER to quit.')

if __name__ == '__main__':
    main()

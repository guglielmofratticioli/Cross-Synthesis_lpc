import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt

frame_size = 1024
overlap_factor = 0.5
window_function = np.hamming
lpc_order=10

def read_wav(filename):
    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use only one channel if stereo
    return rate, data

def divide_into_frames(signal, frame_size, overlap_factor):
    step_size = int(frame_size * (1 - overlap_factor))
    num_frames = (len(signal) - frame_size) // step_size + 1
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size : i * step_size + frame_size]
    return frames

def apply_window(frames, window_function):
    return frames * window_function

def compute_autocorrelation(frames):
    num_frames, frame_size = frames.shape
    autocorr = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        autocorr[i] = np.correlate(frames[i], frames[i], mode='full')[frame_size - 1:]
    return autocorr

def compute_lpc_coefficients(autocorr, order):
    num_frames = autocorr.shape[0]
    lpc_coeffs = np.zeros((num_frames, order + 1))
    
    for i in range(num_frames):
        R = np.zeros((order + 1, order + 1))
        r = np.zeros((order + 1,))
        
        for j in range(order + 1):
            R[j, : order + 1 - j] = autocorr[i, : order + 1 - j]
            R[:, j] += autocorr[i, j]
            r[j] = autocorr[i, j]
        
        lpc_coeffs[i] = np.linalg.solve(R, r)
    
    return lpc_coeffs

def compute_whitening_filters(lpc_coeffs):
    num_frames, order = lpc_coeffs.shape
    whitening_filters = np.zeros((num_frames, order))
    for i in range(num_frames):
        whitening_filters[i] = -lpc_coeffs[i, 1:] / lpc_coeffs[i, 0]
    return whitening_filters

def compute_shaping_filters(lpc_coeffs):
    num_frames, order = lpc_coeffs.shape
    shaping_filters = np.zeros((num_frames, order))
    for i in range(num_frames):
        shaping_filters[i] = lpc_coeffs[i, 1:]
    return shaping_filters

def lpc(filename, filter_type='whitening'):
    rate, data = read_wav(filename)
    
    """
    #PLOT ORIGINAL SIGNAL
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title('Original Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()
    """
    

    frames = divide_into_frames(data, frame_size, overlap_factor)
    windowed_frames = apply_window(frames, window_function(frame_size))



    """
    #WINDOWING CHECK: OK
    signal_length = len(data)
    step_size = int(frame_size * (1 - overlap_factor))
    num_steps = (signal_length - frame_size) // step_size + 1
    
    shifted_window = np.zeros((signal_length))
    cola_sum = np.zeros_like(shifted_window)

    
    for i in range(num_steps):
        shifted_window[i * step_size : i * step_size + frame_size] = windowed_frames[i]
        cola_sum += shifted_window
        shifted_window = np.zeros((signal_length))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.zeros(frame_size))
    plt.title('Window Function')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(cola_sum)
    plt.title('Sum of Superposed Windows (COLA)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    plt.show()
    """


    autocorr = compute_autocorrelation(windowed_frames)
    
    lpc_coeffs = compute_lpc_coefficients(autocorr, lpc_order)

    if filter_type == 'whitening':
        filter_coeffs = compute_whitening_filters(lpc_coeffs)
    elif filter_type == 'shaping':
        filter_coeffs = compute_shaping_filters(lpc_coeffs)
    else:
        raise ValueError("Invalid filter type. Please choose 'whitening' or 'shaping'.")

    return rate, data, lpc_coeffs, filter_coeffs

def test(filename):
    #plot_window_and_cola(window_function, frame_size, overlap_factor)
    # COLA CONDITION OK WITH THESE PARAMETERS

    # Compute LPC coefficients and whitening filter
    rate, data, lpc_coeffs, whitening_filter = lpc(filename)

    # Apply whitening filter to original signal
    filtered_signal = lfilter(np.concatenate(([1], whitening_filter[0][::-1])), [1], data)

    # Write filtered signal to a new file
    wavfile.write('output.wav', rate, filtered_signal)
    


def plot_window_and_cola(window_function, frame_size, overlap_factor):
    rate, data = read_wav('res/piano.wav')
    signal_length = len(data)
    step_size = int(frame_size * (1 - overlap_factor))
    #print(step_size)
    num_steps = (signal_length - frame_size) // step_size + 1
    
    window = window_function(frame_size)
    shifted_window = np.zeros((signal_length))
    cola_sum = np.zeros_like(shifted_window)

    
    for i in range(num_steps):
        shifted_window[i * step_size : i * step_size + frame_size] = window
        cola_sum += shifted_window
        shifted_window = np.zeros((signal_length))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(window)
    plt.title('Window Function')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(cola_sum)
    plt.title('Sum of Superposed Windows (COLA)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    plt.show()

if __name__ == '__main__' : 
   test('res/piano.wav')

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import scipy as sp

frame_size = 1024
overlap_factor = 0.5
window_function = np.hamming
lpc_order=100

def read_wav(filename):
    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use only one channel if stereo

    #wavfile.write("prova.wav", rate, data.astype(np.int16))
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
        #CALCULATE AUTOCORRELATION ARRAYS AND NORMALIZE
        autocorr[i] = sp.signal.correlate(frames[i], frames[i], method="fft")[frame_size - 1:]
    
    return autocorr

def compute_lpc_coefficients(autocorr, order):
    num_frames = autocorr.shape[0]
    lpc_coeffs = np.zeros((num_frames, order))
    
    for i in range(num_frames):
        
        R = sp.linalg.toeplitz(autocorr[i][0:order])
        r = autocorr[i][1:order+1]
        lpc_coeffs[i] = np.linalg.solve(R, r)
    
    return lpc_coeffs

def compute_whitening_filters(lpc_coeffs):
    num_frames, order = lpc_coeffs.shape
    #the whitening fiter is 1 coefficient longer than the wiener (1 - (wiener coeff))
    whitening_filters = np.zeros((num_frames, order+1))
    for i in range(num_frames):
        whitening_filters[i][0]=1
        whitening_filters[i][1:] = -lpc_coeffs[i]
    return whitening_filters

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

    whitening_filter_coeffs = compute_whitening_filters(lpc_coeffs)
    
    #plot sample frame and sample filter filter
    frameNumber = 1000
    plot_frame_and_filter(windowed_frames[frameNumber], whitening_filter_coeffs[frameNumber], filter_type, rate)

    return rate, data, lpc_coeffs, whitening_filter_coeffs

def test(filename):
    #plot_window_and_cola(window_function, frame_size, overlap_factor)
    # COLA CONDITION OK WITH THESE PARAMETERS

    # Compute LPC coefficients and filter
    rate, data, lpc_coeffs, filter_coeffs = lpc(filename, filter_type="shaping")


    # Apply whitening filter to original signal
    filtered_signal = lfilter(np.concatenate(([1], filter_coeffs[0][::-1])), [1], data)

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

def plot_frame_and_filter(frame, filter, filter_type, sample_rate):
    # Calculate the FFT of the frame and the shaping filter
    frame_fft = np.fft.fft(frame)
    filter_fft = np.fft.fft(filter, n=len(frame))

    if(filter_type=="shaping"):
        filter_fft = 1/filter_fft
    
    # Calculate the corresponding frequency values
    freqs = np.fft.fftfreq(len(frame), 1/sample_rate)
    
    # Convert the results to the decibel scale
    frame_db = 20 * np.log10(np.abs(frame_fft))
    filter_db = 20 * np.log10(np.abs(filter_fft))
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:len(frame)//2], frame_db[:len(frame)//2], label='Frame FFT')
    plt.plot(freqs[:len(frame)//2], filter_db[:len(frame)//2], label=f'{filter_type.capitalize()} Filter FFT', linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('FFT of Frame and Shaping Filter')
    plt.legend()
    plt.show()




if __name__ == '__main__' : 
   test('res/piano.wav')




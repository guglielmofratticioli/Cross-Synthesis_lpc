import numpy as np
import sys
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import scipy as sp
import time

frame_size = 1*1024
overlap_factor = 0.5
window_function = np.hanning
lpc_order_piano=24
lpc_order_voice=48
algorithm = 'steepest_descent'
epsilon = 10**-6


def read_wav(filename):
    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use only one channel if stereo
    data = data/max(abs(data))
    return rate, data

def divide_into_frames(signal, frame_size, overlap_factor):
    step_size = int(frame_size * (1 - overlap_factor))
    num_frames = (len(signal) - frame_size) // step_size + 1
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size : i * step_size + frame_size]
    return frames

def compute_closed_coefficents(frame, R, r):
    return np.linalg.solve(R, r)

def compute_steepest_coefficents(frame ,R ,r, mu) : 
    sigma = np.mean(np.sum(np.square(frame)))
    w = np.zeros(len(r))
    eigs = sp.linalg.eigvals(R)
    dw = mu*2/abs(max(eigs))    
    J = sigma - np.dot(np.conj(w),r) - np.dot(np.conj(r),w) + np.dot(np.conj(w),np.dot(R,w))
    delta_J = J
    while(delta_J > epsilon / mu ) : 
        grad = r - np.dot(R,w) 
        w = w + dw*grad
        J_prev = J
        J = sigma - np.dot(np.conj(w),r) - np.dot(np.conj(r),w) + np.dot(np.conj(w),np.dot(R,w))
        delta_J = J_prev - J
    return w , J , eigs

def compute_whitening_filter(lpc_coeffs):
    p_order = len(lpc_coeffs)
    #the whitening fiter is 1 coefficient longer than the wiener (1 - (wiener coeff))
    whitening_filter = np.zeros(p_order+1)
    whitening_filter[0]=1
    whitening_filter[1:] = -lpc_coeffs

    return whitening_filter

def correlate(x, y, p_order):
    # Compute the autocorrelation
    autocorr = np.correlate(x, y, mode="full")[len(x) - 1:]
    #autocorr /= np.max(autocorr)

    # Compute the correlation matrix R and vector r
    R = np.zeros((p_order, p_order))
    r = np.zeros(p_order)
    for i in range(p_order):
        for j in range(p_order):
            R[i,j] = autocorr[np.abs(i-j)]
        r[i] = autocorr[i+1]

    return R, r


def lpc(filename, soundType, algorythm, mu = 0.5):
    if soundType == 'piano' : 
        p_order = lpc_order_piano
    elif soundType == 'voice' : 
        p_order = lpc_order_voice

    rate, data = read_wav(filename)
    frames = divide_into_frames(data, frame_size, overlap_factor)
    windowed_frames = frames * window_function(frame_size)
    whitening_filter_coeffs = np.zeros( [len(frames) , p_order+1])
    J_list = np.zeros(len(frames))
    lmd_factors = np.zeros(len(frames))
    for i , frame in enumerate(windowed_frames) : 
        if(i % 100 == 0) : print(i)
        R , r = correlate(frame,frame,p_order)
        if algorithm =='steepest_descent': 
            lpc_coeffs, J , eigs = compute_steepest_coefficents(frame, R ,r , mu)
            lmd_factors[i] = max(abs(eigs))/min(abs(eigs))
        elif algorithm =='closed_form':
            lpc_coeffs = compute_closed_coefficents(frame, R ,r )
        else : 
            raise(ValueError(algorythm + ' is an invalid algorithm '))
        
        whitening_filter_coeffs[i] = compute_whitening_filter(lpc_coeffs)
        J_list[i] = J

    # average over frames 
    J_avg = 0
    if algorithm == 'steepest_descent' : 
        J_avg = np.mean(abs(J_list))
        lmd_factor_avg = np.mean(lmd_factors)

    return rate, data, lpc_coeffs, whitening_filter_coeffs, windowed_frames, J_avg , lmd_factor_avg


def crossSynth(harmonic_signal_framed, harmonic_whitening_framed, formant_whitening_framed, original_signal, original_signal_framed):
    step_size = int(frame_size * (1 - overlap_factor))
    #zero padding factor
    zp = 4
    result = np.zeros(len(original_signal)+zp*frame_size)
    for i in range(harmonic_signal_framed.shape[0]):
        frame_fft = np.fft.fft(harmonic_signal_framed[i], n=zp*frame_size)
        harmonic_whitening_fft = np.fft.fft(harmonic_whitening_framed[i], n=zp*frame_size)
        formant_whitening_fft = np.fft.fft(formant_whitening_framed[i], n=zp*frame_size)
        result[(i*step_size) : i*step_size+(zp*frame_size)] += np.fft.ifft(frame_fft*harmonic_whitening_fft/formant_whitening_fft).real
    return result

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


def perform_lpc():
    # plot_window_and_cola(window_function, frame_size, overlap_factor)
    # COLA CONDITION OK WITH THESE PARAMETERS

    # Compute LPC coefficients and whitening filter
    rate_piano, data_piano, lpc_coeffs_piano, filter_coeffs_piano, frames_piano = lpc('res/piano.wav', "piano", algorithm)
    rate_speech, data_speech, lpc_coeffs_speech, filter_coeffs_speech, frames_speech = lpc('res/speech.wav', "voice", algorithm)
    
    # Compute cross synthesis
    filtered_signal = crossSynth(frames_piano, filter_coeffs_piano, filter_coeffs_speech, data_piano, frames_piano)
    filtered_signal *= 2**15 - 1
    # Write filtered signal to a new file
    wavfile.write('output.wav', rate_piano, filtered_signal.astype(np.int16))

def steepest_descent_analysis() : 

    # Get the lambda factor 
    args = lpc('res/speech.wav','voice', 'closed_form')
    lambda_factor = args[6]

    # Compute the theoretical average minimum error 
    global epsilon 
    epsilon = 10**-9
    args  = lpc('res/speech.wav', "voice", 'steepest_descent')
    Jmin = args[5]

    # Results by varing mu_values 
    mu_values = [0.1,0.25,0.5,0.75,0.9]
    Javg_mu = np.zeros(len(mu_values))
    Times_mu = np.zeros(len(mu_values))
    for i,mu in enumerate(mu_values): 
        start_t = time.time()
        args = lpc('res/speech.wav', "voice", 'steepest_descent', mu)
        end_t = time.time()
        Times_mu[i] = end_t - start_t

        Javg = args[5]
        Javg_mu[i] = Javg
    plot_analysis(Javg_mu,Times_mu,mu_values,'mu',lambda_factor)

    # Results by varing epsilon 
    eps_values = [10**-2,10**-4,10**-6,10**-8]
    Javg_eps = np.zeros(len(eps_values))
    Times_eps = np.zeros(len(eps_values))
    for i,eps in enumerate(eps_values): 
        epsilon = eps
        start_t = time.time()
        args = lpc('res/speech.wav', "voice", 'steepest_descent')
        end_t = time.time()
        Times_eps[i] = end_t - start_t
        Javg = args[5]
        Javg_eps[i] = Javg
    plot_analysis(Javg_eps,Times_eps,eps_values,'eps',lambda_factor)
    pass

def plot_analysis(Javg,times,param,type,lambda_factor) : 
    fig, ax = plt.subplots()
    plt.bar(range(len(Javg)),Javg)
    # Create x-axis labels with two rows
    labels = [f'{x}\n{y}' for x, y in zip(param, times)]
    ax.set_xticks(range(len(Javg)))
    ax.set_xticklabels(labels)

    
    # Add axis labels and title
    if type == 'mu' : plt.xlabel('mu / time (s)')
    if type == 'eps' : plt.xlabel('epsilon / time (s)')
    plt.ylabel(' Error average over frames')
    plt.title('Error performance, lamda factor = '+lambda_factor)

    # Show the plot
    plt.show()



if __name__ == '__main__' : 
   steepest_descent_analysis()




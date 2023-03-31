import numpy as np
import sys
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import scipy as sp
import time

# Set constants and parameters
frame_size = 1*1024  # Size of the audio frames in samples
overlap_factor = 0.5  # Fraction of overlap between frames
window_function = np.hanning  # Window function used to weight audio frames
check_cola=True  # Check if constant overlap add (COLA) condition is satisfied
lpc_order_piano=24  # LPC order for piano sound
lpc_order_voice=48  # LPC order for voice sound
algorithm = 'closed_form'  # Algorithm for LPC computation ('closed_form' or 'steepest_descent')
mu = 0.8  # Steepest descent step size
epsilon = 10**-3  # Steepest descent convergence threshold

# Read an audio file and return the sample rate and data
def read_wav(filename):
    rate, data = wavfile.read(filename)
    # Use only one channel if stereo
    if len(data.shape) > 1:
        data = data[:, 0]
    # Normalize the data to have maximum absolute value of 1
    data = data/max(abs(data))
    return rate, data

# Divide an audio signal into frames with specified size and overlap
def divide_into_frames(signal, frame_size, overlap_factor):
    # Compute the step size between frames
    step_size = int(frame_size * (1 - overlap_factor))
    # Compute the number of frames needed to cover the signal
    num_frames = (len(signal) - frame_size) // step_size + 1
    # Initialize an array to store the frames
    frames = np.zeros((num_frames, frame_size))
    # Divide the signal into frames and store them in the array
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size : i * step_size + frame_size]
    return frames

# Compute the LPC coefficients for a frame using the closed-form solution
def compute_closed_coefficents(frame, R, r):
    return np.linalg.solve(R, r)

# Compute the LPC coefficients for a frame using the steepest descent algorithm
def compute_steepest_coefficents(frame ,R ,r, mu) :
    # Compute the energy of the frame
    sigma = np.mean(np.sum(np.square(frame)))
    # Initialize the weight vector and step size
    w = np.zeros(len(r))
    eigs = sp.linalg.eigvals(R)
    dw = mu*2/abs(max(eigs))
    # Compute the objective function
    J = sigma - np.dot(np.conj(w),r) - np.dot(np.conj(r),w) + np.dot(np.conj(w),np.dot(R,w))
    delta_J = J
    # Update the weight vector until convergence
    while(delta_J > epsilon / mu ) :
        # Compute the gradient of the objective function
        grad = r - np.dot(R,w)
        # Update the weight vector
        w = w + dw*grad
        # Compute the new objective function and check convergence
        J_prev = J
        J = sigma - np.dot(np.conj(w),r) - np.dot(np.conj(r),w) + np.dot(np.conj(w),np.dot(R,w))
        delta_J = J_prev - J
    return w , J , eigs

# Compute the Whiteing filter coefficents
def compute_whitening_filter(lpc_coeffs):
    p_order = len(lpc_coeffs)
    #the whitening fiter is 1 coefficient longer than the wiener (1 - (wiener coeff))
    whitening_filter = np.zeros(p_order+1)
    whitening_filter[0]=1
    whitening_filter[1:] = -lpc_coeffs

    return whitening_filter

# Compute the correlation vector and matrix between x , y and selects p_order coefficents 
def correlate(x, y, p_order):
    # Compute the autocorrelation
    autocorr = np.correlate(x, y, mode="full")[len(x) - 1:]
    # autocorr /= np.max(autocorr)

    # Compute the correlation matrix R and vector r
    R = np.zeros((p_order, p_order))
    r = np.zeros(p_order)
    for i in range(p_order):
        for j in range(p_order):
            R[i,j] = autocorr[np.abs(i-j)]
        r[i] = autocorr[i+1]
    
    return R, r

# Computes the wiener predition filter
# algorithm : 'closed_form' -> closed form Wiener-Hops equation solution
# algorithm : 'steepest_descent' -> iterative gradient based method
def lpc(filename, soundType, algorithm, mu = 0.5):
    # Set the order based on the signal type
    if soundType == 'piano' : 
        p_order = lpc_order_piano
    elif soundType == 'voice' : 
        p_order = lpc_order_voice
    # Read audio
    rate, data = read_wav(filename)
    # Framing and windowing
    frames = divide_into_frames(data, frame_size, overlap_factor)
    windowed_frames = frames * window_function(frame_size)
    # Initialize an array with zeros to store whiteing coefficents
    whitening_filter_coeffs = np.zeros( [len(frames) , p_order+1])
    # Initialize as array to store final error of the frames (steepest_descent)
    J_list = np.zeros(len(frames))
    lmd_factors = np.zeros(len(frames))
    # Cycle over frames
    for i , frame in enumerate(windowed_frames) : 
        if(i % 100 == 0) : print(i)
        # Compute frame correlation
        R , r = correlate(frame,frame,p_order)

        if algorithm =='steepest_descent': 
            # lpc coefficents , J Error , Rxx Eigenvalues
            lpc_coeffs, J , eigs = compute_steepest_coefficents(frame, R ,r , mu)
            J_list[i] = J
            lmd_factors[i] = max(abs(eigs))/min(abs(eigs))

        elif algorithm =='closed_form':
            lpc_coeffs = compute_closed_coefficents(frame, R ,r )
        else : 
            raise(ValueError(algorithm + ' is an invalid algorithm '))
        
        whitening_filter_coeffs[i] = compute_whitening_filter(lpc_coeffs)

    # Averaging error over frames 
    J_avg = 0
    lmd_factor_avg = 0
    if algorithm == 'steepest_descent' : 
        J_avg = np.mean(abs(J_list))
    lmd_factor_avg = np.mean(lmd_factors)

    return rate, data, lpc_coeffs, whitening_filter_coeffs, windowed_frames, J_avg , lmd_factor_avg

# Perform the cross synthesis between an Harmonic Signal and a Shaping signal 
# In our case Harmonic -> piano.wav , Shaping -> speech.wav
# The armonic Signal is whitened before shaping
def crossSynth(harmonic_signal_framed, harmonic_whitening_framed, formant_whitening_framed, original_signal, original_signal_framed):
    # Step to begin next frame into the result
    step_size = int(frame_size * (1 - overlap_factor))
    # Perform Zero Padding of factor zp 
    zp = 4
    # Initialize an array with zeros to store the filtered signal 
    result = np.zeros(len(original_signal)+zp*frame_size)
    # Cycle over frames
    for i in range(harmonic_signal_framed.shape[0]):
        # Transform in FFT
        frame_fft = np.fft.fft(harmonic_signal_framed[i], n=zp*frame_size)
        harmonic_whitening_fft = np.fft.fft(harmonic_whitening_framed[i], n=zp*frame_size)
        formant_whitening_fft = np.fft.fft(formant_whitening_framed[i], n=zp*frame_size)
        # Filtering in FFT domain
        result[(i*step_size) : i*step_size+(zp*frame_size)] += np.fft.ifft(frame_fft*harmonic_whitening_fft/formant_whitening_fft).real
    return result

# Check if the COLA condition is satisfied for a given window 
# Plot the superimposition of the shifted windows to check if constant
def plot_window_and_cola(window_function, frame_size, overlap_factor):
    # Prepare the array where to superimpose the windows
    rate, data = read_wav('res/piano.wav')
    signal_length = len(data)
    step_size = int(frame_size * (1 - overlap_factor))
    #print(step_size)
    num_steps = (signal_length - frame_size) // step_size + 1
    # Get a numpy window
    window = window_function(frame_size)
    # Initialize the array containg a shifted window
    shifted_window = np.zeros((signal_length))
    # Initialize the array with the overall sum 
    cola_sum = np.zeros_like(shifted_window)
    # Cycle over the shifted windows
    for i in range(num_steps):
        # Shift the window
        shifted_window[i * step_size : i * step_size + frame_size] = window
        # add to the sum
        cola_sum += shifted_window
        # Clear the window array
        shifted_window = np.zeros((signal_length))

    # Plot the Results
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

# Plot an audio frame next to a given filter 
# default filter_type is a whiteing , use 'shaping' to get the reciprocal filter
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

# Check Cola, Perform LPC and CrossSynthesis
def perform_CrossSynth():
    # Check COLA Condition
    if(check_cola):
        plot_window_and_cola(window_function, frame_size, overlap_factor)
    # Compute LPC coefficients and whitening filters
    rate_piano, data_piano, lpc_coeffs_piano, filter_coeffs_piano, frames_piano, J_avg , lmd_factor_avg = lpc('res/piano.wav', "piano", algorithm, mu)
    rate_speech, data_speech, lpc_coeffs_speech, filter_coeffs_speech, frames_speech, J_avg , lmd_factor_avg  = lpc('res/speech.wav', "voice", algorithm, mu)
    
    # Compute cross synthesis
    filtered_signal = crossSynth(frames_piano, filter_coeffs_piano, filter_coeffs_speech, data_piano, frames_piano)
    filtered_signal *= 2**15 - 1
    # Write filtered signal to a new file
    wavfile.write('output.wav', rate_piano, filtered_signal.astype(np.int16))

# Analyze Error by varing mu, epsilon in Steepest Descent algorithm
def steepest_descent_analysis() :  
    # Compute the theoretical average minimum error 
    global epsilon 
    epsilon = 10**-7
    # args  = lpc('res/speech.wav', "voice", 'steepest_descent')
    # Jmin = args[5]
    Jmin = 0.010022139713234417 # evaluated for mu = 0.5, eps = 10e-7
    #lambda_factor = args[6]
    epsilon = 10**-5

    # Results by varing mu_values 
    mu_values = [0.25,0.3,0.5,0.75,0.9]
    Javg_mu = np.zeros(len(mu_values))
    Times_mu = np.zeros(len(mu_values))
    for i,mu in enumerate(mu_values): 
        start_t = time.time()
        args = lpc('res/speech.wav', "voice", 'steepest_descent', mu)
        end_t = time.time()
        Times_mu[i] = end_t - start_t

        Javg = args[5]
        Javg_mu[i] = Javg
        Javg_mu /= Jmin
    plot_analysis(Javg_mu,Times_mu,mu_values,'mu')

    # Results by varing epsilon 
    eps_values = [10**-2,10**-4,10**-5,10**-6]
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
        Javg_eps /= Jmin
    plot_analysis(Javg_eps,Times_eps,eps_values,'eps')
    pass

def plot_analysis(Javg,times,param,type) : 
    fig, ax = plt.subplots()
    plt.bar(range(len(Javg)),Javg)
    plt.yscale('log')
    # Create x-axis labels with two rows
    labels = [f'{x}\n{y:.2f}' for x, y in zip(param, times)]
    ax.set_xticks(range(len(Javg)))
    ax.set_xticklabels(labels)
    # Add axis labels and title
    if type == 'mu' : plt.xlabel('mu / time (s)')
    if type == 'eps' : plt.xlabel('epsilon / time (s)')
    plt.ylabel('J(w)/Jmin avg over frames ')
    # Save the plot 
    if type == 'mu' :
        plt.title('Error over mu (eps = 10e-5)')
        plt.xlabel('mu / time (s)') 
        plt.savefig('mu_plot.pdf', dpi = 300)
    if type == 'eps' : 
        plt.title('Error over epsilon (mu = 0.5)')
        plt.xlabel('eps / time (s)') 
        plt.savefig('eps_plot.pdf', dpi = 300)
    # Show the plot
    plt.show()


if __name__ == '__main__' : 
   perform_CrossSynth()




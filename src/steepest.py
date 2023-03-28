import wave
import numpy as np
import matplotlib.pyplot as plt

overlap_factor = 0.5
frame_size = 1024
window = np.hanning(frame_size)
# Open the audio file named 'example.wav'

def audio_read(file) : 
    with wave.open(file, 'rb') as wav_file:
        # Get the number of frames in the audio file
        num_frames = wav_file.getnframes()
        # Read all the frames in the audio file as bytes
        raw_audio = wav_file.readframes(num_frames)
    samples =  np.frombuffer(raw_audio, dtype=np.int16)

    max_amp = np.max(np.abs(samples))
    # normalize the audio
    if max_amp > 1:
        samples = samples / max_amp
    # Interpret the bytes as a 16-bit integer array
    return samples

def divide_into_frames(signal, frame_size, overlap_factor):
    step_size = int(frame_size * (1 - overlap_factor))
    num_frames = (len(signal) - frame_size) // step_size + 1
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size : i * step_size + frame_size]
    return frames

def apply_window(frames, window_function):
    return frames * window_function

def compute_correlation(x,y): 
    r = np.correlate(x,y,'full')
    var = np.var(x)
    N = len(x)
    R = np.zeros((N, N))
    for i in range(N):
        R[i,:] = r[N-1-i:2*N-1-i]
    R = R / var / N
    return r, Rs

def rec_find_weights(w,r,R,frame,mu,epsilon) : 
    w_next = w + mu*(r + R*w)
    J = np.mean(sum(frame^2)) - w*r - r*w + w*(R*w)
    if J < epsilon: 
        return w_next
    else : 
        return rec_find_weights(w_next,r,R,frame,mu,epsilon)

def steepest(frames , init) : 
    for frame in frames: 
        w  = init
        mu = 0.1
        epsilon = 10^-3
        r, R = compute_correlation(frame,frame)
        w = rec_find_weights(w,r,R,frame,mu,epsilon)
        plt.imshow(R, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':

    samples = audio_read('res/piano.wav')
    frames = divide_into_frames(samples, frame_size, overlap_factor)
    w_frames = apply_window(frames, window)
    
    predictors = steepest(w_frames)
    plt.plot(w_frames[0])
    plt.show()
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

overlap_factor = 0.5
frame_size = 1024
p_order = 512
window = np.hanning(frame_size)
# Open the audio file named 'example.wav'

def audio_read(file) : 
    with wave.open(file, 'rb') as wav_file:
        # Get the number of frames in the audio file
        num_frames = wav_file.getnframes()
        # Read all the frames in the audio file as bytes
        raw_audio = wav_file.readframes(num_frames)
    return np.frombuffer(raw_audio, dtype=np.int16)
     
def divide_into_frames(signal, frame_size, overlap_factor):
    step_size = int(frame_size * (1 - overlap_factor))
    num_frames = (len(signal) - frame_size) // step_size + 1
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size : i * step_size + frame_size]
    return frames

def apply_window(frames, window_function):
    return frames * window_function

def normalize(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_norm = (x - x_mean) / x_std
    return x_norm

def compute_autocorrelation(x,y):
    autocorr = sp.signal.correlate(x, y, method="fft")[frame_size - 1:]
    R = sp.linalg.toeplitz(autocorr[0:p_order])
    r = autocorr[1:p_order+1]
    return r , R

def compute_correlation(x,y): 
    r = np.correlate(x,y,'full')
    var = np.var(x)
    N = len(x)
    R = np.zeros((N, N))
    for i in range(N):
        R[i,:] = r[N-1-i:2*N-1-i]
    return  r[:p_order] , R[:p_order,:p_order]

def rec_find_weights(w,r,R,frame,mu,iter) : 
    if iter > 10: 
        return w
    else : 
        iter +=1
        grad = R@w - r
        w -= mu*grad
        return rec_find_weights(w,r,R,frame,mu,iter)

def steepest(frames , init) : 
    for idx,frame in enumerate(frames): 
        mu = 0.001
        #frame = frame/32767.0
        r, R = compute_autocorrelation(frame,frame)
        plt.imshow(R,'Blues_r')
        plt.colorbar()
        plt.show()
        w = rec_find_weights(init,r,R,frame,mu,0)
        w_whitening = np.zeros(len(w)+1)
        w_whitening[0] = 1
        w_whitening[1:] = -w

        W = np.fft.fft(w_whitening)
        Wf = np.fft.fftfreq(len(W))[:len(W)//2]
        plt.clf()
        plt.semilogx( Wf ,2.0/len(W) * np.abs(W[:len(W)//2]) )
        #plt.plot(Wf,2.0/len(W) * np.abs(W[:len(W)//2]))
        plt.show()


if __name__ == '__main__':
    samples = audio_read('res/piano.wav')
    frames = divide_into_frames(samples, frame_size, overlap_factor)
    w_frames = apply_window(frames, window)
    w_init = np.zeros(p_order)
    predictors = steepest(w_frames,w_init)
    plt.plot(w_frames[0])
    plt.show()
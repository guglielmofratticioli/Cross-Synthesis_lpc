import numpy as np
import matplotlib.pyplot as plt

def get_overlapped_frames(samples, overlap, window ): 
# Get array lenghts 
    M = window.size
    N = len(samples)
    n_frames =int(np.ceil(N / (M - overlap)))
# Split frames into array 
    frames = [ np.zeros(M) for i in range(n_frames)]
    for i in range(n_frames):
        start = i*(M-overlap)
        end = np.clip(start + M ,0,N)
# Frame Windowing
        win_frame = np.zeros(M)
        win_frame[:(end-start)] += samples[start:end] * window[:(end-start)] 
        frames[i] = win_frame
    return frames
    
def overlapp_add_frames(frames, overlap, window=1):
    if window is not 1 :  M = window.size
    else : M = len(frames[0])
    n_frames = len(frames)
# initialize out buffer
    out = np.zeros((M-overlap)*(n_frames))
    for i ,frame in enumerate(frames) : 
        start = i*(M - overlap)
        end = np.clip(start+M,0,len(out))
        out[start:end] += frame[:(end-start)]*window
    return out
        
def check_COLA(window, R) :
    sums = []
    for n in range(window.size//2):
        sum = 0
        for m in range(-window.size//R//2, window.size//R//2):
            sum += window[n+m*R]
        sums.append(sum)
# Check if sums are constant
    if np.allclose(sums,sums[0]):
        return sums[0]
    else :
        return False 

if __name__ == '__main__' : 
    N = 101
    waveform = np.bartlett(N)
    signal =  np.tile(waveform,10)
    plt.figure()
    plt.plot(signal)

    window = np.bartlett(32)
    print(check_COLA(window,1))
    frames = get_overlapped_frames(signal,16,window)
    signal2 = overlapp_add_frames(frames,16)
    plt.figure()
    plt.plot(signal2)
    plt.show()

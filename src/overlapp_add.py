import numpy as np
# how can i get the lenght of a numpy array
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
        # frames[ i*M : (i+1)*M] = win_frame
        frames[i] = win_frame
    return frames
    
def overlapp_add_frames(frames, R, window):
    M = window.size
    N = len(frames)
    overlap = (1-R)*M
    out = np.zeros((M-overlap)*N)
    for i ,frame in enumerate(frames) : 
        start = i*(M - overlap)
        end = start+M
        out[start:end] += frame*window

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
    w = np.bartlett(100)
    check_COLA(w,50)

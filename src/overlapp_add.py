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
    for i in range(window.size-R):
        samples = window[i::R]
        if sum(samples) > 1 : 
            return False 
    return True



import wave
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    samples = audio_read('res/piano.wav')

    plt.plot(samples)
    plt.show()
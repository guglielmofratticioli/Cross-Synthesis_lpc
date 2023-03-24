import numpy as np
import wave 
import matplotlib.pyplot as plt
import overlapp_add as ola

# Open the WAV file
wav_file = wave.open('res/speech.wav', 'r')

# Get the number of frames, channels, and sample rate
num_frames = wav_file.getnframes()
num_channels = wav_file.getnchannels()
sample_rate = wav_file.getframerate()

# Read the frames into a byte string
wav_data = wav_file.readframes(num_frames)

# Convert the byte string to a NumPy array
wav_samples = np.frombuffer(wav_data, dtype=np.int16)

buffSize = 512
window = np.ones(buffSize)
frames = ola.get_overlapped_frames(wav_samples,50,window)

plt.plot(range(len(frames[0:1000])),frames[0:1000])

plt.show()
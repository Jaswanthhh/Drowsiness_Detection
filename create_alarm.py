import numpy as np
from scipy.io import wavfile

# Generate a simple alarm sound
duration = 1.0  # seconds
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration))

# Create a mix of frequencies for an alarm-like sound
frequency1 = 440.0  # Hz
frequency2 = 880.0  # Hz
waveform1 = np.sin(2 * np.pi * frequency1 * t)
waveform2 = np.sin(2 * np.pi * frequency2 * t)
waveform = (waveform1 + waveform2) / 2

# Normalize and convert to 16-bit integer
waveform = np.int16(waveform * 32767)

# Save the file
wavfile.write('alarm.wav', sample_rate, waveform) 
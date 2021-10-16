# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
"""
# Understanding Audio data

# Short Time Fourier Transform

Take small time sections of the input Amplitude waveform in time domain
and perform FFT to obtain a freq vs time spectrogram.

Repeat unitil the end of the input set is reached.

# We take spectrogram as the input layer for the DL model

# Mel Frequency Cepstral Coefficients (MFCCs)

They capture the timbral/textural aspects of the sound.
Frequency domain feature.
Approximate the human auditory system.
13-40 Coefficients.
Calculated at each frame.
Used for Speech Recognition, Music Genre classification and Musical Instrument
classification
"""
# %% [markdown]
"""
Importing necessary libraries
"""
# %%
# data from
# https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
file = "../data/genres_original/blues/blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # sr*T -> 22050*30
# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# %%
# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency) / 2)]
left_magnitude = magnitude[:int(len(magnitude) / 2)]

# %%
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# %%
# stft -> spectrogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# %%
# MFCCs
MFCCs = librosa.feature.mfcc(signal,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()

# %%

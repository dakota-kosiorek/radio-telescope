import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

sample_rate = 2.4e6

# RTL-SDR Settings
sdr = RtlSdr()
sdr.sample_rate = sample_rate  # 2.4 MHz sample rate
#sdr.center_freq = 1420.4e6  # Hydrogen Line frequency (1.42 GHz)
sdr.center_freq = 89.7e6  # WITR (89.7 MHz)
sdr.freq_correction = 25  # PPM
sdr.gain = 30  # Adjust for best SNR
sdr.set_bias_tee(True)

fft_size = 2048
num_rows = 512

x = sdr.read_samples(2048) # get rid of initial empty samples

spectrogram = np.zeros((num_rows, fft_size))

extent = [(sdr.center_freq + sdr.sample_rate/-2)/1e6,
            (sdr.center_freq + sdr.sample_rate/2)/1e6,
            len(spectrogram)/sdr.sample_rate, 0]

print('Collecting samples...')
x = sdr.read_samples(fft_size*num_rows)

sdr.close()

print('Processing samples...')
for j in range(num_rows):
    spectrogram[j,:] = 10*np.log10(
        np.abs(
            np.fft.fftshift(
                np.fft.fft(x[j*fft_size:(j+1)*fft_size])
            )
        )**2
    )


print('Displaying data...')
plt.figure(figsize=(10, 6))
plt.plot(
    np.linspace(extent[0], extent[1], fft_size),
    np.sum(spectrogram, axis=0) * 1/len(spectrogram)
)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power [dB]")
plt.title("Spectrogram")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(spectrogram, aspect='auto', extent=extent)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.title("Spectrogram")
plt.colorbar(label="Power [dB]")
plt.show()
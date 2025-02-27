import numpy as np
import matplotlib.pyplot as plt
import subprocess

seconds_to_record = 2.5

sample_rate = 2.4e6 # In Hz
#center_freq = 1420.4e6 # Hydrogen line
center_freq = 89.7e6 # WITR
gain = 30 # In dB
output_file = './tmp/recording.bin'

# NOTE: With a FFT of this size, 1 minutes worth of data is ~280 Mb
fft_size = 2048
one_sec_data = round(1/fft_size * sample_rate) # How many samples needed for 1 second of data
num_rows = int(np.floor(one_sec_data * seconds_to_record))
num_samples = fft_size * num_rows # Total number of samples
spectrogram = np.zeros((num_rows, fft_size))

print(f'Collecting {round((num_samples) * 1/sample_rate,2)}s of data...' )

# Turn on bias tee
result = subprocess.run(
    ['rtl_biast', '-b 1'], 
    capture_output=True, text=True, check=True
)

# Get samples
result = subprocess.run([
    'rtl_sdr',
    f'-s {sample_rate}',
    f'-f {center_freq}',
    f'-g {gain}',
    f'-n {num_samples}',
    f'{output_file}'],
    capture_output=True, text=True, check=True
)

print(f'Reading from file "{output_file}"...')
iq_samples_raw = np.fromfile(output_file, np.uint8) # Read in file.

print(f'Processing sample data...')
if len(iq_samples_raw) % 2 != 0:
    iq_samples_raw = iq_samples_raw[:len(iq_samples_raw) - 1]

# Initialize array
iq_samples = np.zeros(int(len(iq_samples_raw) / 2), dtype=np.complex64)

######################################
###### TODO: MULTI THREAD THIS #######
######################################

for i, j in enumerate(range(0, len(iq_samples_raw), 2)):
    real = (float(iq_samples_raw[j]) - 127.5) / 127.5
    imag = (float(iq_samples_raw[j + 1]) - 127.5) / 127.5

    iq_samples[i] = complex(real, imag)

######################################
######################################
######################################

iq_samples = np.array(iq_samples)
iq_samples = iq_samples[2048: ] # Ignore grabage initial samples

num_rows = int(np.floor(1/fft_size * len(iq_samples)))

print('Running FFT...')
######################################
###### TODO: MULTI THREAD THIS #######
######################################
for i in range(num_rows):
    sgement = iq_samples[i * fft_size:(i+1) * fft_size]
    spectrogram[i: ] = 10*np.log10(
        np.abs(
            np.fft.fftshift(
                np.fft.fft(sgement)
            )
        )**2
    )

    
######################################
######################################
######################################

extent = [(center_freq + sample_rate/-2)/1e6,
            (center_freq + sample_rate/2)/1e6,
            seconds_to_record, 0]

print('Graphing...')

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
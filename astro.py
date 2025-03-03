import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import concurrent.futures

# usbipd attach --wsl --busid [id]

def process_fft(data):
    """
    Run a FFT, FFT shift, and scale to dB for an array of complex values
    
    `data`: An array of np.complex64 values.
    """
    
    fft = np.fft.fft(data)
    shift = np.fft.fftshift(fft)
    abs_vals = np.abs(shift)**2
    db_vals = 10 * np.log10(abs_vals)

    return db_vals

def get_iq_raw(sr: int, cf: int, g: float, ns: int, output: str):
    """
    Uses the `rtl_sdr` command line tool to collect IQ data in a binary file.

    `sr`: Sampling rate.

    `cf`: Center frequency.

    `g`: gain.

    `ns`: Number of samples.

    `output`: Binary file output path.
    """

    # Turn on bias tee
    result = subprocess.run(
        ['rtl_biast', '-b 1'], 
        capture_output=True, text=True, check=True
    )

    # Get samples
    result = subprocess.run([
        'rtl_sdr',
        f'-s {sr}',
        f'-f {cf}',
        f'-g {g}',
        f'-n {ns}',
        f'{output}'],
        capture_output=True, text=True, check=True
    )

def process_iq_samples(riq, piq):
    """
    Process IQ samples from the `uint8` encoded binary file to a complex python object.

    `riq`: Raw IQ sample array. Each Sample is two entires of this array encoded as `unit8`.

    `piq`: Processed IQ sample array this function writes to. It should be half the size of the raw IQ sample array.
    """

    real = (riq[::2] - 127.5) / 127.5
    imag = (riq[1::2] - 127.5) / 127.5

    piq[:] = real + 1J * imag

def main():
    #########################
    ### Sample Collection ###
    #########################

    seconds_to_record = 8 # How many seconds worth of data the RTL SDR should record

    sample_rate = 2.4e6 # In Hz
    #center_freq = 1420.4e6 # Hydrogen line
    center_freq = 89.7e6 # WITR
    gain = 30 # In dB
    output_file = './tmp/recording.bin' # Where the IQ data recorded is being stored

    # NOTE: With a FFT of this size, 1 minutes worth of data is ~280 Mb
    fft_size = 2048
    one_sec_data = round(1/fft_size * sample_rate) # How many samples needed for 1 second of data
    num_samples = fft_size * int(np.floor(one_sec_data * seconds_to_record)) # Total number of samples

    print(f'Collecting {round((num_samples) * 1/sample_rate,2)}s of data...' )
    get_iq_raw(sample_rate, center_freq, gain, num_samples, output_file)

    #########################
    ### Sample Processing ###
    #########################

    print(f'Reading from file "{output_file}"...')
    iq_samples_raw = np.fromfile(output_file, np.uint8).astype(float) # Read in file.

    process_sample_time = time.time()
    print(f'Processing sample data...')

    # If (for whatever reason) there are is an odd number in the raw IQ array (# of I != # of Q), 
    # remove the last uint8 value to get everything matching
    if len(iq_samples_raw) % 2 != 0:
        iq_samples_raw = iq_samples_raw[:len(iq_samples_raw) - 1]

    # Initialize array
    iq_samples = np.zeros(int(len(iq_samples_raw) / 2), dtype=np.complex64)
    process_iq_samples(iq_samples_raw, iq_samples) # Turn seperate uint8 pairs of samples into complex objects
    iq_samples = iq_samples[2048: ] # Ignore grabage initial samples

    process_sample_time = time.time() - process_sample_time
    print(f'Sample data processing: {round(process_sample_time, 2)}s')

    #########################
    ########## FFT ##########
    #########################

    print('Running FFT...')
    fft_time = time.time()

    num_rows = int(np.floor(1/fft_size * len(iq_samples)))
    split_iq = np.reshape(iq_samples[:num_rows * fft_size], (num_rows, fft_size))

    result = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as exe:
        # Map each row to the processing function
        result = list(exe.map(process_fft, split_iq))

    # Convert the result back to a 2D array
    spectrogram = np.array(result)

    fft_time = time.time() - fft_time
    print(f'FFT time: {round(fft_time, 2)}s')

    extent = [(center_freq + sample_rate/-2)/1e6,
                (center_freq + sample_rate/2)/1e6,
                seconds_to_record, 0]

    #########################
    ####### Graphing ########
    #########################

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

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt


# assumed duration of one heartbeat, in seconds
HEARTBEAT_PERIOD = 1.0


def _fourier_1D(array: np.array, timestep: float, freq_cut: int, over_sampling=False, period=HEARTBEAT_PERIOD):
    """
    Compute the Fourier series of a 1D array and reconstruct the signal
    cutting the highest freq_cut frequencies.
    """

    # create a linear interpolation between each point in the array, to have
    # over sampling
    if over_sampling:
        array_fitted = []
        for i in range(len(array)-1):
            for t in np.linspace(0, 1, 20, endpoint=False):
                array_fitted.append((1-t)*array[i] + t*array[i+1])
        array_fitted.append(array[-1])

        array = np.array(array_fitted)

    M = len(array)

    freq_to_keep = M/2 - freq_cut
    dt = period / M
    # gives frequencies in order matching np.fft.fft
    freqs = np.fft.fftfreq(M, d=dt)

    Xk = np.fft.fft(array) / (M)  # normalize so sum matches original signal

    # frequencies to keep
    mask = np.abs(freqs) <= freq_to_keep
    Xk = Xk * mask

    T = int(period/timestep)
    t = np.linspace(0, period, T, endpoint=False)
    
    result = np.zeros_like(t, dtype=complex)
    for k in range(M):
        result += Xk[k] * np.exp(2j * np.pi * freqs[k] * t)

    result = np.real(result)

    return result

import numpy as np

def fourier_1D(array: np.ndarray, timestep: float, freq_cut: int, period=HEARTBEAT_PERIOD):
    """
    Compute the Fourier series of a 1D array and reconstruct the signal
    on a new timestep grid, cutting the highest `freq_cut` frequency bins.
    
    Note: Manual linear oversampling has been removed because the continuous
    reconstruction grid naturally provides perfect trigonometric oversampling.
    """
    M = len(array)
    
    # 1. Correct sampling interval and frequency resolution (df)
    dt = period / M
    df = 1.0 / period
    
    # 2. Compute frequencies and normalized FFT coefficients
    freqs = np.fft.fftfreq(M, d=dt)
    Xk = np.fft.fft(array) / M  

    # 3. Corrected Cutoff Logic: Convert the bin target into physical Hz
    # The highest frequency index sits at M // 2. 
    # Subtracting freq_cut bins and multiplying by df gives the true Hz threshold.
    freq_to_keep = (M // 2 - freq_cut) * df
    
    # Apply mask to eliminate the highest frequencies
    mask = np.abs(freqs) <= freq_to_keep
    Xk = Xk * mask

    # 4. Define the target reconstruction time grid
    T = int(period / timestep)
    t = np.linspace(0, period, T, endpoint=False)
    
    # 5. Blazing fast vectorized Fourier reconstruction 
    # freqs[:, None] shapes freqs to (M, 1) and t is (T,), broadcasting to (M, T)
    reconstruction_matrix = Xk[:, None] * np.exp(2j * np.pi * freqs[:, None] * t)
    result = np.real(np.sum(reconstruction_matrix, axis=0))

    return result


def fourier_3D(array: np.array, timestep: float, freq_cut: int, period=HEARTBEAT_PERIOD):
    """
    Compute the Fourier series of a 3D array and reconstruct the signal
    """

    # extract the x,y,z coordinates from the array
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]

    cleaned_signal_x = fourier_1D(x, timestep=timestep, freq_cut=freq_cut, period=period)
    cleaned_signal_y = fourier_1D(y, timestep=timestep, freq_cut=freq_cut, period=period)
    cleaned_signal_z = fourier_1D(z, timestep=timestep, freq_cut=freq_cut, period=period)

    cleaned_signal = np.stack([cleaned_signal_x,
                               cleaned_signal_y,
                               cleaned_signal_z], axis=1)

    return cleaned_signal

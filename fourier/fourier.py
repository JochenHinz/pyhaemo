import numpy as np
import matplotlib.pyplot as plt



def fourier_1D(array: np.array, freq_cut: int = 0, over_sampling=False):
    """
    Compute the Fourier series of a 1D array and reconstruct the signal
    cutting the frequencies higher than freq_cut.
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
    dt = 1 / M
    freqs = np.fft.fftfreq(M, d=dt)  # gives frequencies in order matching np.fft.fft

    Xk = np.fft.fft(array) / M  # normalize so sum matches original signal

    # frequencies to keep
    mask = np.abs(freqs) <= freq_to_keep
    Xk = Xk * mask

    # If I use M > 22, I get a wrong reconstruction
    t = np.linspace(0, 1, 8000, endpoint=False)

    result = np.zeros_like(t, dtype=complex)
    for k in range(M):
        result += Xk[k] * np.exp(2j * np.pi * freqs[k] * t)

    result = np.real(result)

    return result


def fourier_3D(array: np.array, freq_cut: int = 0):
    """
    Compute the Fourier series of a 3D array and reconstruct the signal
    """

    # extract the x,y,z coordinates from the array
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]

    cleaned_signal_x = fourier_1D(x, freq_cut)
    cleaned_signal_y = fourier_1D(y, freq_cut)
    cleaned_signal_z = fourier_1D(z, freq_cut)

    cleaned_signal = np.stack([cleaned_signal_x,
                               cleaned_signal_y,
                               cleaned_signal_z], axis=1)

    return cleaned_signal


def plot_fourier(original_signal: list,
                 cleaned_signals: list,
                 freqs_cut: list[int]):
    """
    Plot original and cleaned signals after Fourier transformation.
    """

    for cleaned_signal, freq_cut in zip(cleaned_signals, freqs_cut):
        plt.plot(np.linspace(0,1,cleaned_signal.shape[0], endpoint=False),
                                 cleaned_signal, label=f'cleaned cut{freq_cut} freqs',
                                  linewidth=5)


    plt.plot(np.linspace(0,1,original_signal.shape[0], endpoint=False),
                                original_signal, label='original',
                                linewidth=5)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Time', fontsize=28)
    plt.ylabel('Signal', fontsize=28)
    plt.legend(fontsize=28)

    plt.show()

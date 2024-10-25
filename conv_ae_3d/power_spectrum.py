import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


def compute_1d_power_spectrum(arr: np.ndarray, P: float):
    """Compute the 1D power spectrum of a 1D array, given the period of the array.
    """
    N = len(arr)
    dt = P / N
    freqs = np.fft.rfftfreq(N, d=dt)
    delta_f = 1 / P  # Frequency resolution (= fs / N, where fs = 1 / dt) = gap between frequency bins

    arr = arr - np.mean(arr)
    fft = np.fft.rfft(arr) / N  # Normalise by length of signal
    power_spectrum = fft * np.conj(fft) / delta_f  # Normalise by delta_f

    return freqs, power_spectrum


def compute_3d_power_spectrum(density_patch: np.ndarray, P: float):
    """Compute the 3D power spectrum of a 3D array, given the length of the array.

    If P is the length of each side [m] of the 3D array, then the wavenumber resolution is 1 / P [m^-1].
    """
    assert density_patch.ndim == 3
    assert density_patch.shape[0] == density_patch.shape[1] == density_patch.shape[2]

    N = density_patch.shape[0]
    delta_k = 1 / P # Wavenumber resolution (= 1 / dx) = gap between wavenumber bins

    density_patch = density_patch - np.mean(density_patch)
    fft = np.fft.rfftn(density_patch) / N**3  # Normalise by size of signal
    power_spectrum_3d = np.abs(fft)**2 / delta_k**3  # Normalise by wavenumber bin volume

    #plt.imshow(np.log(power_spectrum_3d[:, :, N // 2]))
    #plt.colorbar()
    #plt.show()
    #assert 0

    # Compute the radial distance from each point in the power spectrum to the center
    #power_spectrum_3d = np.fft.fftshift(power_spectrum_3d)  # Shift the zero frequency component to the center, so that wavenumbers start from -k/2 and end at k/2
    #kx = np.arange(-density_patch.shape[0] // 2, density_patch.shape[0] // 2)
    #ky = np.arange(-density_patch.shape[1] // 2, density_patch.shape[1] // 2)
    #kz = np.arange(-density_patch.shape[2] // 2, density_patch.shape[2] // 2)
    #k = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)

    # Compute the wavenumbers using np.fft.fftfreq
    kx = np.fft.fftfreq(N, d=delta_k)
    ky = np.fft.fftfreq(N, d=delta_k)
    kz = np.fft.rfftfreq(N, d=delta_k)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    k_rounded = np.round(k).astype(int)
    index = np.arange(N // 2 + 1)
    wavenumbers = index * delta_k  # Wavenumber [m^-1] of each index

    # Compute the 1D power spectrum by radially averaging the 3D power spectrum
    power_spectrum_1d = ndimage.mean(power_spectrum_3d, labels=k_rounded, index=index)

    return wavenumbers[1:], power_spectrum_1d[1:]

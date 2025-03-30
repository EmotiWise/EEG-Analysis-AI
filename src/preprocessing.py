import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_lowpass(cutoff, fs, order=5):
    """
    Calcula los coeficientes para un filtro Butterworth pasa-bajos.

    Parámetros:
    -----------
    cutoff : float
        Frecuencia de corte del filtro en Hz.
    fs : float
        Frecuencia de muestreo de la señal en Hz.
    order : int
        Orden del filtro.

    Retorna:
    --------
    b, a : array_like
        Coeficientes numerador (b) y denominador (a) del filtro IIR.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    """
    Aplica un filtro pasa-bajos Butterworth a una señal.

    Parámetros:
    -----------
    data : array_like
        Señal de entrada (array 1D).
    cutoff : float
        Frecuencia de corte en Hz.
    fs : float
        Frecuencia de muestreo en Hz.
    order : int
        Orden del filtro.

    Retorna:
    --------
    y : ndarray
        Señal filtrada.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Aplica un filtro notch para eliminar una frecuencia específica (p. ej., 50 o 60 Hz) de la señal.

    Parámetros:
    -----------
    data : array_like
        Señal de entrada (array 1D).
    notch_freq : float
        Frecuencia a eliminar (ej. 50 o 60 Hz).
    fs : float
        Frecuencia de muestreo en Hz.
    quality_factor : float
        Factor de calidad del filtro notch (Q). Un Q mayor implica una banda de rechazo más estrecha.

    Retorna:
    --------
    y : ndarray
        Señal filtrada.
    """
    nyq = 0.5 * fs
    norm_notch = notch_freq / nyq
    b, a = iirnotch(norm_notch, quality_factor)
    y = filtfilt(b, a, data)
    return y

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica un filtro pasa-banda Butterworth a la señal.

    Parámetros:
    -----------
    data : array_like
        Señal de entrada (array 1D).
    lowcut : float
        Frecuencia de corte inferior en Hz.
    highcut : float
        Frecuencia de corte superior en Hz.
    fs : float
        Frecuencia de muestreo en Hz.
    order : int
        Orden del filtro.

    Retorna:
    --------
    y : ndarray
        Señal filtrada.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

if __name__ == "__main__":
    # Ejemplo de uso del módulo
    import matplotlib.pyplot as plt

    # Parámetros de ejemplo
    fs = 128  # Frecuencia de muestreo en Hz
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    # Señal de prueba: suma de una señal lenta (1 Hz) y ruido de red (60 Hz)
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

    # Aplicar filtro pasa-bajos para atenuar el ruido de 60 Hz
    lowpass_cutoff = 40  # Hz
    filtered_signal_lp = apply_lowpass_filter(signal, lowpass_cutoff, fs, order=4)

    # Aplicar filtro notch para eliminar 60 Hz
    notch_freq = 60  # Hz
    filtered_signal_notch = apply_notch_filter(signal, notch_freq, fs, quality_factor=30)

    # Aplicar filtro pasa-banda (por ejemplo, para extraer la banda alfa: 8-13 Hz)
    bandpass_lowcut = 8
    bandpass_highcut = 13
    filtered_signal_bp = apply_bandpass_filter(signal, bandpass_lowcut, bandpass_highcut, fs, order=4)

    # Graficar las señales para comparar
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label='Señal original')
    plt.title('Señal Original')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(t, filtered_signal_lp, label=f'Filtro Pasa-bajos (corte={lowpass_cutoff} Hz)', color='green')
    plt.title('Señal Filtrada (Pasa-bajos)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(t, filtered_signal_notch, label=f'Filtro Notch (elimina {notch_freq} Hz)', color='red')
    plt.title('Señal Filtrada (Notch)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(t, filtered_signal_bp, label=f'Filtro Pasa-banda ({bandpass_lowcut}-{bandpass_highcut} Hz)', color='blue')
    plt.title('Señal Filtrada (Pasa-banda)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

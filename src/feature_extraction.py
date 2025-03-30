import numpy as np
from scipy.signal import welch

def bandpower(data, fs, band, nperseg=256):
    """
    Calcula la potencia integrada de la señal en una banda de frecuencia específica usando el método de Welch.
    
    Parámetros:
    -----------
    data : array_like (1D)
        La señal de entrada (por ejemplo, un segmento de EEG).
    fs : float
        Frecuencia de muestreo en Hz.
    band : tuple (f_low, f_high)
        Rango de la banda de interés en Hz.
    nperseg : int, opcional (default=256)
        Número de muestras por segmento para el cálculo de Welch.
    
    Retorna:
    --------
    float
        Potencia integrada en la banda (en unidades de la señal al cuadrado, por ejemplo, µV² si la señal está en µV).
    """
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    freq_idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[freq_idx], f[freq_idx])

def extract_features(segment, fs, bands=None, nperseg=256):
    """
    Extrae características simples de un segmento de señal EEG.
    
    Se calcula la potencia (energía) en cada banda de frecuencia definida.
    
    Parámetros:
    -----------
    segment : array_like (1D)
        Segmento de la señal EEG.
    fs : float
        Frecuencia de muestreo en Hz.
    bands : dict, opcional
        Diccionario con claves como nombre de banda y valores como tuplas (f_low, f_high).
        Si no se especifica, se usan las siguientes bandas por defecto:
            - 'delta': (0.5, 4)
            - 'theta': (4, 8)
            - 'alpha': (8, 13)
            - 'beta': (13, 30)
    nperseg : int, opcional (default=256)
        Número de muestras por segmento para el cálculo de Welch.
        
    Retorna:
    --------
    dict
        Diccionario con la potencia integrada en cada banda.
    """
    # Definir bandas por defecto si no se proporcionan
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
    
    features = {}
    for band_name, band_range in bands.items():
        features[band_name] = bandpower(segment, fs, band_range, nperseg=nperseg)
    return features

def extract_features_from_segments(segments, fs, bands=None, nperseg=256):
    """
    Extrae características para múltiples segmentos de señal EEG.
    
    Parámetros:
    -----------
    segments : array_like (2D)
        Matriz donde cada fila corresponde a un segmento de señal EEG.
    fs : float
        Frecuencia de muestreo en Hz.
    bands : dict, opcional
        Diccionario de bandas de frecuencia. Ver extract_features.
    nperseg : int, opcional (default=256)
        Número de muestras por segmento para el cálculo de Welch.
        
    Retorna:
    --------
    list of dict
        Lista en la que cada elemento es un diccionario con las características extraídas de un segmento.
    """
    features_list = []
    for segment in segments:
        features = extract_features(segment, fs, bands=bands, nperseg=nperseg)
        features_list.append(features)
    return features_list

if __name__ == "__main__":
    # Ejemplo de uso: Generar una señal de prueba y extraer características
    import matplotlib.pyplot as plt

    # Parámetros de ejemplo
    fs = 128  # Frecuencia de muestreo en Hz
    duration = 2  # Duración en segundos
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    # Señal de prueba: Sinusoide de 10 Hz + ruido blanco
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

    # Graficar la señal de prueba
    plt.figure(figsize=(10, 4))
    plt.plot(t, test_signal, label="Señal de prueba")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title("Señal de prueba para extracción de características")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Extraer características del segmento completo
    features = extract_features(test_signal, fs)
    print("Características extraídas:")
    for band, power in features.items():
        print(f"Potencia en {band}: {power:.4f}")

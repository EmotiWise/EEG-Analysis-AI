import numpy as np
import pandas as pd
from scipy.signal import coherence
import matplotlib.pyplot as plt
import seaborn as sns

def compute_correlation_matrix(data):
    """
    Calcula la matriz de correlación de Pearson para un conjunto de señales EEG.
    
    Parámetros:
    -----------
    data : array_like o pandas.DataFrame
         Datos EEG. Si es un array, se asume que tiene forma (n_samples, n_channels).
         Si es un DataFrame, cada columna representa un canal.
         
    Retorna:
    --------
    corr_matrix : pandas.DataFrame
         Matriz de correlación de forma (n_channels, n_channels) con valores entre -1 y 1.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    corr_matrix = data.corr()
    return corr_matrix

def compute_coherence_matrix(data, fs, band=None, nperseg=256):
    """
    Calcula la matriz de coherencia promedio entre pares de canales EEG usando la función de coherencia de Welch.
    
    Parámetros:
    -----------
    data : array_like o pandas.DataFrame
         Datos EEG con forma (n_samples, n_channels) o DataFrame donde cada columna es un canal.
    fs : float
         Frecuencia de muestreo en Hz.
    band : tuple, opcional
         Rango de frecuencia (f_low, f_high) sobre el cual promediar la coherencia.
         Si se omite, se promedia sobre todas las frecuencias.
    nperseg : int, opcional (default=256)
         Número de muestras por segmento para el cálculo de la coherencia.
    
    Retorna:
    --------
    coh_matrix : numpy.ndarray
         Matriz de coherencia (n_channels x n_channels) con valores promedio.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    channels = data.columns
    n_channels = len(channels)
    coh_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            f, Cxy = coherence(data.iloc[:, i].values, data.iloc[:, j].values, fs=fs, nperseg=nperseg)
            if band is not None:
                # Seleccionar índices dentro del rango de la banda
                band_idx = (f >= band[0]) & (f <= band[1])
                avg_coh = np.mean(Cxy[band_idx])
            else:
                avg_coh = np.mean(Cxy)
            coh_matrix[i, j] = avg_coh
            coh_matrix[j, i] = avg_coh  # La matriz es simétrica
    return coh_matrix

def plot_connectivity_matrix(matrix, title="Matriz de Conectividad", channel_labels=None, cmap="viridis"):
    """
    Genera un mapa de calor para visualizar una matriz de conectividad (correlación o coherencia).
    
    Parámetros:
    -----------
    matrix : array_like
         Matriz de conectividad de forma (n_channels, n_channels).
    title : str, opcional
         Título del gráfico.
    channel_labels : list de str, opcional
         Lista de etiquetas para los canales. Si se omite, se generan nombres genéricos.
    cmap : str, opcional
         Colormap para el mapa de calor (default 'viridis').
    """
    plt.figure(figsize=(10, 8))
    if channel_labels is None:
        channel_labels = [f"Ch{i+1}" for i in range(matrix.shape[0])]
    df_matrix = pd.DataFrame(matrix, index=channel_labels, columns=channel_labels)
    sns.heatmap(df_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title(title)
    plt.xlabel("Canales")
    plt.ylabel("Canales")
    plt.show()

if __name__ == "__main__":
    # Ejemplo de uso del módulo connectivity_analysis.py

    # Generar datos simulados: 1000 muestras para 19 canales (por ejemplo, ruido blanco)
    n_samples = 1000
    n_channels = 19
    np.random.seed(42)
    simulated_data = np.random.randn(n_samples, n_channels)
    
    # Convertir a DataFrame y asignar nombres de canales (estándar 10-20)
    channel_names = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 
                     'F4', 'F7', 'F8', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    df_simulated = pd.DataFrame(simulated_data, columns=channel_names)
    
    # Calcular y visualizar la matriz de correlación
    corr_matrix = compute_correlation_matrix(df_simulated)
    print("Matriz de correlación:")
    print(corr_matrix)
    plot_connectivity_matrix(corr_matrix, title="Matriz de Correlación (Simulada)", channel_labels=channel_names, cmap="coolwarm")
    
    # Calcular y visualizar la matriz de coherencia en la banda alfa (8-13 Hz)
    fs = 128  # Frecuencia de muestreo simulada
    band_alpha = (8, 13)
    coh_matrix = compute_coherence_matrix(df_simulated, fs, band=band_alpha, nperseg=128)
    print("Matriz de coherencia (banda alfa):")
    print(coh_matrix)
    plot_connectivity_matrix(coh_matrix, title="Matriz de Coherencia (Banda Alfa)", channel_labels=channel_names, cmap="viridis")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_series(time, signal, title="Time Series", xlabel="Time", ylabel="Amplitude", label=None, show=True):
    """
    Grafica una señal en función del tiempo.

    Parámetros:
    -----------
    time : array_like
        Eje temporal.
    signal : array_like
        Señal a graficar.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    label : str, opcional
        Etiqueta para la leyenda.
    show : bool, opcional (default True)
        Si es True, muestra el gráfico inmediatamente.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid(True)
    if show:
        plt.show()

def plot_multiple_time_series(time, signals, labels, title="Time Series Comparison", xlabel="Time", ylabel="Amplitude", show=True):
    """
    Grafica múltiples señales en el mismo gráfico para comparación.

    Parámetros:
    -----------
    time : array_like
        Eje temporal.
    signals : list of array_like
        Lista de señales a graficar.
    labels : list of str
        Lista de etiquetas para cada señal.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    show : bool, opcional (default True)
        Si es True, muestra el gráfico inmediatamente.
    """
    plt.figure(figsize=(12, 5))
    for signal, label in zip(signals, labels):
        plt.plot(time, signal, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()

def plot_histogram(data, bins=50, title="Histogram", xlabel="Value", ylabel="Frequency", show=True):
    """
    Grafica un histograma de los datos.

    Parámetros:
    -----------
    data : array_like
        Datos a graficar.
    bins : int, opcional (default 50)
        Número de bins para el histograma.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    show : bool, opcional (default True)
        Si es True, muestra el gráfico inmediatamente.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if show:
        plt.show()

def plot_bar_chart(x, y, title="Bar Chart", xlabel="Categories", ylabel="Values", show=True):
    """
    Grafica un gráfico de barras.

    Parámetros:
    -----------
    x : array_like
        Categorías o etiquetas.
    y : array_like
        Valores numéricos.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    show : bool, opcional (default True)
        Si es True, muestra el gráfico inmediatamente.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y')
    if show:
        plt.show()

def plot_heatmap(matrix, title="Heatmap", xlabel="X", ylabel="Y", xticklabels=True, yticklabels=True, cmap="viridis", show=True):
    """
    Genera un mapa de calor para visualizar una matriz.

    Parámetros:
    -----------
    matrix : array_like
        Matriz de datos.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje x.
    ylabel : str
        Etiqueta del eje y.
    xticklabels : bool o list, opcional
        Etiquetas para el eje x. Si es True, se usan índices; si es False, no se muestran; o se puede pasar una lista de etiquetas.
    yticklabels : bool o list, opcional
        Etiquetas para el eje y.
    cmap : str, opcional
        Colormap a utilizar.
    show : bool, opcional (default True)
        Si es True, muestra el gráfico inmediatamente.
    """
    plt.figure(figsize=(10, 8))
    if isinstance(matrix, np.ndarray):
        df_matrix = pd.DataFrame(matrix)
    else:
        df_matrix = matrix
    sns.heatmap(df_matrix, annot=True, fmt=".2f", cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()

if __name__ == "__main__":
    # Ejemplo de uso del módulo de visualización.

    # Ejemplo 1: Graficar una señal de 5 Hz
    fs = 128
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    plot_time_series(t, signal, title="Señal de 5 Hz", xlabel="Tiempo (s)", ylabel="Amplitud", label="5 Hz Sinusoide")

    # Ejemplo 2: Histograma de la señal
    plot_histogram(signal, bins=30, title="Histograma de la Señal", xlabel="Amplitud", ylabel="Frecuencia")

    # Ejemplo 3: Gráfico de barras para valores de potencia en bandas
    categories = ["Delta", "Theta", "Alpha", "Beta"]
    values = [1.2, 0.8, 1.5, 0.9]
    plot_bar_chart(categories, values, title="Potencia en Bandas", xlabel="Bandas", ylabel="Potencia (µV²)")

    # Ejemplo 4: Mapa de calor con una matriz simulada
    matrix = np.random.rand(5, 5)
    channel_labels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]
    plot_heatmap(matrix, title="Mapa de Calor Simulado", xlabel="Canales", ylabel="Canales", xticklabels=channel_labels, yticklabels=channel_labels, cmap="coolwarm")

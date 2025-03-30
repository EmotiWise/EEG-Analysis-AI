#!/usr/bin/env python3

"""
Demo App: Autotest EEG
Esta aplicación de demostración integra la carga, preprocesamiento, 
extracción de características y clasificación de señales EEG para simular
un autotest que evalúa estados de atención (Control vs TDAH) basados en datos.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Agrega el directorio padre (donde se encuentra 'src') al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar módulos del proyecto
from src.data_loader import load_mat_data, mat_to_dataframe
from src.preprocessing import apply_lowpass_filter, apply_notch_filter
from src.feature_extraction import extract_features_from_segments
from src.ml_classification import train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Configuración de parámetros
    data_file = os.path.join("data", "raw", "v1p.mat")
    variable_name = "v1p"
    channel_names = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 
                     'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 
                     'T5', 'T6', 'O1', 'O2']
    fs = 128  # Frecuencia de muestreo en Hz
    segment_length = fs  # Segmentos de 1 segundo (128 muestras)

    # Cargar datos EEG
    print("Cargando datos EEG desde:", data_file)
    try:
        data = load_mat_data(data_file, variable_name=variable_name)
    except Exception as e:
        print("Error al cargar datos:", e)
        return

    # Convertir a DataFrame y asignar nombres de canales
    df = mat_to_dataframe(data, channel_names=channel_names)
    print("Datos cargados con forma:", df.shape)

    # Seleccionar el canal de interés para el demo (ej. Fz)
    signal = df['Fz'].values

    # Preprocesamiento: Aplicar filtro pasa-bajos y filtro notch
    print("Aplicando preprocesamiento a la señal...")
    filtered_signal = apply_lowpass_filter(signal, cutoff=40, fs=fs, order=4)
    filtered_signal = apply_notch_filter(filtered_signal, notch_freq=50, fs=fs, quality_factor=30)

    # Visualizar señal original vs filtrada
    t = np.arange(len(signal)) / fs  # eje temporal en segundos
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="Señal Original", alpha=0.6)
    plt.plot(t, filtered_signal, label="Señal Filtrada", linewidth=2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title("Comparación: Señal Original vs Filtrada (Canal Fz)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Segmentación de la señal filtrada
    num_samples = len(filtered_signal)
    num_segments = num_samples // segment_length
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(filtered_signal[start:end])
    segments = np.array(segments)
    print("Número de segmentos:", segments.shape[0])

    # Extracción de características para cada segmento
    features_list = extract_features_from_segments(segments, fs, nperseg=segment_length)
    features_df = pd.DataFrame(features_list)
    
    # Simulación de etiquetas basada en la potencia en banda alpha
    median_alpha = features_df['alpha'].median()
    features_df['Label'] = (features_df['alpha'] > median_alpha).astype(int)
    features_df['Label_str'] = features_df['Label'].map({1: 'Control', 0: 'TDAH'})
    print("\nTabla de características con etiquetas simuladas:")
    print(features_df.head())

    # Preparar datos para clasificación
    X = features_df[['delta', 'theta', 'alpha', 'beta']].values
    y = features_df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("\nDatos divididos en entrenamiento y prueba:")
    print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)

    # Entrenar un modelo de Regresión Logística
    print("\nEntrenando modelo de Regresión Logística...")
    model = train_logistic_regression(X_train, y_train, max_iter=1000)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Mostrar resultados del modelo
    print("\nResultados del modelo de Regresión Logística:")
    print("Precisión:", metrics['accuracy'])
    print("Matriz de Confusión:\n", metrics['confusion_matrix'])
    print("Reporte de Clasificación:\n", metrics['classification_report'])
    
    # Mensaje final del demo
    print("\nDemo finalizada.")
    print("El autotest EEG simulado indica que el modelo clasifica segmentos como:")
    print("1 (Control - alta atención) y 0 (TDAH - baja atención) con una precisión de {:.2f}%.".format(metrics['accuracy'] * 100))

if __name__ == "__main__":
    main()

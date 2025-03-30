import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_mat_data(file_path, variable_name='v1p', squeeze_me=True):
    """
    Carga un archivo MATLAB (.mat) y extrae la variable especificada.

    Parámetros:
    -----------
    file_path : str
        Ruta al archivo .mat.
    variable_name : str, opcional (default 'v1p')
        Nombre de la variable a extraer del archivo MATLAB.
    squeeze_me : bool, opcional (default True)
        Si se activa, simplifica la estructura de los datos.

    Retorna:
    --------
    data : numpy.ndarray
        Array con los datos cargados.

    Excepciones:
    ------------
    FileNotFoundError:
        Si el archivo no existe.
    KeyError:
        Si la variable no se encuentra en el archivo.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    mat_data = loadmat(file_path, squeeze_me=squeeze_me)
    if variable_name not in mat_data:
        raise KeyError(f"La variable '{variable_name}' no se encontró en el archivo. Llaves disponibles: {list(mat_data.keys())}")
    
    data = mat_data[variable_name]
    return data

def mat_to_dataframe(data, channel_names=None):
    """
    Convierte un array de datos (numpy.ndarray) obtenido de un archivo MATLAB a un DataFrame de pandas.

    Parámetros:
    -----------
    data : numpy.ndarray
        Array de datos cargados desde el archivo MATLAB.
    channel_names : list de str, opcional
        Lista de nombres para asignar a las columnas del DataFrame.
        Si no se proporciona, se utilizarán índices numéricos.

    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame con los datos.
    """
    df = pd.DataFrame(data)
    if channel_names:
        if len(channel_names) == df.shape[1]:
            df.columns = channel_names
        else:
            print("Advertencia: La cantidad de nombres de canales no coincide con el número de columnas. Se usarán nombres por defecto.")
    return df

if __name__ == "__main__":
    # Ejemplo de uso desde la línea de comandos:
    # python src/data_loader.py ruta_al_archivo.mat --variable v1p --channels Fz Cz Pz C3 T3 C4 T4 Fp1 Fp2 F3 F4 F7 F8 P3 P4 T5 T6 O1 O2
    import argparse

    parser = argparse.ArgumentParser(description="Carga un archivo MATLAB y lo convierte a DataFrame")
    parser.add_argument("file_path", type=str, help="Ruta al archivo MATLAB (.mat)")
    parser.add_argument("--variable", type=str, default="v1p", help="Nombre de la variable a extraer (default: v1p)")
    parser.add_argument("--channels", type=str, nargs="*", default=None, help="Lista de nombres de canales para las columnas del DataFrame")
    
    args = parser.parse_args()
    
    try:
        data = load_mat_data(args.file_path, variable_name=args.variable)
        df = mat_to_dataframe(data, channel_names=args.channels)
        print("Datos cargados exitosamente. Forma del DataFrame:", df.shape)
        print(df.head())
    except Exception as e:
        print("Error al cargar los datos:", e)

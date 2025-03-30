-------------------------
Proyecto: EEG-Analysis-AI
-------------------------

![PortadaIA](https://github.com/user-attachments/assets/eb63ed36-2550-4f9f-9856-6042d8437e12)


**Descripción:**
Este proyecto es una plataforma modular y escalable para el análisis de señales EEG mediante técnicas de Inteligencia Artificial.  
Su objetivo es permitir que desde un usuario casero, que realice un autotest en su hogar, hasta investigadores de alto nivel puedan:
  - Cargar y preprocesar datos EEG (por ejemplo, desde archivos MATLAB .mat).
  - Visualizar la señal en el tiempo y extraer características relevantes (potencia en bandas delta, theta, alpha, beta).
  - Analizar la conectividad funcional entre canales.
  - Entrenar y evaluar modelos clásicos (Regresión Logística, SVM) y modelos deep learning (MLP, redes convolucionales) para clasificar estados mentales (alta/baja atención, TDAH, ansiedad, depresión, etc.).
  - Utilizar un demo interactivo para obtener un autotest en tiempo real.

Arquitectura del Repositorio:
-----------------------------

EEG-Analysis-AI/
├── README.md              # Este archivo (documentación completa)
├── LICENSE                # Licencia del proyecto (por ejemplo, MIT)
├── requirements.txt       # Dependencias del proyecto
├── setup.py               # (Opcional) Script de instalación
├── docs/
│   ├── manual_usuario.md       # Guía básica para usuarios caseros
│   ├── manual_avanzado.md      # Documentación avanzada para expertos
│   └── presentacion_proyecto.pdf   # Presentación para stakeholders
├── data/
│   ├── raw/               # Datos originales (por ejemplo, v1p.mat)
│   └── processed/         # Datos procesados (CSV, numpy arrays, etc.)
├── notebooks/             # Notebooks demostrativos por cada "ticket":
│   ├── Ticket1_Configuracion.ipynb   # Configuración y carga de datos
│   ├── Ticket2_Visualizacion.ipynb     # Visualización básica de señales
│   ├── Ticket3_Filtrado.ipynb          # Preprocesamiento de señal EEG (filtrado)
│   ├── Ticket4_Features.ipynb          # Extracción de características simples
│   ├── Ticket5_Clasificacion_Clasico.ipynb  # Modelo clásico de ML
│   ├── Ticket7_RedesNeuronales.ipynb        # Red neuronal básica (MLP)
│   └── Ticket12_Innovacion.ipynb            # Propuesta innovadora (análisis de conectividad)
├── src/
│   ├── data_loader.py             # Funciones para cargar y convertir archivos MATLAB a DataFrame
│   ├── preprocessing.py           # Funciones para filtrar y limpiar señales EEG
│   ├── feature_extraction.py      # Funciones para extraer características (energía en bandas, etc.)
│   ├── connectivity_analysis.py   # Funciones para analizar conectividad (correlación, coherencia)
│   ├── ml_classification.py       # Modelos clásicos de ML (Regresión Logística, SVM)
│   ├── deep_learning.py           # Arquitecturas y entrenamiento de redes neuronales (PyTorch)
│   ├── visualization.py           # Funciones para visualización (gráficos, mapas de calor, etc.)
│   └── utils.py                   # Funciones utilitarias
├── experiments/           # Notebooks y scripts de experimentación
└── examples/              # Ejemplos y demo (por ejemplo, demo_app.py)

Uso del Proyecto:
-----------------

1. **Instalación:**
   - Clona el repositorio.
   - Instala las dependencias con:
     ```
     pip install -r requirements.txt
     ```

2. **Documentación:**

   - Consulta los manuales en `docs/manual_usuario.md` (para usuarios caseros) y `docs/manual_avanzado.md` (para desarrolladores y expertos).

3. **Ejecución del Demo:**

   - Ejecuta el demo interactivo desde `examples/demo_app.py` para realizar un autotest EEG:
     ```
     python examples/demo_app.py
     ```
   - Este demo cargará los datos EEG (por ejemplo, `v1p.mat` en `data/raw`), aplicará preprocesamiento, segmentación y extracción de características, y finalmente entrenará un modelo de clasificación para evaluar el estado de atención.

4. **Modificación y Extensión:**

   - Los módulos en `src/` están organizados de forma modular para que cualquier usuario pueda modificar, extender o mejorar la funcionalidad (por ejemplo, cambiar parámetros de filtrado, ajustar modelos, implementar nuevas técnicas de análisis).

5. **Autoanálisis Personal:**

   - Adquiere tu señal EEG utilizando un dispositivo compatible siguiendo el estándar 10-20.
   - Guarda la señal en un archivo `.mat` (por ejemplo, `v1p.mat`) y colócala en `data/raw/`.
   - Ejecuta el demo y sigue las instrucciones para obtener un informe de tu estado mental basado en la atención, TDAH, ansiedad, depresión, etc.

Propuesta Innovadora:
----------------------
Además de los análisis tradicionales, el proyecto incluye una propuesta innovadora que utiliza el análisis de conectividad funcional para detectar biomarcadores de condiciones como la depresión.  
Se calcula la matriz de correlación y coherencia entre los 19 canales EEG para identificar patrones de conectividad alterados, lo cual puede mejorar la precisión en la clasificación de estados mentales y aportar a diagnósticos más personalizados.

Instrucciones para Usuarios Avanzados:
---------------------------------------
- Explora el código en `src/` para comprender el flujo completo (desde la carga de datos hasta el modelado y la visualización).
- Utiliza los notebooks en `notebooks/` y `experiments/` para probar nuevas ideas y ajustar parámetros.
- Consulta `docs/manual_avanzado.md` para una descripción detallada de la arquitectura, los algoritmos y las recomendaciones para ampliar el proyecto.
- Contribuye mediante pull requests y comparte tus mejoras con la comunidad.

Conclusión:
-----------
EEG-Analysis-AI es una plataforma integral diseñada para el análisis de señales EEG y la evaluación de estados mentales mediante Inteligencia Artificial. Su modularidad y documentación exhaustiva permiten que tanto un usuario casual pueda realizar un autotest en su hogar, como que un equipo de investigación internacional pueda ampliar y personalizar la herramienta para aplicaciones avanzadas en neurotecnología.

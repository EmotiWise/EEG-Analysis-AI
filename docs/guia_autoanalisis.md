========================================================
Guía Completa para Autoanálisis EEG con EEG-Analysis-AI
========================================================

INTRODUCCIÓN
------------
EEG-Analysis-AI es una plataforma modular y escalable diseñada para el análisis de señales EEG mediante técnicas de Inteligencia Artificial. Con ella, podrás obtener un autotest en tiempo real que evalúe diferentes aspectos de tu actividad cerebral, tales como niveles de atención, posibles indicios de TDAH, ansiedad, depresión y más.

Esta guía te explica cómo adquirir tus datos EEG, cargarlos en el sistema, ejecutar el análisis de forma automatizada y, en caso de ser experto, cómo personalizar y ampliar la herramienta.

REQUISITOS
----------
- Dispositivo EEG compatible:
  • **OpenBCI Cyton Board** (ampliable con el Daisy Module) 
    - Ventajas: Código abierto, alta flexibilidad y escalabilidad.
  • **Muse Headband**
    - Ventajas: Fácil de usar, portátil. (En este caso, la señal cuenta con 4 canales: TP9, AF7, AF8, TP10.)
- Computadora con Python 3.x.
- Conexión a Internet para instalar dependencias.
- (Opcional) Software para captura de datos:
  • Muse-IO o muse-lsl (para Muse Headband) que permita exportar la señal a un archivo MATLAB (.mat).

INSTALACIÓN DEL PROYECTO
-------------------------
1. Clona el repositorio:
   git clone https://github.com/EmotiWise/EEG-Analysis-AI.git
2. Navega al directorio del proyecto:
   cd EEG-Analysis-AI
3. Instala las dependencias:
   pip install -r requirements.txt

ESTRUCTURA DEL PROYECTO
------------------------
El repositorio se organiza de la siguiente manera:

EEG-Analysis-AI/
├── README.md              --> Documentación completa del proyecto.
├── LICENSE                --> Tipo de licencia (por ejemplo, MIT).
├── requirements.txt       --> Dependencias necesarias.
├── docs/                  --> Manuales de usuario y avanzado.
│   ├── manual_usuario.md
│   └── manual_avanzado.md
├── data/
│   ├── raw/               --> Datos originales (por ejemplo, v1p.mat).
│   └── processed/         --> Datos procesados.
├── notebooks/             --> Notebooks demostrativos (Tickets 1,2,3,4,5,7,12).
├── src/                   --> Módulos de código para carga, preprocesamiento, extracción de features, análisis, visualización, etc.
├── experiments/           --> Scripts y notebooks experimentales.
└── examples/              --> Demo interactivo (demo_app.py).

CAPTURA Y PREPARACIÓN DE DATOS EEG
----------------------------------
1. **Adquisición:**
   - **Si usas Muse Headband:**
     • Conecta el Muse y utiliza Muse-IO o muse-lsl para capturar la señal.
     • Exporta los datos en formato MATLAB (.mat) con la variable "v1p". Nota: Los datos tendrán 4 canales.
   - **Si usas OpenBCI:**
     • Configura el dispositivo y registra tus datos siguiendo el estándar 10-20.
     • Exporta los datos en formato MATLAB (.mat) con la variable "v1p". Podrás tener más canales (por ejemplo, 16 o 19).

2. **Preparación del Archivo:**
   - Coloca el archivo generado (por ejemplo, "v1p.mat") en la carpeta "data/raw/" del repositorio.
   - Para Muse Headband, modifica la asignación de canales a:
       ['TP9', 'AF7', 'AF8', 'TP10']
     Mientras que para OpenBCI, utiliza la asignación que corresponda (por ejemplo, los canales estándar 10-20).

EJECUCIÓN DEL ANÁLISIS (DEMO)
-----------------------------
Para facilitar el proceso, el proyecto incluye un demo interactivo que realiza todos los pasos del flujo:
1. Carga del archivo EEG desde "data/raw/v1p.mat".
2. Preprocesamiento de la señal (filtro pasa-bajos y filtro notch).
3. Segmentación de la señal en ventanas (por ejemplo, de 1 segundo, 128 muestras).
4. Extracción de características: Cálculo de la potencia en bandas de frecuencia (delta, theta, alpha, beta) utilizando el método de Welch.
5. Simulación de etiquetas (por ejemplo, basadas en la potencia en la banda alpha) para clasificar la señal en “Control” (alta atención) o “TDAH” (baja atención).
6. Entrenamiento y evaluación de un modelo de clasificación (Regresión Logística).
7. Visualización de resultados: Gráficos comparativos (señal original vs filtrada), tablas de características y reportes de desempeño.

Para ejecutar el demo, abre una terminal en el directorio del proyecto y escribe:
   python examples/demo_app.py

El script realizará automáticamente todo el proceso y te mostrará los resultados tanto en gráficos como en la consola.

INTERPRETACIÓN DE RESULTADOS
----------------------------
- **Gráficos:** Revisa la señal original y la señal filtrada para confirmar que los filtros han eliminado el ruido.
- **Reporte de Clasificación:** El sistema mostrará la precisión del modelo y te indicará la clasificación de los segmentos (por ejemplo, "Control" vs "TDAH").
- **Tablas de Características:** Podrás ver la potencia en cada banda de frecuencia para cada segmento, lo cual sirve de base para el análisis.

PERSONALIZACIÓN Y EXTENSIÓN
---------------------------
- **Para Usuarios Avanzados:**
  • Modifica los módulos en "src/" para ajustar parámetros del preprocesamiento (por ejemplo, frecuencias de corte, orden de los filtros).
  • Amplía la extracción de características con nuevos indicadores (por ejemplo, medidas de conectividad, entropía, análisis de grafos).
  • Entrena nuevos modelos o modifica los existentes en "src/ml_classification.py" o "src/deep_learning.py".
  • Consulta "docs/manual_avanzado.md" para una guía detallada sobre la arquitectura y las posibilidades de extensión.
- **Interfaz Gráfica (Opcional):**
  • Considera desarrollar una interfaz interactiva con herramientas como Streamlit o Dash para ejecutar el análisis con un solo clic.

USO CONTINUO Y AUTOANÁLISIS
--------------------------
Cada vez que realices un nuevo registro EEG:
1. Coloca el nuevo archivo en "data/raw/" (reemplazando o agregando, según convenga).
2. Ejecuta nuevamente "python examples/demo_app.py" para obtener un análisis actualizado.
3. Revisa los resultados y, si lo deseas, ajusta los parámetros según tus necesidades.

CONCLUSIÓN
----------
EEG-Analysis-AI es una herramienta integral que te permite realizar un autoanálisis EEG de forma automatizada y escalable. La plataforma ha sido diseñada para ser accesible y personalizable, permitiendo que tanto usuarios caseros como investigadores obtengan insights valiosos sobre su actividad cerebral.

Consejo Profesional:
---------------------
La clave del éxito en este proyecto es la integración sencilla de hardware y software. Asegúrate de seguir esta guía paso a paso para configurar tu dispositivo, capturar y analizar tus datos EEG. La iteración y la personalización continua te permitirán adaptar la herramienta a tus necesidades y contribuir al avance en neurotecnología. ¡Sigue explorando, aprendiendo y compartiendo tus hallazgos!

=======================================================
Fin de la Guía de Autoanálisis EEG con EEG-Analysis-AI
=======================================================
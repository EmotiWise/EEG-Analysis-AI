====================================
Manual de Usuario - EEG-Analysis-AI
====================================

Introducción
------------
Bienvenido a EEG-Analysis-AI, una plataforma diseñada para que cualquier usuario, desde un entusiasta casero hasta alguien en formación, pueda realizar un análisis sencillo de señales EEG y obtener un autotest de su estado mental. Este manual te guiará paso a paso en el uso básico de la herramienta.

Requisitos
----------
- Un dispositivo EEG compatible con el estándar 10-20.
- Archivo de datos EEG en formato MATLAB (.mat), por ejemplo, "v1p.mat".
- Python 3.x instalado.
- Conexión a internet para descargar las dependencias del proyecto.

Instalación
-----------
1. Clona el repositorio:
git clone https://github.com/EmotiWise/EEG-Analysis-AI.git

2. Navega al directorio del proyecto:
cd EEG-Analysis-AI

3. Instala las dependencias:
pip install -r requirements.txt

Uso Básico
----------

1. Coloca tu archivo EEG (por ejemplo, `v1p.mat`) en la carpeta `data/raw/`.

2. Ejecuta el demo interactivo:
python examples/demo_app.py

3. El sistema cargará los datos, aplicará el preprocesamiento (filtrado, segmentación), extraerá características y entrenará un modelo de clasificación para evaluar tu señal EEG.

4. Se mostrarán gráficos comparativos (señal original vs. filtrada) y se presentará un informe en consola con los resultados (por ejemplo, clasificación en “Control” o “TDAH”).

Interpretación de Resultados
----------------------------

- **Visualización de la Señal:** Comprueba cómo se elimina el ruido mediante los filtros aplicados.

- **Reporte de Clasificación:** Indica el estado de atención según el modelo entrenado, basándose en características extraídas (por ejemplo, potencia en la banda alpha).

Soporte y Ayuda
---------------

Si tienes dudas o encuentras problemas, consulta este manual o abre un issue en el repositorio para recibir asistencia.

¡Disfruta explorando tu estado mental con EEG-Analysis-AI!
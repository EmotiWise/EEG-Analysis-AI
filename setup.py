from setuptools import setup, find_packages

setup(
    name="EEG-Analysis-AI",
    version="0.1.0",
    description="Plataforma modular para el análisis y modelado de señales EEG con herramientas de ML y DL.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EmotiWise/EEG-Analysis-AI",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "torch",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            #**************** Agrega aquí tus scripts de consola si los tienes *************
        ],
    },
)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Entrena un modelo de Regresión Logística utilizando scikit-learn.
    
    Parámetros:
    -----------
    X_train : array_like
        Características de entrenamiento.
    y_train : array_like
        Etiquetas de entrenamiento.
    **kwargs : dict
        Parámetros adicionales para el constructor de LogisticRegression (por ejemplo, max_iter).
        
    Retorna:
    --------
    model : LogisticRegression
        Modelo entrenado.
    """
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, **kwargs):
    """
    Entrena un modelo de Máquina de Vectores de Soporte (SVM) utilizando scikit-learn.
    
    Parámetros:
    -----------
    X_train : array_like
        Características de entrenamiento.
    y_train : array_like
        Etiquetas de entrenamiento.
    **kwargs : dict
        Parámetros adicionales para el constructor de SVC (por ejemplo, kernel, C, probability).
        
    Retorna:
    --------
    model : SVC
        Modelo entrenado.
    """
    model = SVC(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación y retorna métricas de desempeño.
    
    Parámetros:
    -----------
    model : objeto
        Modelo entrenado.
    X_test : array_like
        Conjunto de características de prueba.
    y_test : array_like
        Etiquetas verdaderas de prueba.
        
    Retorna:
    --------
    metrics : dict
        Diccionario con precisión, matriz de confusión y reporte de clasificación.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    return metrics

def cross_validate_model(model, X, y, cv=5):
    """
    Realiza validación cruzada para evaluar el desempeño de un modelo.
    
    Parámetros:
    -----------
    model : objeto
        Modelo de clasificación (sin entrenar).
    X : array_like
        Características.
    y : array_like
        Etiquetas.
    cv : int, opcional (default=5)
        Número de folds para la validación cruzada.
        
    Retorna:
    --------
    scores : ndarray
        Array de puntajes de precisión para cada fold.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores

if __name__ == "__main__":
    # Ejemplo de uso del módulo ml_classification.py

    # Crear un dataset de ejemplo utilizando make_classification
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generar dataset: 200 muestras, 4 features, 2 clases
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, 
                               n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entrenar y evaluar Regresión Logística
    model_lr = train_logistic_regression(X_train, y_train, max_iter=1000)
    metrics_lr = evaluate_model(model_lr, X_test, y_test)
    print("Desempeño de Regresión Logística:")
    print("Precisión:", metrics_lr['accuracy'])
    print("Matriz de Confusión:\n", metrics_lr['confusion_matrix'])
    print("Reporte de Clasificación:\n", metrics_lr['classification_report'])
    
    # Entrenar y evaluar SVM
    model_svm = train_svm(X_train, y_train, kernel='linear', probability=True)
    metrics_svm = evaluate_model(model_svm, X_test, y_test)
    print("\nDesempeño de SVM:")
    print("Precisión:", metrics_svm['accuracy'])
    print("Matriz de Confusión:\n", metrics_svm['confusion_matrix'])
    print("Reporte de Clasificación:\n", metrics_svm['classification_report'])
    
    # Validación cruzada con Regresión Logística
    cv_scores = cross_validate_model(LogisticRegression(max_iter=1000), X, y, cv=5)
    print("\nValidación Cruzada (Regresión Logística):")
    print("Puntajes por fold:", cv_scores)
    print("Precisión media:", np.mean(cv_scores))

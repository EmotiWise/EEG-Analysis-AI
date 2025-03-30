import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    """
    Modelo de red neuronal Multilayer Perceptron (MLP) simple para clasificación.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión del vector de entrada.
    hidden_dims : list of int
        Lista con las dimensiones de cada capa oculta.
    output_dim : int
        Número de clases de salida.
    """
    def __init__(self, input_dim=4, hidden_dims=[16, 8], output_dim=2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """
    Entrena el modelo utilizando un ciclo de entrenamiento.

    Parámetros:
    -----------
    model : nn.Module
        Modelo a entrenar.
    train_loader : DataLoader
        DataLoader con el conjunto de entrenamiento.
    criterion : función de pérdida
    optimizer : optimizador de PyTorch.
    num_epochs : int
        Número de épocas de entrenamiento.
    device : str
        Dispositivo donde entrenar ('cpu' o 'cuda').
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evalúa el modelo en el conjunto de prueba y muestra métricas de desempeño.

    Parámetros:
    -----------
    model : nn.Module
        Modelo entrenado.
    test_loader : DataLoader
        DataLoader con el conjunto de prueba.
    device : str
        Dispositivo donde evaluar ('cpu' o 'cuda').

    Retorna:
    --------
    accuracy : float
        Precisión del modelo en el conjunto de prueba.
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds)
    
    print("Accuracy: {:.4f}".format(accuracy))
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return accuracy, conf_matrix, class_report

if __name__ == "__main__":
    # Ejemplo de uso: Entrenar un modelo MLP con datos sintéticos.
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generar datos sintéticos: 200 muestras, 4 features, 2 clases.
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, 
                               n_redundant=0, random_state=42)
    # Dividir en entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convertir a tensores de PyTorch.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Crear DataLoaders.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Instanciar el modelo.
    model = MLP(input_dim=4, hidden_dims=[16, 8], output_dim=2)
    print("Arquitectura del modelo:")
    print(model)

    # Definir función de pérdida y optimizador.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Entrenar el modelo.
    print("Entrenando el modelo...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device='cpu')

    # Evaluar el modelo.
    print("\nEvaluando el modelo en el conjunto de prueba:")
    evaluate_model(model, test_loader, device='cpu')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import sawtooth

# Configuración inicial para Matplotlib (similar a set(0, ...) en MATLAB)
# Establecer color de fondo de la figura en blanco
plt.rcParams['figure.facecolor'] = 'white'
# Establecer ancho de línea predeterminado
plt.rcParams['lines.linewidth'] = 2

# --- 1. Generar señales ---
# Generar señal de tiempo
num_points = 100
t = np.linspace(0, 2 * np.pi, num_points)

# Señal de entrada (triangular - sawtooth con ancho del 0.5)
# MATLAB's sawtooth(t, 0.5) ranges from -1 to 1.
frecuencia = 1.0
fase_desfase = -np.pi / 2.0

# Señal triangular de entrada: (t * frecuencia) es el tiempo normalizado. Le restamos la fase.
x_np = -sawtooth(frecuencia * t + fase_desfase, 0.5)

# Señal objetivo (seno)
y_np = np.sin(t)

# --- 2. Visualizar las señales originales ---
plt.figure()
plt.plot(t, x_np, 'r', label='Señal de Entrada (Triangular)')
plt.plot(t, y_np, 'b', label='Señal Objetivo (Seno)')
plt.title('Señal de Entrada y Objetivo')
plt.xlabel('t')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show(block=False)  # Muestra la figura sin detener la ejecución

# --- 3. Preparar los datos para PyTorch ---
# MATLAB's feedforwardnet espera datos con:
# - Entrada (x): PxQ (P=características, Q=muestras)
# - Objetivo (y): SxQ (S=salidas, Q=muestras)

# En este caso, P=1, S=1, Q=100.

# Convertir NumPy arrays a PyTorch Tensors
# Redimensionar para que tengan la forma [muestras, características] para PyTorch
# o [características, muestras] para ser más parecido a MATLAB, pero
# usaremos [muestras, características] que es más estándar en PyTorch (100x1)

X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # Forma (100, 1)
Y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # Forma (100, 1)

# --- 4. Definir la Red Neuronal Feedforward en PyTorch ---


class FeedforwardNet(nn.Module):
    def __init__(self, hidden_layers):
        super(FeedforwardNet, self).__init__()
        # Definir la estructura de las capas
        # La entrada es 1 (una característica: el valor de la señal triangular)
        # La salida es 1 (una predicción: el valor de la señal seno)
        input_size = 1
        output_size = 1
        layer_sizes = [input_size] + hidden_layers + [output_size]

        layers = []
        for i in range(len(layer_sizes) - 1):
            # Capa Lineal (Totalmente Conectada)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Función de activación (ReLU es común, similar a tan-sigmoide en propósito)
            # En MATLAB, feedforwardnet por defecto usa 'tansig' en capas ocultas.
            # Aquí, usaremos ReLU. Si se necesita 'tanh', se cambiaría.
            if i < len(layer_sizes) - 2:  # No aplica activación a la capa de salida
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Definir la estructura de las capas ocultas
hidden_layer_sizes = [4, 4, 4]
net_model = FeedforwardNet(hidden_layer_sizes)

# --- 5. Configurar Entrenamiento (Optimización) ---
# En PyTorch, no hay una función 'trainscg' directa. Usaremos un optimizador común como Adam.
# Para replicar 'trainscg' (Scaled Conjugate Gradient) se necesitaría una implementación
# personalizada o librerías externas. Adam o LBFGS son buenas alternativas.
# Tasa de aprendizaje pequeña
optimizer = optim.Adam(net_model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Función de pérdida de Error Cuadrático Medio (MSE)
epochs = 2500
# El 'goal' en PyTorch se maneja deteniendo el entrenamiento si la pérdida baja de este valor.
goal = 0.0001

# --- 6. Entrenamiento de la Red ---
print("\n--- Entrenamiento de la Red ---")
for epoch in range(epochs):
    # Puesta a cero de los gradientes
    optimizer.zero_grad()

    # Forward pass
    y_pred_tensor = net_model(X)

    # Calcular la pérdida (Error Cuadrático Medio)
    loss = criterion(y_pred_tensor, Y)

    # Backward pass y optimización
    loss.backward()
    optimizer.step()

    current_mse = loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], MSE Loss: {current_mse:.6f}')

    # Condición de detención por objetivo
    if current_mse <= goal:
        print(f"Objetivo de pérdida alcanzado ({goal}) en la época {epoch+1}.")
        break

# --- 7. Salida de la Red y Cálculo de Errores ---
# Obtener predicciones finales
net_model.eval()  # Poner el modelo en modo de evaluación
with torch.no_grad():  # Desactivar el cálculo de gradientes
    y_pred_tensor = net_model(X)

# Convertir la predicción de PyTorch Tensor a NumPy array
y_pred_np = y_pred_tensor.numpy().squeeze()

# Calcular el error MSE (ya lo tenemos de la última iteración, pero lo volvemos a calcular para ser exactos)
mse_Error = np.mean((y_np - y_pred_np)**2)
print(f'\nError MSE (PyTorch): {mse_Error*100:.4f}%')

# Calcular el error relativo
# Nota: La variable 't_pred' en tu código MATLAB es probablemente un error tipográfico y debería ser 'y_pred'
error_relativo = np.max(np.abs(y_np - y_pred_np)) / np.max(np.abs(y_np)) * 100
print(f'Error Relativo: {error_relativo:.2f}%')


# --- 8. Visualizar Resultados ---
plt.figure()
plt.plot(t, y_np, 'b', label='Seno Real')
plt.plot(t, y_pred_np, 'g', label='Salida de la Red Neuronal')
plt.title('Comparación: Seno Real vs. Salida de la Red')
plt.xlabel('t')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()

# Nota: La aleatoriedad inicial en PyTorch y el uso del optimizador Adam en lugar de SCG
# significa que los resultados del entrenamiento (MSE final, error relativo) serán diferentes a los de MATLAB.

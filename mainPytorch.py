import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

# Cargar datos desde CSV (ajusta las rutas y nombres de archivo)
# header=None si no hay encabezado
dog_wave_df = pd.read_csv('./perros.csv', header=None)
cat_wave_df = pd.read_csv('./gatos.csv', header=None)

# Convertir dataframes a tensores PyTorch (float)
dog_wave = torch.tensor(dog_wave_df.values, dtype=torch.float32)
cat_wave = torch.tensor(cat_wave_df.values, dtype=torch.float32)

# Preparar datos, segmentar y concatenar igual que en MATLAB
x = torch.cat((dog_wave[:, :40], cat_wave[:, :40]), dim=1).T
x2 = torch.cat((dog_wave[:, 40:80], cat_wave[:, 40:80]), dim=1).T

# Crear etiquetas: 0 para dog, 1 para cat
labels = torch.cat((torch.ones(40), torch.zeros(40))).long()

print(x.shape)
print(x2.shape)
# Definir red neuronal (1 capa oculta de 10 neuronas con tanh)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x.shape[1], 10)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(10, 2)  # salida 2 clases

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Entrenamiento
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# Evaluar en x y x2
with torch.no_grad():
    outputs_x = net(x)
    _, predicted_x = torch.max(outputs_x, 1)

    outputs_x2 = net(x2)
    _, predicted_x2 = torch.max(outputs_x2, 1)

# Calcular precisión (como ejemplo de performance)
accuracy_x = (predicted_x == labels).float().mean().item()
accuracy_x2 = (predicted_x2 == labels).float().mean().item()

print(f'Accuracy x: {accuracy_x:.2f}')
print(f'Accuracy x2: {accuracy_x2:.2f}')

# Preparar clases para graficar (similar a clases2 y clases3)
classes2 = predicted_x.numpy()
classes3 = predicted_x2.numpy()

# Graficar barras (similar al subplot y bar de MATLAB)
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.bar(range(len(predicted_x.numpy())), predicted_x.numpy(),
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x')

plt.subplot(4, 1, 2)
plt.bar(range(len(predicted_x2.numpy())), predicted_x2.numpy(),
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x2')

plt.tight_layout()
plt.show()

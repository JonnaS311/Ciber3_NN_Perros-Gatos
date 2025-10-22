import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Carga de datos (igual que en PyTorch) ---
# Asegúrate de que los archivos './perros.csv' y './gatos.csv' existan
try:
    dog_wave_df = pd.read_csv('./perros.csv', header=None)
    cat_wave_df = pd.read_csv('./gatos.csv', header=None)
except FileNotFoundError:
    print("Advertencia: No se encontraron 'perros.csv' o 'gatos.csv'.")
    print("Creando datos ficticios para que el script se pueda ejecutar...")
    # Creamos datos ficticios (N=50 características, 80 muestras/columnas)
    num_features_demo = 50
    num_samples_demo = 80
    dog_wave_df = pd.DataFrame(np.random.rand(
        num_features_demo, num_samples_demo))
    # Hacemos los datos de gatos ligeramente diferentes
    cat_wave_df = pd.DataFrame(np.random.rand(
        num_features_demo, num_samples_demo) + 0.5)


# Convertir a NumPy arrays (TensorFlow/Keras trabaja nativamente con NumPy)
dog_wave = dog_wave_df.values
cat_wave = cat_wave_df.values

# --- Preparar datos ---
# La lógica es idéntica, pero usamos np.concatenate y .T
# x: Primeras 40 muestras (columnas) de cada clase
x = np.concatenate((dog_wave[:, :40], cat_wave[:, :40]), axis=1).T
# x2: Siguientes 40 muestras (columnas 40 a 80)
x2 = np.concatenate((dog_wave[:, 40:80], cat_wave[:, 40:80]), axis=1).T

# Crear etiquetas: 1 para dog (primeras 40), 0 para cat (siguientes 40)
# Usamos .astype(np.int32) para que Keras las reconozca como etiquetas de clase
labels = np.concatenate((np.ones(40), np.zeros(40))).astype(np.int32)

# Obtenemos el número de características de entrada
num_features = x.shape[1]

print(f"Shape de x (entrenamiento): {x.shape}")
print(f"Shape de x2 (prueba): {x2.shape}")
print(f"Shape de labels: {labels.shape}")


# --- Definir red neuronal con Keras Sequential ---
# Esta es la forma más simple en Keras de apilar capas
model = keras.Sequential([
    # Capa oculta: 10 neuronas, activación 'tanh'
    # 'input_shape' solo se necesita en la primera capa
    layers.Dense(10, activation='tanh', input_shape=(num_features,)),

    # Capa de salida: 2 neuronas (una para cada clase)
    # No especificamos activación, por lo que la salida son 'logits' (valores crudos)
    # Esto es lo que espera la función de pérdida CrossEntropy
    layers.Dense(2)
])

# Muestra un resumen del modelo
model.summary()

# --- Compilar el modelo ---
# En Keras, 'compilar' configura el proceso de entrenamiento
model.compile(
    # Optimizador Adam, equivalente a optim.Adam
    optimizer=keras.optimizers.Adam(learning_rate=0.005),

    # Función de pérdida:
    # SparseCategoricalCrossentropy es la equivalente a nn.CrossEntropyLoss
    # cuando las etiquetas son enteros (0, 1, 2...)
    # 'from_logits=True' le dice que la salida del modelo no tiene activación (como en PyTorch)
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    # Métrica para monitorear durante el entrenamiento
    metrics=['accuracy']
)

# --- Callback para imprimir cada 100 epochs ---
# Esto replica el comportamiento de tu bucle de PyTorch


class PrintLossCallback(keras.callbacks.Callback):
    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        # El epoch es basado en 0, por eso sumamos 1
        if (epoch + 1) % self.frequency == 0:
            print(
                f'Epoch {epoch + 1} Loss: {logs["loss"]:.4f} Accuracy: {logs["accuracy"]:.4f}')


# --- Entrenamiento ---
epochs = 1500
print("\nIniciando entrenamiento...")

history = model.fit(
    x,
    labels,
    epochs=epochs,
    # Usamos un tamaño de batch igual al total de datos (full-batch)
    # para que sea idéntico a tu entrenamiento de PyTorch
    batch_size=x.shape[0],
    verbose=0,  # Silenciamos el log por defecto de Keras...
    # ...y usamos nuestro callback personalizado
    callbacks=[PrintLossCallback(100)]
)

print("Entrenamiento finalizado.")

# --- Evaluar en x y x2 ---
# 'model.predict' devuelve las salidas (logits)
outputs_x_logits = model.predict(x)
outputs_x2_logits = model.predict(x2)

# Usamos np.argmax para obtener el índice de la clase predicha (0 o 1)
# axis=1 para encontrar el máximo en cada fila (muestra)
predicted_x = np.argmax(outputs_x_logits, axis=1)
predicted_x2 = np.argmax(outputs_x2_logits, axis=1)

# Calcular precisión (NumPy)
accuracy_x = np.mean(predicted_x == labels)
accuracy_x2 = np.mean(predicted_x2 == labels)

print(f'\nAccuracy x: {accuracy_x:.2f}')
print(f'Accuracy x2: {accuracy_x2:.2f}')


# --- Graficar barras (idéntico al código PyTorch) ---
# Matplotlib funciona perfectamente con arrays de NumPy
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.bar(range(len(predicted_x)), predicted_x,
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x (TensorFlow)')

plt.subplot(4, 1, 2)
plt.bar(range(len(predicted_x2)), predicted_x2,
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x2 (TensorFlow)')

plt.tight_layout()
plt.show()

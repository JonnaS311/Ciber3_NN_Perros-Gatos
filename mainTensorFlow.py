import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos desde CSV
dog_wave_df = pd.read_csv('./perros.csv', header=None)
cat_wave_df = pd.read_csv('./gatos.csv', header=None)

# Convertir DataFrames a tensores float32
dog_wave = tf.convert_to_tensor(dog_wave_df.values, dtype=tf.float32)
cat_wave = tf.convert_to_tensor(cat_wave_df.values, dtype=tf.float32)

# Preparar datos (igual que en PyTorch)
x = tf.concat([dog_wave[:, :40], cat_wave[:, :40]], axis=1)
x2 = tf.concat([dog_wave[:, 40:80], cat_wave[:, 40:80]], axis=1)
x = tf.transpose(x)
x2 = tf.transpose(x2)

# Etiquetas: 0 gato, 1 perro (igual al original)
labels = tf.concat([tf.ones(40, dtype=tf.int32),
                   tf.zeros(40, dtype=tf.int32)], axis=0)

# Definir red neuronal (1 capa oculta de 10 neuronas con tanh)
model = models.Sequential([
    layers.Dense(10, activation='tanh', input_shape=(x.shape[1],)),
    layers.Dense(2)  # salida con 2 clases
])

# Compilar modelo
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar modelo
history = model.fit(x, labels, epochs=1000, verbose=0)
for epoch in range(0, 1000, 100):
    print(f"Epoch {epoch} Loss: {history.history['loss'][epoch]:.4f}")

# Evaluar en x y x2
pred_x = tf.argmax(model(x), axis=1)
pred_x2 = tf.argmax(model(x2), axis=1)

# Calcular precisión
accuracy_x = tf.reduce_mean(
    tf.cast(pred_x == tf.cast(labels, tf.int64), tf.float32)).numpy()
accuracy_x2 = tf.reduce_mean(
    tf.cast(pred_x2 == tf.cast(labels, tf.int64), tf.float32)).numpy()


print(f'Accuracy x: {accuracy_x:.2f}')
print(f'Accuracy x2: {accuracy_x2:.2f}')

# Graficar barras (igual que en PyTorch)
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.bar(range(len(pred_x.numpy())), pred_x.numpy(),
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x')

plt.subplot(4, 1, 2)
plt.bar(range(len(pred_x2.numpy())), pred_x2.numpy(),
        color=[.6, .6, .6], edgecolor='k')
plt.title('Predicción en x2')

plt.tight_layout()
plt.show()

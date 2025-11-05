import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Sistema de Rössler
# ---------------------------


def rossler(state, t, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]


# Simulación real
t = np.linspace(0, 200, 5000)
state0 = [0.1, 0.1, 0.1]
states = odeint(rossler, state0, t)

# Dataset
X = states[:-1]
y = states[1:]

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test,  y_test = X[train_size:], y[train_size:]

# ---------------------------
# Normalización
# ---------------------------
scaler_in = StandardScaler()
scaler_out = StandardScaler()

X_train = scaler_in.fit_transform(X_train)
y_train = scaler_out.fit_transform(y_train)
X_test = scaler_in.transform(X_test)
y_test = scaler_out.transform(y_test)

# ---------------------------
# MLP Regressor (Optimizado)
# ---------------------------
net = MLPRegressor(
    hidden_layer_sizes=(65, 65, 65, 65),
    activation='relu',
    solver='adam',
    learning_rate_init=0.00001,
    max_iter=1500,
    tol=1e-7,
    n_iter_no_change=100,
    random_state=1,
    verbose=True
)

net.fit(X_train, y_train)

# ---------------------------
# Predicción Auto-Regresiva
# ---------------------------
pred = []
current = X_test[0]

for _ in range(len(X_test)):
    out = net.predict(current.reshape(1, -1))
    pred.append(out[0])
    current = out

pred = scaler_out.inverse_transform(pred)
real = y[len(X_train):len(X_train)+len(pred)]

# ---------------------------
# Gráfica
# ---------------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Eje X
axs[0].plot(real[:, 0], label="Real x")
axs[0].plot(pred[:, 0], "--", label="NN x")
axs[0].set_ylabel("x(t)")
axs[0].legend()
axs[0].grid(True)

# Eje Y
axs[1].plot(real[:, 1], label="Real y")
axs[1].plot(pred[:, 1], "--", label="NN y")
axs[1].set_ylabel("y(t)")
axs[1].legend()
axs[1].grid(True)

# Eje Z
axs[2].plot(real[:, 2], label="Real z")
axs[2].plot(pred[:, 2], "--", label="NN z")
axs[2].set_ylabel("z(t)")
axs[2].set_xlabel("Tiempo")
axs[2].legend()
axs[2].grid(True)

plt.suptitle(
    "Sistema de Rössler — Predicción Auto-Regresiva (x, y, z)", fontsize=14)
plt.tight_layout()
plt.show()

# Trayectoria 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(real[:, 0], real[:, 1], real[:, 2], label="Real")
ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "--", label="NN")
ax.set_title("Atractor de Rössler — Real vs NN")
ax.legend()
plt.show()

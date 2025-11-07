import numpy as np
from scipy import signal
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------- 1. Parámetros del sistema ----------
a4, a3, a2, a1, a0 = 1.0, 2.0, 10.0, 6.0, 5.0
num = [1.0]
den = [a4, a3, a2, a1, a0]
sys = signal.TransferFunction(num, den)

# -------- 2. Simulación del sistema ----------
Ts = 0.05
sysd = sys.to_discrete(Ts, method='bilinear')

# ⚡ OPTIMIZACIÓN 1: Menos datos (pero suficientes)
N = 7000 
t = np.arange(N) * Ts
np.random.seed(1)
u = np.random.randn(N)
u = signal.lfilter([1/4, 1/4, 1/4, 1/4], [1.0], u)

out = signal.dlsim(sysd, u, t=t)
y = out[1].flatten()

# Normalización
u_n = (u - np.mean(u)) / np.std(u)
y_n = (y - np.mean(y)) / np.std(y)

# -------- 3. Dataset ----------
N_lags_y = 4
N_lags_u = 1
lag = max(N_lags_y, N_lags_u)
X, Y = [], []
for i in range(lag, N-1):
    entrada = [y_n[i - k] for k in range(1, N_lags_y + 1)] + [u_n[i - k] for k in range(1, N_lags_u + 1)]
    X.append(entrada)
    Y.append(y_n[i])

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# -------- 4. ANFIS ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ANFIS(nn.Module):
    def __init__(self, n_inputs, mfs_per_input):
        super().__init__()
        self.n_inputs = n_inputs
        self.mfs_per_input = [mfs_per_input]*n_inputs
        self.rule_count = int(np.prod(self.mfs_per_input))

        # Rango extendido [-2.5, 2.5] para capturar picos
        self.centers = nn.Parameter(torch.linspace(-2.5, 2.5, mfs_per_input).repeat(n_inputs, 1))
        self.sigmas = nn.Parameter(torch.ones((n_inputs, mfs_per_input)) * 1.0)
        self.beta = nn.Parameter(torch.randn(self.rule_count, n_inputs + 1) * 0.01) # Inicialización más pequeña

        grids = np.meshgrid(*[np.arange(m) for m in self.mfs_per_input], indexing='ij')
        self.register_buffer('rule_indices', torch.tensor(np.stack(grids, axis=-1).reshape(-1, n_inputs), dtype=torch.long))

    def forward(self, x):
        B = x.size(0)
        mus = []
        for i in range(self.n_inputs):
            xi = x[:, i].unsqueeze(1)
            c = self.centers[i].unsqueeze(0)
            s = torch.abs(self.sigmas[i].unsqueeze(0)) + 1e-6
            mus.append(torch.exp(-0.5 * ((xi - c) / s)**2))

        mu_stack = torch.stack([mus[i][:, self.rule_indices[:, i]] for i in range(self.n_inputs)], dim=2)
        firing = torch.prod(mu_stack, dim=2)
        firing_norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-9)

        x_aug = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=1)
        y = torch.sum(firing_norm * (x_aug @ self.beta.T), dim=1, keepdim=True)
        return y

# -------- 5. Entrenamiento ----------
model = ANFIS(X_train.shape[1], mfs_per_input=2).to(device)
# OPTIMIZACIÓN 2: Learning rate mucho más alto (0.01)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

Xtr, Ytr = torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device)
Xv, Yv = torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device)

best_val = 1e9
patience = 30 # OPTIMIZACIÓN 3: Menos paciencia
wait = 0
best_state = model.state_dict()

print("Entrenando rápidamente...")
for ep in range(1000): # Menos épocas máximas necesarias
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr), Ytr)
    loss.backward()
    opt.step()

    if ep % 10 == 0: # Validar cada 10 épocas para acelerar
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 10 # Incrementamos por 10 porque saltamos épocas
            
        if ep % 50 == 0:
            print(f"Epoch {ep}: val MSE = {val_loss:.6f}")
            
        if wait >= patience:
            print(f"Early stopping at epoch {ep}")
            break

model.load_state_dict(best_state)

# -------- 6. Evaluación ----------
Xt, Yt = torch.tensor(X_test).to(device), torch.tensor(Y_test).to(device)
with torch.no_grad(): y_pred = model(Xt).cpu().numpy().flatten()
y_true = Y_test.flatten()

y_pred_real = y_pred * np.std(y) + np.mean(y)
y_true_real = y_true * np.std(y) + np.mean(y)

max_err_pct = 100 * np.max(np.abs(y_true_real - y_pred_real)) / np.max(np.abs(y_true_real))
mse_pct = 100 * np.mean((y_true_real - y_pred_real)**2) / np.var(y_true_real)

print("\n========== RESULTADOS ==========")
print(f"Error máximo relativo: {max_err_pct:.3f} % (< 5%)")
print(f"Error cuadrático medio: {mse_pct:.3f} % (< 2%)")
print("CUMPLE:", "✅" if (max_err_pct <= 5 and mse_pct <= 2) else "❌")

plt.figure(figsize=(10,4))
plt.plot(y_true_real[:300], 'b', label='Real')
plt.plot(y_pred_real[:300], 'r--', label='ANFIS')
plt.legend(); plt.grid(True); plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xanfis import GdAnfisRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ======================================================================
# 4. SCRIPT ANFIS (Versión NARX 2.0) - Seno Completo
# ======================================================================

# --- 1. Generar datos AMBIGUOS (Onda completa) ---
t = np.linspace(0, 1, 1000)
x_orig = np.abs((t * 2) % 2 - 1) # (1 -> 0 -> 1)
y_orig = np.sin(2 * np.pi * t) # (0 -> 1 -> 0 -> -1 -> 0)

# --- 2. Estructurar Datos (NARX - ¡CORREGIDO!) ---
# Modelo: y(t) = f( x(t), y(t-1) )
print("--- Estructurando datos (NARX y(t) = f(x(t), y(t-1))) ---")

df = pd.DataFrame({'x': x_orig, 'y': y_orig})
# Creamos la columna con el dato pasado (la "memoria" correcta)
df['y_t_minus_1'] = df['y'].shift(1) # <-- ¡CAMBIO 1!

# Eliminamos la primera fila que ahora tiene NaN
df = df.dropna()

# Extraemos los datos para ANFIS
# X ahora tiene 2 columnas (2 entradas)
X_data = df[['x', 'y_t_minus_1']].values # <-- ¡CAMBIO 2!
Y_data = df['y'].values.reshape(-1, 1)

# --- 3. Normalización ---
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))
X_norm = scaler_X.fit_transform(X_data)
Y_norm = scaler_Y.fit_transform(Y_data)

# --- 4. Dividir en entrenamiento y prueba (¡SIN BARAJAR!) ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X_norm, Y_norm,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# --- 5. Bucle de Entrenamiento ---
mf_per_input_list = [5]
results = {}

for n_mf in mf_per_input_list:
    num_total_rules = n_mf * n_mf

    print(f"--- Entrenando con {n_mf} MFs/entrada ({num_total_rules} reglas totales) ---")

    model = GdAnfisRegressor(
        num_rules = num_total_rules,
        mf_class = "Gaussian",
        epochs = 400,
        batch_size = 64,
        optim = "Adam",
        optim_params = {"lr": 0.0005},
        verbose = True
    )

    # 6. Entrenar
    model.fit(X_train, Y_train.ravel())

    # 7. Predecir en Test (SOLO para métricas)
    Y_pred_norm_test = model.predict(X_test)
    Y_test_orig = scaler_Y.inverse_transform(Y_test)
    Y_pred_orig_test = scaler_Y.inverse_transform(Y_pred_norm_test.reshape(-1, 1))

    # Guardar resultados
    results[n_mf] = {
        'model': model,
        'Y_pred_orig_test': Y_pred_orig_test,
        'Y_test_orig': Y_test_orig
    }

print("--- Entrenamiento completado ---")

# --- 8. Métricas y Reporte (Basado en el Test Set) ---
print("\n========= REPORTE DE REQUISITOS (SOBRE CONJUNTO DE PRUEBA) ==========")

for n_mf in mf_per_input_list:
    Y_test_orig = results[n_mf]['Y_test_orig']
    Y_pred_orig_test = results[n_mf]['Y_pred_orig_test']

    e = Y_test_orig - Y_pred_orig_test
    mse = mean_squared_error(Y_test_orig, Y_pred_orig_test)

    max_signal_val = np.max(np.abs(Y_test_orig))
    err_max_abs = np.max(np.abs(e))
    err_max_pct = (err_max_abs / max_signal_val) * 100

    print(f"\n--- Resultados con {n_mf} MFs/entrada ({n_mf*n_mf} reglas) ---")
    print(f"  Valor Máximo de Señal (Ref): {max_signal_val:.4f}")
    print(f"  Error Cuadrático Medio (MSE): {mse:.6f} (Req: < 0.02)")
    print(f"  Error Máximo (% de la señal): {err_max_pct:.2f}% (Req: < 5%)")

    if (mse < 0.02 and err_max_pct < 5):
        print("  ESTADO: CUMPLE REQUERIMIENTOS")
    else:
        print("  ESTADO: NO CUMPLE REQUERIMIENTOS")

# --- 9. GRÁFICA DE SIMULACIÓN COMPLETA (¡Lo importante!) ---
print("\n--- Generando gráficas de la simulación completa ---")

plt.figure(figsize=(12, 7))
plt.plot(t[1:], Y_data, 'b-', label='Señal Real Completa (Y)', linewidth=3)

for n_mf in mf_per_input_list:
    model = results[n_mf]['model']
    Y_pred_full_norm = model.predict(X_norm)
    Y_pred_full_orig = scaler_Y.inverse_transform(Y_pred_full_norm.reshape(-1, 1))
    plt.plot(t[1:], Y_pred_full_orig, '--', label=f'Predicción Completa ({n_mf*n_mf} reglas)')

plt.legend()
plt.title("SIMULACIÓN COMPLETA (NARX v2): y(t) = f(x(t), y(t-1))")
plt.xlabel("Tiempo (t)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# --- 10. Graficar Funciones de Pertenencia (Manual) ---
print("\n--- Graficando Funciones de Pertenencia ---")

x_plot = np.linspace(0, 1, 100) # Rango normalizado

for n_mf in mf_per_input_list:
    model = results[n_mf]['model']

    try:
        # Gráfica para Entrada 1: x(t)
        mf_params_e1 = model.mf_parms[0]
        plt.figure(figsize=(10, 4))
        for idx, params in enumerate(mf_params_e1):
            a, b, c = params
            y_plot = np.interp(x_plot, [a, b, c], [0, 1, 0])
            plt.plot(x_plot, y_plot, label=f"MF {idx+1}")
        plt.title(f"MFs para Entrada 1 (x(t)) - Config {n_mf} MFs")
        plt.grid(True)
        plt.show()

        # Gráfica para Entrada 2: y(t-1)
        mf_params_e2 = model.mf_parms[1]
        plt.figure(figsize=(10, 4))
        for idx, params in enumerate(mf_params_e2):
            a, b, c = params
            y_plot = np.interp(x_plot, [a, b, c], [0, 1, 0])
            plt.plot(x_plot, y_plot, label=f"MF {idx+1}")
        plt.title(f"MFs para Entrada 2 (y(t-1)) - Config {n_mf} MFs")
        plt.grid(True)
        plt.show()

    except AttributeError:
        print(f"Error: 'model.mf_parms' no encontrado. No se pueden graficar MFs.")
    except Exception as e:
        print(f"Ocurrió un error al graficar MFs: {e}")
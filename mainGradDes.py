import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D

# --- Configuración Inicial ---
h = 0.1
# np.arange, a diferencia de ':', excluye el valor final.
# Sumamos 'h' para asegurar que '6' esté incluido.
x_vec = np.arange(-6, 6 + h, h)
y_vec = np.arange(-6, 6 + h, h)

# 'n' no se usa en el código original, pero lo mantenemos por fidelidad
n = len(x_vec)
X, Y = np.meshgrid(x_vec, y_vec)

# --- Definición de la función ---
# En NumPy, '**' es el operador de potencia (equivalente a '.^')
F1 = 1.5 - 1.6 * np.exp(-0.05 * ((3 * (X + 3))**2 + (Y + 3)**2))
F = F1 + (0.5 - np.exp(-0.1 * ((3 * (X - 3))**2 + (Y - 3)**2)))

# --- Gradientes ---
# np.gradient devuelve [grad_y, grad_x] (dim 0, dim 1)
# Asignamos correspondientemente a dfy y dfx
dfy, dfx = np.gradient(F, h, h)

# --- Puntos de inicio y colores ---
x0 = [4, 0, -5]
y0 = [0, 2, 5]
col = ['r', 'b', 'm']

# --- Interpoladores (CORREGIDO) ---
# Creamos los interpoladores fuera del bucle para mayor eficiencia
points = (y_vec, x_vec)

# MODIFICACIÓN:
# Añadimos bounds_error=False y fill_value=np.nan
# Esto evita el error y hace que devuelva NaN si se sale de los límites.
interp_F = RegularGridInterpolator(
    points, F, bounds_error=False, fill_value=np.nan)
interp_dfx = RegularGridInterpolator(
    points, dfx, bounds_error=False, fill_value=np.nan)
interp_dfy = RegularGridInterpolator(
    points, dfy, bounds_error=False, fill_value=np.nan)


# Almacenaremos los resultados de cada camino en una lista
path_data = []

# --- Bucle Principal (Descenso de Gradiente) ---
for jj in range(3):

    # Usamos listas de Python para guardar los caminos (paths)
    x_path = [x0[jj]]
    y_path = [y0[jj]]

    # El interpolador espera una lista de puntos [[y, x]]
    # Usamos [0] para extraer el valor escalar del array resultante
    point_start = [[y_path[0], x_path[0]]]
    f_path = [interp_F(point_start)[0]]
    dfxi = interp_dfx(point_start)[0]
    dfyi = interp_dfy(point_start)[0]

    tau = 0.1

    for j in range(1000):
        # Calcular el nuevo punto
        x_new = x_path[j] - tau * dfxi
        y_new = y_path[j] - tau * dfyi

        point_new = [[y_new, x_new]]
        f_new = interp_F(point_new)[0]

        # 1. Comprobar si f_new es NaN (está fuera de límites)
        if np.isnan(f_new):
            x_path.append(x_new)  # Guardamos el último punto (inválido)
            y_path.append(y_new)
            f_path.append(f_new)
            break  # Detener este camino

        # Si es válido, guardar y continuar
        x_path.append(x_new)
        y_path.append(y_new)
        f_path.append(f_new)

        # Calcular nuevos gradientes
        dfxi = interp_dfx(point_new)[0]
        dfyi = interp_dfy(point_new)[0]

        # 2. Comprobar si los gradientes son NaN
        if np.isnan(dfxi) or np.isnan(dfyi):
            break  # No podemos calcular el siguiente paso, detener

        # 3. Condición de convergencia (original)
        if np.abs(f_path[j+1] - f_path[j]) < 1e-6:
            break

    # Guardar los resultados de este camino
    # Convertimos a array de NumPy para graficar más fácil
    path_data.append({
        'x': np.array(x_path),
        'y': np.array(y_path),
        'f': np.array(f_path)
    })

# --- Gráfica de contorno ---
fig1, ax1 = plt.subplots()
# colormap([0 0 0]) significa contornos negros
ax1.contour(X, Y, F, 10, colors='black')

# Graficar cada camino
for i in range(3):
    data = path_data[i]
    color = col[i]
    # Puntos 'o'
    ax1.plot(data['x'], data['y'], marker='o', color=color, linestyle='none')
    # Línea 'k'
    ax1.plot(data['x'], data['y'], color='k', linewidth=2)

ax1.tick_params(labelsize=18)
ax1.set_title('Gráfica de Contorno con Descenso de Gradiente')

# --- Gráfica 3D ---
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# surfl -> plot_surface con cmap='gray' y 'shade=True'
# shading interp -> antialiased=True (comportamiento por defecto)
ax2.plot_surface(X, Y, F, cmap='gray', shade=True, antialiased=True, alpha=0.8)

# Graficar cada camino en 3D
for i in range(3):
    data = path_data[i]
    color = col[i]
    # El '+ 0.1' es para elevar los puntos sobre la superficie
    ax2.plot(data['x'], data['y'], data['f'] + 0.1,
             marker='o', color=color, linestyle='none')
    ax2.plot(data['x'], data['y'], data['f'] + 0.1, color='k', linewidth=2)

ax2.tick_params(labelsize=18)
ax2.set_xlim([-6, 6])
ax2.set_ylim([-6, 6])
# view([-25 60]) -> azim=-25, elev=60
ax2.view_init(elev=60, azim=-25)
ax2.set_title('Gráfica 3D con Descenso de Gradiente')

# Mostrar todas las figuras
plt.show()

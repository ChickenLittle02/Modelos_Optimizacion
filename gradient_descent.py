import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Función objetivo y gradiente
# -----------------------------
def f(x):
    return x[0]**4 - 4*x[0]**3 + 4*x[0] + x[1]**2

def grad_f(x):
    return np.array([4*x[0]**3 - 12*x[0]**2 + 4, 2*x[1]])

# -----------------------------
# Descenso por Gradiente
# -----------------------------
def gradient_descent(x0, lr=0.01, tol=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        x_new = x - lr * g
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(history)

# -----------------------------
# Generación de puntos iniciales
# -----------------------------
def generar_puntos_iniciales():
    puntos = []

    # Región 1: x > 2
    for _ in range(3):
        puntos.append([np.random.uniform(2.1, 5), np.random.uniform(-3, 3)])
        puntos.append([np.random.uniform(2.1, 5), 0])  # incluir y = 0

    # Región 2: x < 0
    for _ in range(3):
        puntos.append([np.random.uniform(-5, -0.1), np.random.uniform(-3, 3)])
        puntos.append([np.random.uniform(-5, -0.1), 0])

    # Región 3: 0 < x < 2
    for _ in range(3):
        puntos.append([np.random.uniform(0.1, 1.9), np.random.uniform(-3, 3)])
        puntos.append([np.random.uniform(0.1, 1.9), 0])

    return puntos

# -----------------------------
# Análisis Multi-LR y CSV
# -----------------------------
def ejecutar_analisis():
    puntos = generar_puntos_iniciales()

    # -------------------------
    # Recomendación sobre los LR
    # -------------------------
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

    resultados = []
    plt.figure(figsize=(10, 7))

    colores = ["blue", "green", "red", "purple", "orange"]

    for lr_idx, lr in enumerate(learning_rates):
        color = colores[lr_idx % len(colores)]

        for x0 in puntos:
            x_final, trayectoria = gradient_descent(x0, lr=lr)

            # Guardar en CSV
            resultados.append({
                "x0": x0,
                "lr": lr,
                "x_final": x_final,
                "f_final": f(x_final),
                "num_iter": len(trayectoria)
            })

            # Graficar trayectoria
            xs = trayectoria[:, 0]
            ys = trayectoria[:, 1]
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1, color=color, alpha=0.6)

    # Guardar CSV
    df = pd.DataFrame(resultados)
    df.to_csv("resultados_gradiente.csv", index=False)
    print("✔ Archivo 'resultados_gradiente.csv' guardado en la carpeta actual.")

    # Decoración del gráfico
    plt.title("Trayectorias del Descenso por Gradiente para distintos LR")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# Ejecutar todo
ejecutar_analisis()

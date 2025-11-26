import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# -----------------------------
# Función objetivo y derivadas
# -----------------------------
def f(x):
    return x[0]**4 - 4*x[0]**3 + 4*x[0] + x[1]**2

def grad_f(x):
    return np.array([4*x[0]**3 - 12*x[0]**2 + 4, 2*x[1]])

def hess_f(x):
    return np.array([[12*x[0]**2 - 24*x[0], 0],
                     [0, 2]])

# -----------------------------
# Descenso por Gradiente
# -----------------------------
def gradient_descent(x0, lr=0.01, tol=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        x_new = x - lr*g
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(history)

# -----------------------------
# Newton
# -----------------------------
def newton_method(x0, tol=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break  # Hessiano singular
        x_new = x - delta
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(history)

# -----------------------------
# Quasi-Newton BFGS usando scipy
# -----------------------------
def quasi_newton_bfgs(x0):
    callback_history = []
    def cb(xk):
        callback_history.append(xk.copy())
    res = minimize(f, x0, method='BFGS', jac=grad_f, callback=cb, options={'gtol':1e-6, 'maxiter':200, 'disp':False})
    hist = [x0] + callback_history
    return res.x, np.array(hist)

# -----------------------------
# Multi-start con 100 puntos iniciales
# -----------------------------
np.random.seed(42)
n_starts = 100
x_range = [-1, 4]  # rango en x
y_range = [-2, 2]  # rango en y

# Guardar todas las trayectorias
trajectories_gd = []
trajectories_newton = []
trajectories_qn = []

for _ in range(n_starts):
    x0 = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
    
    _, hist_gd = gradient_descent(x0, lr=0.01)
    trajectories_gd.append(hist_gd)
    
    _, hist_newton = newton_method(x0)
    trajectories_newton.append(hist_newton)
    
    _, hist_qn = quasi_newton_bfgs(x0)
    trajectories_qn.append(hist_qn)

# -----------------------------
# Graficar contorno y trayectorias
# -----------------------------
x1 = np.linspace(x_range[0]-0.5, x_range[1]+0.5, 200)
x2 = np.linspace(y_range[0]-0.5, y_range[1]+0.5, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = f([X1, X2])

plt.figure(figsize=(10, 7))
plt.contour(X1, X2, Z, levels=30, cmap='coolwarm')

# Función para dibujar trayectorias
def plot_trajectories(trajs, color, label):
    for hist in trajs:
        plt.plot(hist[:,0], hist[:,1], color=color, alpha=0.3)
    plt.plot([], [], color=color, label=label)  # solo para la leyenda

plot_trajectories(trajectories_gd, 'blue', 'Gradiente')
plot_trajectories(trajectories_newton, 'red', 'Newton')
plot_trajectories(trajectories_qn, 'green', 'Quasi-Newton (BFGS)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectorias de optimización - Multi-start')
plt.legend()
plt.grid(True)
# Verificar carpeta actual
print("Directorio actual:", os.getcwd())
plt.savefig('trayectorias.png', dpi=300)
plt.show()
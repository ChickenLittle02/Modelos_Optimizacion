import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**4 - 4*x[0]**3 + 4*x[0] + x[1]**2

def grad_f(x):
    return np.array([4*x[0]**3 - 12*x[0]**2 + 4, 2*x[1]])

# 1. COMPARAR CONVERGENCIA DESDE DIFERENTES PUNTOS INICIALES
initial_points = [
    [3.0, 2.0],    # Punto con gradiente grande
    [0.5, 2.0],    # Punto más suave
    [-1.0, 2.0]    # Puesto en región diferente
]

for x0 in initial_points:
    # Ejecutar ambos algoritmos
    x_gd, hist_gd = gradient_descent_fixed(x0, lr=0.01, max_iter=1000)
    x_arm, hist_arm = gradient_descent_armijo(x0, max_iter=1000)
    
    # Calcular métricas
    f_vals_gd = [f(x) for x in hist_gd]
    f_vals_arm = [f(x) for x in hist_arm]
    
    print(f"\nPunto inicial: {x0}")
    print(f"GD fijo: {len(f_vals_gd)} iteraciones, f final = {f_vals_gd[-1]:.6f}")
    print(f"GD Armijo: {len(f_vals_arm)} iteraciones, f final = {f_vals_arm[-1]:.6f}")
    print(f"Valor mínimo teórico ~ 3.0 (para x1 ≈ 2.0)")

# 2. ANALIZAR LOS PASOS UTILIZADOS POR ARMJIO
def analyze_step_sizes(x0):
    x = np.array(x0, dtype=float)
    steps = []
    gradients = []
    
    for _ in range(50):
        g = grad_f(x)
        gradients.append(np.linalg.norm(g))
        alpha = backtracking_armijo(x, g, f)
        steps.append(alpha)
        x = x - alpha * g
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, 'o-')
    plt.title('Tamaño de paso $\alpha$ por iteración (Armijo)')
    plt.xlabel('Iteración')
    plt.ylabel('$\\alpha$')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(gradients, 's-', color='red')
    plt.title('Norma del gradiente por iteración')
    plt.xlabel('Iteración')
    plt.ylabel('$||\\nabla f||$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. COMPARAR TRAYECTORIAS
def plot_trajectories(x0):
    x_gd, hist_gd = gradient_descent_fixed(x0, lr=0.01, max_iter=200)
    x_arm, hist_arm = gradient_descent_armijo(x0, max_iter=200)
    
    hist_gd = np.array(hist_gd)
    hist_arm = np.array(hist_arm)
    
    plt.figure(figsize=(10, 6))
    
    # Crear contorno de la función
    x1 = np.linspace(-2, 4, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1**4 - 4*X1**3 + 4*X1 + X2**2
    
    plt.contour(X1, X2, Z, levels=50, alpha=0.5)
    
    # Trayectorias
    plt.plot(hist_gd[:, 0], hist_gd[:, 1], 'o-', label='GD tasa fija (lr=0.01)', 
             markersize=4, linewidth=1)
    plt.plot(hist_arm[:, 0], hist_arm[:, 1], 's-', label='GD con Armijo', 
             markersize=4, linewidth=1, alpha=0.7)
    
    plt.scatter([x0[0]], [x0[1]], color='red', s=100, label='Inicio', zorder=5)
    plt.scatter([x_gd[0]], [x_gd[1]], color='blue', s=100, label='Fin GD fijo', zorder=5)
    plt.scatter([x_arm[0]], [x_arm[1]], color='green', s=100, label='Fin GD Armijo', zorder=5)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Comparación de trayectorias de convergencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. ANÁLISIS DE ROBUSTEZ CON DIFERENTES TASAS DE APRENDIZAJE FIJAS
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
x0 = [3.0, 2.0]

results = []
for lr in learning_rates:
    x_gd, hist_gd = gradient_descent_fixed(x0, lr=lr, max_iter=1000)
    f_vals = [f(x) for x in hist_gd]
    results.append({
        'lr': lr,
        'iteraciones': len(hist_gd),
        'f_final': f_vals[-1],
        'converge': f_vals[-1] < 10  # Umbral arbitrario
    })

print("\n=== ROBUSTEZ GD CON TASA FIJA ===")
for r in results:
    print(f"lr={r['lr']:.3f}: {r['iteraciones']} iter, f={r['f_final']:.6f}, converge={'Sí' if r['converge'] else 'No'}")

# 5. ANÁLISIS DE VELOCIDAD DE CONVERGENCIA
def analyze_convergence_speed(x0):
    criterios = [1e-2, 1e-4, 1e-6]
    
    for tol in criterios:
        # GD fijo con mejor lr encontrado
        _, hist_gd = gradient_descent_fixed(x0, lr=0.05, tol=tol, max_iter=5000)
        # GD Armijo
        _, hist_arm = gradient_descent_armijo(x0, tol=tol, max_iter=5000)
        
        print(f"\nTolerancia: {tol}")
        print(f"GD fijo (lr=0.05): {len(hist_gd)} iteraciones para ||∇f|| < {tol}")
        print(f"GD Armijo: {len(hist_arm)} iteraciones para ||∇f|| < {tol}")
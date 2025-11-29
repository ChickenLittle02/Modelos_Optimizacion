def backtracking_armijo(x, g, f, alpha0=1.0, beta=0.5, c=1e-4):
    """Retorna un paso alpha que cumple la condición de Armijo."""
    alpha = alpha0
    fx = f(x)
    g_norm_sq = np.dot(g, g)

    # Condición de Armijo:
    # f(x - alpha*g) <= f(x) - c*alpha*||g||^2
    while f(x - alpha * g) > fx - c * alpha * g_norm_sq:
        alpha *= beta   # reducir paso

    return alpha


def gradient_descent(x0, tol=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for _ in range(max_iter):

        g = grad_f(x)

        # Criterio de parada clásico
        if np.linalg.norm(g) < tol:
            break

        # ⬅⬅ AQUI ENTRA LA BÚSQUEDA EN LÍNEA
        alpha = backtracking_armijo(x, g, f)

        x_new = x - alpha * g
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)

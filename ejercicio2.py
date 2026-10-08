import time
import numpy as np
import matplotlib.pyplot as plt

TAMANOS = [100, 200, 300, 1000]  # tamaños a evaluar (el último es el del enunciado)


def crear_matrices(n):
    # Dos matrices n x n con números aleatorios (misma semilla => mismos datos)
    np.random.seed(42)
    A_np = np.random.rand(n, n)
    B_np = np.random.rand(n, n)
    # Copia en listas de Python para el bucle tradicional
    return A_np, B_np, A_np.tolist(), B_np.tolist()


def multiplicar_bucle(A, B, n):
    # Multiplicación clásica con tres bucles anidados (Python puro)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            suma = 0.0
            for k in range(n):
                suma += A[i][k] * B[k][j]
            C[i][j] = suma
    return C


def multiplicar_numpy(A, B):
    # Multiplicación vectorizada (SIMD) con NumPy
    return np.dot(A, B)


def medir(funcion, *args, repeticiones=5):
    # Ejecuta varias veces y devuelve el resultado y la lista de tiempos
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        resultado = funcion(*args)
        tiempos.append(time.perf_counter() - inicio)
    return resultado, tiempos


def comparar(n, rep_bucle, rep_numpy=5):
    # Mide ambas versiones sobre las mismas matrices
    A_np, B_np, A, B = crear_matrices(n)

    multiplicar_numpy(A_np, B_np)  # calentamiento de NumPy, no se mide

    C_np, t_np = medir(multiplicar_numpy, A_np, B_np, repeticiones=rep_numpy)
    C_bucle, t_bucle = medir(multiplicar_bucle, A, B, n, repeticiones=rep_bucle)

    correcto = np.allclose(np.array(C_bucle), C_np)  # verificación de resultados
    return sum(t_bucle) / len(t_bucle), sum(t_np) / len(t_np), correcto


def graficar(resultados):
    # Gráfico 1: tiempos en escala log-log. Gráfico 2: speedup por tamaño
    ns = [r[0] for r in resultados]
    t_bucle = [r[1] for r in resultados]
    t_np = [r[2] for r in resultados]
    speedups = [b / m for b, m in zip(t_bucle, t_np)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(ns, t_bucle, 'o-', label='Bucle Python')
    ax1.plot(ns, t_np, 's-', label='NumPy')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Tamaño N de la matriz (N x N)')
    ax1.set_ylabel('Tiempo (s)')
    ax1.set_title('Tiempo de multiplicación (escala logarítmica)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    barras = ax2.bar([str(n) for n in ns], speedups)
    ax2.bar_label(barras, fmt='%.0fx')
    ax2.set_xlabel('Tamaño N de la matriz (N x N)')
    ax2.set_ylabel('Speedup (tiempo bucle / tiempo NumPy)')
    ax2.set_title('Speedup de NumPy frente al bucle')

    plt.tight_layout()
    plt.savefig('comparacion_ejercicio2.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    print(f"{'N':>6} | {'Bucle (s)':>12} | {'NumPy (s)':>10} | {'Speedup':>9} | Coincide")
    resultados = []
    for n in TAMANOS:
        if n >= 1000:
            print(f"Calculando N={n} con bucle (puede tardar varios minutos)...", flush=True)
        rep_bucle = 1 if n >= 1000 else 3  # el de 1000 se mide una sola vez
        t_bucle, t_np, ok = comparar(n, rep_bucle)
        resultados.append((n, t_bucle, t_np))
        print(f"{n:>6} | {t_bucle:>12.4f} | {t_np:>10.4f} | {t_bucle / t_np:>8.1f}x | {ok}", flush=True)

    graficar(resultados)
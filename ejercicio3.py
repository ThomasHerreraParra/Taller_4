import threading
import time
import numpy as np

# Tamaño de la matriz y de cada bloque (según el enunciado)
N = 10000
TAM_BLOQUE = 1000


def posiciones_bloques(n, tam):
    # Esquina superior izquierda de cada bloque (100 bloques para 10000/1000)
    return [(f, c) for f in range(0, n, tam) for c in range(0, n, tam)]


def sumar_bloque_puro(matriz, f, c, tam):
    # Suma elemento a elemento con ciclos de Python (sin SIMD)
    total = 0
    for i in range(f, f + tam):
        fila = matriz[i]
        for j in range(c, c + tam):
            total += fila[j]
    return total


def suma_secuencial_pura(matriz, n, tam):
    # Un solo hilo, sin SIMD: recorre los bloques uno tras otro
    total = 0
    for f, c in posiciones_bloques(n, tam):
        total += sumar_bloque_puro(matriz, f, c, tam)
    return total


def sumar_bloque_simd(matriz, f, c, tam):
    # Vista del bloque (no copia datos) y suma de cada fila con NumPy (SIMD)
    bloque = matriz[f:f + tam, c:c + tam]
    sumas_filas = bloque.sum(axis=1, dtype=np.int64)
    return int(sumas_filas.sum())


def tarea_hilo(matriz, f, c, tam, resultado, lock):
    # Cada hilo (SMP) suma su bloque con NumPy (SIMD) y acumula el parcial
    parcial = sumar_bloque_simd(matriz, f, c, tam)
    with lock:  # sección crítica: un solo hilo actualiza el total a la vez
        resultado['total'] += parcial


def suma_hibrida(matriz, n, tam):
    resultado = {'total': 0}  # memoria compartida entre hilos
    lock = threading.Lock()

    # Un hilo por bloque: (10000/1000)^2 = 100 hilos
    hilos = [threading.Thread(target=tarea_hilo,
                              args=(matriz, f, c, tam, resultado, lock))
             for f, c in posiciones_bloques(n, tam)]

    for h in hilos:
        h.start()
    for h in hilos:
        h.join()  # espera a que todos terminen antes de combinar

    return resultado['total'], len(hilos)


def medir(funcion, *args, repeticiones=5):
    # Ejecuta varias veces y devuelve el resultado y la lista de tiempos
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        resultado = funcion(*args)
        tiempos.append(time.perf_counter() - inicio)
    return resultado, tiempos


def promedio(tiempos):
    return sum(tiempos) / len(tiempos)


if __name__ == '__main__':
    np.random.seed(42)
    matriz = np.random.randint(0, 101, size=(N, N), dtype=np.int32)
    esperado = int(matriz.sum(dtype=np.int64))  # referencia, fuera de la medición
    lista = matriz.tolist()  # listas de Python para la versión pura, fuera de la medición

    print(f"Matriz: {N}x{N} | Bloques: {TAM_BLOQUE}x{TAM_BLOQUE} | Bloques/hilos: {len(posiciones_bloques(N, TAM_BLOQUE))}")
    print(f"Suma esperada: {esperado}\n")

    print("Midiendo secuencial en Python puro (puede tardar)...", flush=True)
    total_pura, t_pura = medir(suma_secuencial_pura, lista, N, TAM_BLOQUE, repeticiones=5)
    (total_hib, num_hilos), t_hib = medir(suma_hibrida, matriz, N, TAM_BLOQUE, repeticiones=5)

    print(f"Secuencial puro   -> suma: {total_pura} ({'OK' if total_pura == esperado else 'ERROR'})")
    print(f"Híbrido SMP-SIMD  -> suma: {total_hib} ({'OK' if total_hib == esperado else 'ERROR'})\n")

    print("Ejecución | Sec. puro | Híbrido")
    for i, (a, b) in enumerate(zip(t_pura, t_hib), 1):
        print(f"    {i}     | {a:>9.4f} | {b:>7.4f}")
    print(f"Promedio  | {promedio(t_pura):>9.4f} | {promedio(t_hib):>7.4f}\n")

    speedup = promedio(t_pura) / promedio(t_hib)
    print(f"Speedup (puro / híbrido): {speedup:.1f}x")
    print(f"Reducción de tiempo: {(1 - 1 / speedup) * 100:.2f} %")
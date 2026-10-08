import random
import threading
import time

# Tamaño de la matriz y de cada bloque (según el enunciado)
N = 1000
TAM_BLOQUE = 100


def crear_matriz(n):
    # Matriz n x n con enteros aleatorios entre 0 y 100
    return [[random.randint(0, 100) for _ in range(n)] for _ in range(n)]


def sumar_bloque(matriz, fila_ini, col_ini, tam):
    # Suma los elementos del bloque que empieza en (fila_ini, col_ini)
    total = 0
    for i in range(fila_ini, fila_ini + tam):
        fila = matriz[i]
        for j in range(col_ini, col_ini + tam):
            total += fila[j]
    return total


def posiciones_bloques(n, tam):
    # Esquina superior izquierda de cada bloque (100 bloques para 1000/100)
    return [(f, c) for f in range(0, n, tam) for c in range(0, n, tam)]


def suma_secuencial(matriz, n, tam):
    # Recorre los bloques uno tras otro en un solo hilo
    total = 0
    for f, c in posiciones_bloques(n, tam):
        total += sumar_bloque(matriz, f, c, tam)
    return total


def tarea_hilo(matriz, f, c, tam, resultado, lock):
    # Cada hilo suma su bloque y acumula en la variable compartida
    parcial = sumar_bloque(matriz, f, c, tam)
    with lock:  # sección crítica: un solo hilo actualiza el total a la vez
        resultado['total'] += parcial


def suma_con_hilos(matriz, n, tam):
    resultado = {'total': 0}  # memoria compartida entre hilos
    lock = threading.Lock()

    # Un hilo por bloque
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


if __name__ == '__main__':
    random.seed(42)
    matriz = crear_matriz(N)  # una sola matriz para ambas versiones
    esperado = sum(sum(fila) for fila in matriz)

    print(f"Matriz: {N}x{N} | Bloques: {TAM_BLOQUE}x{TAM_BLOQUE} | Hilos: {len(posiciones_bloques(N, TAM_BLOQUE))}")
    print(f"Suma esperada (sum): {esperado}\n")

    total_sec, t_sec = medir(suma_secuencial, matriz, N, TAM_BLOQUE)
    (total_par, num_hilos), t_par = medir(suma_con_hilos, matriz, N, TAM_BLOQUE)

    prom_sec = sum(t_sec) / len(t_sec)
    prom_par = sum(t_par) / len(t_par)
    speedup = prom_sec / prom_par

    print(f"Secuencial -> suma: {total_sec} ({'OK' if total_sec == esperado else 'ERROR'})")
    print(f"Con hilos  -> suma: {total_par} ({'OK' if total_par == esperado else 'ERROR'})\n")

    print("Ejecución | Secuencial | Con hilos")
    for i, (a, b) in enumerate(zip(t_sec, t_par), 1):
        print(f"    {i}     |   {a:.4f}   |  {b:.4f}")
    print(f"Promedio  |   {prom_sec:.4f}   |  {prom_par:.4f}\n")

    print(f"Speedup (prom. sec / prom. hilos): {speedup:.2f}")
    print(f"Reducción de tiempo: {(1 - 1 / speedup) * 100:.1f} %")
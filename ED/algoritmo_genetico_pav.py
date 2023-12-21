import time
import numpy as np
import random
import matplotlib.pyplot as plt

def distancia(ciudad1, ciudad2):
    return np.linalg.norm(np.array(ciudad1) - np.array(ciudad2))

def calcular_distancia_total(ruta, ciudades):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += distancia(ciudades[ruta[i]], ciudades[ruta[i+1]])
    distancia_total += distancia(ciudades[ruta[-1]], ciudades[ruta[0]])
    return distancia_total

def generar_ruta_aleatoria(ciudades):
    ruta = list(ciudades.keys())
    random.shuffle(ruta)
    return ruta

def cruzar_padres(padre1, padre2):
    punto_corte = random.randint(0, len(padre1) - 1)
    hijo = padre1[:punto_corte] + [ciudad for ciudad in padre2 if ciudad not in padre1[:punto_corte]]
    return hijo

def mutar_ruta(ruta):
    indice1, indice2 = random.sample(range(len(ruta)), 2)
    ruta[indice1], ruta[indice2] = ruta[indice2], ruta[indice1]
    return ruta

def algoritmo_genetico(ciudades, tamano_poblacion, num_generaciones):
    poblacion = [generar_ruta_aleatoria(ciudades) for _ in range(tamano_poblacion)]

    mejores_distancias = []

    for generacion in range(num_generaciones):
        #Ordenamos las rutas generadas de acuerdo a su distancia
        poblacion = sorted(poblacion, key=lambda ruta: calcular_distancia_total(ruta, ciudades))

        # Guardamos la mejor distancia generada de cada generacion
        mejores_distancias.append(calcular_distancia_total(poblacion[0], ciudades))

        # Selección de los mejores padres de acuerdo a las 10 mejores distancias
        elite_size = int(tamano_poblacion * 0.2)
        padres = poblacion[:elite_size]

        # Cruzamiento y mutación para crear la nueva generación
        descendencia = []
        while len(descendencia) < (tamano_poblacion - elite_size):
            padre1, padre2 = random.sample(padres, 2)
            hijo = cruzar_padres(padre1, padre2)
            if random.random() < 0.1:  # Establecemos una probabilidad de mutacion
                hijo = mutar_ruta(hijo)
            descendencia.append(hijo)

        # Generamos una nueva poblacion (padres  + descendencia)
        poblacion = padres + descendencia

    mejor_ruta = poblacion[0]
    mejor_distancia = calcular_distancia_total(mejor_ruta, ciudades)
    return mejor_ruta, mejor_distancia, mejores_distancias


def visualizar_ruta(ciudades, ruta):
    coordenadas_x = [ciudades[ciudad][0] for ciudad in ruta]
    coordenadas_y = [ciudades[ciudad][1] for ciudad in ruta]

    # Agregar el punto de inicio al final para cerrar el ciclo
    coordenadas_x.append(coordenadas_x[0])
    coordenadas_y.append(coordenadas_y[0])
    print(coordenadas_x[0])
    print(coordenadas_y[0])
    plt.figure(figsize=(8, 8))
    
    plt.plot(coordenadas_x, coordenadas_y, marker='o', linestyle='-', color='b')
    plt.scatter(coordenadas_x[0], coordenadas_y[0], color='r', marker='x', label='Inicio')
    
    plt.title('Ruta Óptima')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Establecemos ciudades
ciudades = {
    'A': [0, 0],
    'B': [1, 2],
    'C': [3, 1],
    'D': [5, 2],
    'E': [3, 4],
    'F': [8, 7],
    'G': [6, 5],
    'H': [1, 8],
    'I': [3, 10],
    'J': [7, 9 ],
    'K': [0, 1],
    'L': [5, 6],
    'M': [3, 9],
    'N': [5, 10],
    'O': [3, 11],
    'P': [8, 2],
    'Q': [6, 1],
    'R': [1, 5],
    'S': [2, 6],
    'T': [4, 7 ],
}

tamano_poblacion = 50
num_generaciones = 500

start_time = time.time()
ruta_optima, distancia_optima,mejores_distancias  = algoritmo_genetico(ciudades, tamano_poblacion, num_generaciones)
end_time = time.time()

print("Ruta óptima:", ruta_optima)
print("Distancia óptima:", distancia_optima)
print("Tiempo estimado:", end_time - start_time)

# Evolucion del algoritmos por generacion
plt.plot(mejores_distancias, linestyle='-', marker='o', color='g')
plt.title('Evolución de la Mejor Distancia')
plt.xlabel('Generación')
plt.ylabel('Distancia')
plt.grid(True)
plt.show()

# Visualizar la ruta óptima
visualizar_ruta(ciudades, ruta_optima)
import numpy as np
import cv2
import time

import pyoints as pyt

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

### Função para reduzir a quantidade dos pontos do point cloud
def filtrarPC(A, B):
    r = 0.004
    A = A[list(pyt.filters.ball(pyt.indexkd.IndexKD(A.coords), r))]
    B = B[list(pyt.filters.ball(pyt.indexkd.IndexKD(B.coords), r))]
    return A, B

### Função para pegar os menores e maiores limites de acordo com o maior ou menor de A e B
def acharLimites(coords_dict):
    coordsMax = []
    coordsMin = []
    coordsMaxA = coords_dict['A'].max(axis=0)
    coordsMinA = coords_dict['A'].min(axis=0)
    coordsMaxB = coords_dict['B'].max(axis=0)
    coordsMinB = coords_dict['B'].min(axis=0)
    for lim in range(3):
        if(coordsMaxA[lim] > coordsMaxB[lim]):
            coordsMax.append(coordsMaxA[lim])
        else:
            coordsMax.append(coordsMaxB[lim])

        if(coordsMinA[lim] < coordsMinB[lim]):
            coordsMin.append(coordsMinA[lim])
        else:
            coordsMin.append(coordsMinB[lim])

    coordsMax = np.array(coordsMax)
    coordsMin = np.array(coordsMin)
    return coordsMax, coordsMin

### Função para plotar o gráfico 3d dos point clouds, de acordo com os seus limites
def plotarGraf3d(coords_dict, coordsMin, coordsMax):
    # Cores e limites de eixos
    colors = {'A': 'red', 'B': 'blue'}
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_xlim(coordsMin[0], coordsMax[0])
    ax.set_ylim(coordsMin[1], coordsMax[1])
    ax.set_zlim(coordsMin[2], coordsMax[2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.scatter(*coords_dict['A'].T, color=colors['A'])
    ax.scatter(*coords_dict['B'].T, color=colors['B'])
    plt.show()

### Função para comparar dois point clouds
def compararPC(cloud1, cloud2):
    A = pyt.storage.PlyHandler.loadPly(cloud1)
    B = pyt.storage.PlyHandler.loadPly(cloud2)

    ### Reduzir quantidade de pontos de cada point cloud
    A, B = filtrarPC(A, B)

    coords_dict = {
    'A': A.coords,
    'B': B.coords,
    }

    ### Junto com os prints parecidos mais abaixo no código, comprovam que mesmo depois de algoritmo de ICP a quantidade de pontos de cada point cloud é o mesmo
    #print("A antes: {}".format(coords_dict['A'].shape))
    #print("B antes: {}".format(coords_dict['B'].shape))

    ### Definir limites de acordo com as coordenadas de A e B
    coordsMax, coordsMin = acharLimites(coords_dict)

    ### Plotar gráfico 3d dos pontos iniciais de A e B, sem ICP
    #plotarGraf3d(coords_dict, coordsMin, coordsMax)


    ### Algoritmo de ICP
    d_th = 0.04
    radii = [d_th, d_th, d_th]
    icp = pyt.registration.ICP(
        radii,
        max_iter=100,
        max_change_ratio=0.000001,
        k=1
    )

    print("Processo do algoritmo de ICP começando")
    start = time.time()
    T_dict, pairs_dict, report = icp(coords_dict)
    end = time.time()
    print("Processo do algoritmo de ICP terminado em {} segundos".format(end - start))

    ### Continuação de cima, provando que após ICP tamanho dos point clouds se mantém
    #print("A icp: {}".format(coords_dict['A'].shape))
    #print("B icp: {}".format(coords_dict['B'].shape))


    ### Plot gráficos 2d dos supostos erros do pairs_dict
    x = []
    for i in range(1, pairs_dict['B']['A'][1].size + 1):
        x.append(i)
    y = pairs_dict['B']['A'][1]
    plt.plot(x,y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("B: A")
    plt.show()

    x = []
    for i in range(1, pairs_dict['A']['B'][1].size + 1):
        x.append(i)
    y = pairs_dict['A']['B'][1]
    plt.plot(x,y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("A: B")
    plt.show()

    ### Plotar o gráfico 3d depois que o algoritmo de ICP foi rodado
    coords_dict = {
    'A': pyt.transformation.transform(coords_dict['A'], T_dict['A']),
    'B': pyt.transformation.transform(coords_dict['B'], T_dict['B']),
    }

    coords_TESTE = {}
    coords_TESTE['A'] = pyt.transformation.transform(coords_dict['A'], T_dict['A'])
    coords_TESTE['B'] = pyt.transformation.transform(coords_dict['B'], T_dict['B'])

    coordsMax, coordsMin = acharLimites(coords_dict)
    plotarGraf3d(coords_dict, coordsMin, coordsMax)
     
    ### Cria o numpy.ndarray do x e y das coordenadas com pointcloud, com o z como o erro de cada ponto
    coords_errosA = np.empty((0,3))
    for coordA in range (coords_dict['A'].shape[0]):
        coords_errosA = np.append(coords_errosA, np.array([[coords_dict['A'][coordA][0], coords_dict['A'][coordA][1], pairs_dict['B']['A'][1][coordA] - 1]]), axis=0)

    coords_errosB = np.empty((0,3))
    for coordB in range (coords_dict['B'].shape[0]):
        coords_errosB = np.append(coords_errosB, np.array([[coords_dict['B'][coordB][0], coords_dict['B'][coordB][1], pairs_dict['A']['B'][1][coordB] - 1]]), axis=0)

    coords_dict = {
    'A': coords_errosA,
    'B': coords_errosB,
    }


    coords_TESTE['AB'] = coords_errosB
    coords_TESTE['BA'] = coords_errosA

    coordsMax, coordsMin = acharLimites(coords_dict)
    plotarGraf3d(coords_dict, coordsMin, coordsMax)


    ### TESTES

    # Cores e limites de eixos
    colors = {'A': 'red', 'B': 'blue', 'AB': 'yellow', 'BA': 'black'}
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    ax.set_xlim(coordsMin[0], coordsMax[0])
    ax.set_ylim(coordsMin[1], coordsMax[1])
    ax.set_zlim(-0.755, 0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.scatter(*coords_TESTE['A'].T, color=colors['A'])
    ax.scatter(*coords_TESTE['B'].T, color=colors['B'])
    ax.scatter(*coords_TESTE['AB'].T, color=colors['AB'])
    ax.scatter(*coords_TESTE['BA'].T, color=colors['BA'])
    plt.show()


   # errosAB = []
   # for erro in coords_dict['B']:
   #     if(erro[2] < 0.961):
   #         errosAB.append(erro)
    
    #print(errosAB[0])

    ### Plotar o gráfico do RMSE existente ao longo das iterações do processo de ICP
    #fig = plt.figure(figsize=(15, 8))
    #plt.xlim(0, len(report['RMSE']) + 1)
    #plt.xlabel('Iteration')
    #plt.ylabel('RMSE')

    #plt.bar(np.arange(len(report['RMSE']))+1, report['RMSE'], color='gray')
    #plt.show()
    

def main():
    while True:
        # Pegar informações dois dois point clouds a serem comparados
        numPC = input("Insira o número do primeiro point cloud para comparação:\n")
        cloud1 = "pointclouds/processados/pointcloud{}.ply".format(numPC)
        numPC = input("Insira o número do segundo point cloud para comparação:\n")
        cloud2 = "pointclouds/processados/pointcloud{}.ply".format(numPC) 
        # Função para fazer a comparação
        compararPC(cloud1, cloud2)
        

if __name__ == "__main__":
    main()
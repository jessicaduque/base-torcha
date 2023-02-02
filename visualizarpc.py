import numpy as np
import open3d as o3d 
import cv2
import math
import itertools

def retirarChao(cloud):
    pontos = np.asarray(cloud.points)  
    cloud = cloud.select_by_index(np.where(pontos[:, 2] > -0.52)[0])
    return cloud

def retirarPontosAleatorios(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = cloud.select_by_index(ind)
    return inlier_cloud

def salvarProcessado(cloud, n):
    o3d.io.write_point_cloud("pointclouds/processados/pointcloud33_2.ply".format(n), cloud)

def rotacionarPC(cloud):
    o3d.visualization.draw_geometries_with_editing([cloud]) # Visualize the point cloud
    R = cloud.get_rotation_matrix_from_xyz((0, 0, 3))
    cloud = cloud.rotate(R, center=(0,0,0))
    return cloud

def main():
    while True:
        # Pegar informação do point cloud a ser manipulado
        numPC = int(input("Qual o número do point cloud a ser manipulado?\n"))
        cloud = o3d.io.read_point_cloud("pointclouds/pointcloud{}.ply".format(numPC)) # Read the point cloud    
        
        # Retirar o chão do point cloud
        #cloud = retirarChao(cloud)

        # Retirar pontos muito isolados do point cloud
        cloud = retirarPontosAleatorios(cloud)
        
        # Rotacionar point cloud
        cloud = rotacionarPC(cloud)

        o3d.visualization.draw_geometries_with_editing([cloud]) # Visualize the point cloud 
        salvar = input("Você gostaria de salvar o point cloud processado? (s / n)\n")
        if(salvar == "s"):
            salvarProcessado(cloud, numPC)
            print("Point cloud salvo com sucesso")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
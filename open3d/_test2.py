# import open3d as o3d
# import numpy as np
# import os
# import sys

# def load_images(folder_path):
#     images = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img_path = os.path.join(folder_path, filename)
#             images.append(img_path)
#     return images

# def create_point_cloud(images):
#     # Questo esempio è semplificato e richiede ulteriori passaggi per la creazione di un modello 3D preciso
#     # Utilizza feature detection e matching per trovare punti comuni tra le immagini
#     # Poi, utilizza la triangolazione per creare una nuvola di punti 3D
    
#     # Per semplicità, supponiamo di avere già una nuvola di punti
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))  # Esempio di nuvola di punti casuali
#     return pcd

# def main(folder_path, output_file):
#     images = load_images(folder_path)
#     pcd = create_point_cloud(images)
    
#     # Visualizza la nuvola di punti
#     # o3d.visualization.draw_geometries([pcd])

#     # Salva la nuvola di punti in un file
#     o3d.io.write_point_cloud(output_file, pcd)


# if __name__ == "__main__":
#     input_folder = sys.argv[1]
#     output_file = sys.argv[2]

#     main(input_folder, output_file)
import open3d as o3d
import os
import numpy as np
import re

def create_3d_model_from_images(input_folder):
    """
    Crea un modello 3D (nuvola di punti e mesh) da immagini RGB e di profondità.

    Args:
        input_folder (str): Il percorso della cartella contenente le immagini.
                            Le immagini a colori devono avere il formato *_subject.estensione,
                            e le immagini di profondità devono avere il formato *_depth.estensione.
                            Es. image01_subject.jpg e image01_depth.png
    """
    rgbd_images = []
    camera_poses = []
    
    # 1. **Lettura delle immagini RGB-D e creazione degli oggetti RGBDImage**
    for filename in os.listdir(input_folder):
        if filename.endswith("_subject.jpg") or filename.endswith("_subject.png"):  # Adatta le estensioni se necessario
            base_name = filename.replace("_subject.jpg","").replace("_subject.png","")
            color_path = os.path.join(input_folder, filename)
            depth_path = os.path.join(input_folder, base_name + "_depth.png") # Adatta l'estensione se necessario

            if os.path.exists(depth_path):
                color_image = o3d.io.read_image(color_path)
                depth_image = o3d.io.read_image(depth_path)
            
                # **Creazione dell'oggetto RGBDImage**
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, depth_image, convert_rgb_to_intensity=False
                )
                rgbd_images.append(rgbd_image)
            
                # Inizializzazione della matrice di trasformazione (pose della camera).
                # Questo è un punto di partenza, è necessario stimare le vere pose della camera
                # usando tecniche SLAM o SfM.
                camera_poses.append(np.identity(4)) # Imposta la matrice di identità come posa iniziale.
            else:
                print(f"Attenzione: Immagine di profondità non trovata per {filename}")

    if not rgbd_images:
        print("Nessuna immagine RGB-D trovata.")
        return

    # 2. **Calibrazione della Camera** (esempio con parametri di default)
    # È importante sostituire questi valori con quelli corretti della tua fotocamera.
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgbd_images.color.get_width(),
        height=rgbd_images.color.get_height(),
        fx=500,  # Lunghezza focale x
        fy=500,  # Lunghezza focale y
        cx=rgbd_images.color.get_width() / 2, # Centro x
        cy=rgbd_images.color.get_height() / 2  # Centro y
    )

    # 3. **Ricostruzione della Nuvola di Punti**
    point_clouds = []
    for i, rgbd_image in enumerate(rgbd_images):
        # Creazione della nuvola di punti
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsic
        )
        # Trasformazione della nuvola di punti in base alla posa della camera
        pcd.transform(camera_poses[i])
        point_clouds.append(pcd)
    
    # Unione di tutte le nuvole di punti
    merged_point_cloud = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        merged_point_cloud += pcd

    # 4. **Ricostruzione della Mesh**
    # Algoritmo di Ball-Pivoting per la ricostruzione della mesh.
    # Potrebbe essere necessario ottimizzare il parametro di raggio.
    # Potresti provare altri metodi (es. surface reconstruction basata su ottimizzazione).
    distances = merged_point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist  # Adatta questo valore in base alla qualità del tuo input
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        merged_point_cloud,
        o3d.utility.DoubleVector([radius, radius * 2])
        )
    mesh.compute_vertex_normals()

    # 5. **Texturing della Mesh**
    # Questo è un esempio semplificato. 
    # La texturing corretta richiederebbe una proiezione più accurata e un uv mapping.
    # Questo esempio usa il colore medio della nuvola di punti come colore della mesh.
    if merged_point_cloud.has_colors():
        mesh.paint_uniform_color(np.mean(np.asarray(merged_point_cloud.colors), axis=0))

    # Salvataggio del modello 3D
    output_path = os.path.join(input_folder, "output_model.ply")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Modello 3D salvato in {output_path}")

    # Visualizzazione del modello 3D
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    input_directory = "/input/HEIC" # Modifica questo percorso se necessario
    create_3d_model_from_images(input_directory)
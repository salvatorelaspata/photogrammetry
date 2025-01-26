import open3d as o3d
import numpy as np
import os
import sys
def create_3d_model_from_images(input_folder, output_file):
    # Carica le immagini e crea una lista di RGBD images
    rgbd_images = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            image = o3d.io.read_image(img_path)
            # Supponiamo che tu abbia anche la profondità (placeholder)
            depth = o3d.geometry.Image(np.zeros(image.shape))  # Sostituisci con l'immagine di profondità reale
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth)
            rgbd_images.append(rgbd_image)

    # Creazione della nuvola di punti da RGBD images (placeholder)
    pcd = o3d.geometry.PointCloud()
    for rgbd in rgbd_images:
        # Qui dovresti implementare la logica per convertire RGBD in PointCloud
        pass  # Placeholder

    # Salva il modello 3D in un file
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Modello 3D salvato in: {output_file}")


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    create_3d_model_from_images(input_folder, output_file)
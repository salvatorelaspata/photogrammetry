import open3d as o3d
import numpy as np
import cv2
import os
import sys


def create_point_cloud(rgb_path, depth_path, output_path, focal_length=500):
    """
    Crea una nuvola di punti da un'immagine RGB e una mappa di profondità.

    Args:
        rgb_path (str): Percorso dell'immagine RGB
        depth_path (str): Percorso della mappa di profondità
        output_path (str): Percorso di output per la nuvola di punti (.ply)
        focal_length (float): Lunghezza focale approssimativa della fotocamera
    """
    # Carica le immagini
    # Carica le immagini con OpenCV
    color_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Converti in formato Open3D
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    # Verifica dimensioni
    if color_image.shape[:2] != depth_image.shape:
        raise ValueError(
            f"Dimensioni mismatch: RGB {color_image.shape[:2]} vs Depth {depth_image.shape}"
        )

    # Crea oggetto RGBDImage
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    # Calcola parametri della fotocamera
    height, width = np.asarray(color_image).shape[:2]
    cx = width / 2
    cy = height / 2

    # Crea point cloud
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=focal_length, fy=focal_length, cx=cx, cy=cy
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # Salva la nuvola di punti
    o3d.io.write_point_cloud(output_path, pcd)
    return pcd


def process_directory(rgb_dir, depth_dir, output_dir):
    """
    Elabora tutte le immagini in una directory generando nuvole di punti

    Args:
        rgb_dir (str): Directory con immagini RGB
        depth_dir (str): Directory con mappe di profondità
        output_dir (str): Directory di output per le nuvole di punti
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_files = [
        f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for rgb_file in rgb_files:
        try:
            depth_file = os.path.splitext(rgb_file)[0] + ".png"
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, depth_file)

            # Verifica esistenza file
            if not os.path.exists(depth_path):
                print(f"File depth mancante: {depth_path}")
                continue

            output_path = os.path.join(
                output_dir, os.path.splitext(rgb_file)[0] + ".ply"
            )

            create_point_cloud(rgb_path, depth_path, output_path)
            print(f"Generata nuvola di punti: {output_path}")

        except Exception as e:
            print(f"Errore elaborando {rgb_file}: {str(e)}")


if __name__ == "__main__":
    process_directory(sys.argv[1], sys.argv[2], sys.argv[3])

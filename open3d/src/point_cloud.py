import os
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import sys

def load_rgbd_image(color_path, depth_path, depth_scale=1000.0, depth_trunc=10.0):
    """
    Carica un'immagine RGBD con pre-elaborazione accurata.
    
    Args:
        color_path (str): Percorso dell'immagine a colori
        depth_path (str): Percorso della mappa di profondità
        depth_scale (float): Fattore di scala per la profondità
        depth_trunc (float): Valore massimo per troncare la profondità
    
    Returns:
        o3d.geometry.RGBDImage: Immagine RGBD elaborata
        (int, int): Dimensioni dell'immagine (altezza, larghezza)
    """
    # Carica le immagini
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    
    # Ridimensiona le immagini (opzionale, per ridurre carico computazionale)
    scale_factor = 0.25  # Riduzione al 25% della dimensione originale
    color_raw = cv2.resize(np.asarray(color_raw), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    depth_raw = cv2.resize(np.asarray(depth_raw), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Converti in float32 e normalizza
    depth_raw = depth_raw.astype(np.float32) / 255.0 * depth_scale
    
    # Converti in Open3D Image
    color_o3d = o3d.geometry.Image(color_raw)
    depth_o3d = o3d.geometry.Image(depth_raw)
    
    # Crea RGBD Image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d, 
        depth_scale=1.0,  # Già scalata precedentemente
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    
    return rgbd_image, color_raw.shape[:2]

def create_point_cloud(image_dir, depth_dir, output_path, depth_scale=500.0):
    """
    Crea una nuvola di punti da immagini e mappe di profondità.
    
    Args:
        image_dir (str): Cartella contenente le immagini
        depth_dir (str): Cartella contenente le mappe di profondità
        output_path (str): Percorso di output per la nuvola di punti
        depth_scale (float): Fattore di scala per la profondità
    
    Returns:
        o3d.geometry.PointCloud: Nuvola di punti finale
    """
    # Ordina i file per garantire corrispondenza
    image_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith((".png")) and "_subject" in f])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith((".png")) and "_depth" in f])
    
    if len(image_files) != len(depth_files):
        print("Errore: numero di immagini e mappe di profondità non corrispondente")
        return None
    
    point_clouds = []
    
    for image_file, depth_file in zip(image_files, depth_files):
        image_path = os.path.join(image_dir, image_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        try:
            # Carica l'immagine RGBD
            rgbd_image, (height, width) = load_rgbd_image(image_path, depth_path, depth_scale)
            
            # Crea la camera intrinseca
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=width * 0.8,   # Stima del fuoco
                fy=height * 0.8,  # Stima del fuoco
                cx=width / 2,
                cy=height / 2
            )
            
            # Crea la point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, 
                camera_intrinsic
            )
            
            # Trasforma la point cloud (opzionale)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            point_clouds.append(pcd)
            print(f"Point cloud generata per {image_file}")
        
        except Exception as e:
            print(f"Errore elaborando {image_file}: {e}")
    
    # Combina le point cloud
    if point_clouds:
        final_pcd = point_clouds[0]
        for pcd in point_clouds[1:]:
            final_pcd += pcd
        
        # Rimozione outliers
        final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Salvataggio
        o3d.io.write_point_cloud(output_path, final_pcd)
        print(f"Nuvola di punti salvata in {output_path}")
        
        return final_pcd
    
    return None

def generate_mesh(point_cloud_path, output_mesh_path):
    """
    Genera una mesh da una nuvola di punti.
    
    Args:
        point_cloud_path (str): Percorso della nuvola di punti in input
        output_mesh_path (str): Percorso di output per la mesh 3D
    
    Returns:
        o3d.geometry.TriangleMesh: Mesh 3D generata
    """
    # Carica la point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    # Stima e orientamento delle normali
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Ricostruzione mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=9, width=0, scale=1.1, linear_fit=True
    )
    
    # Pulizia mesh
    mesh.remove_duplicate_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    
    # Normalizzazione
    mesh.normalize_normals()
    mesh.paint_uniform_color([1, 0.706, 0])  # Colore arancione
    
    # Salvataggio
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"Mesh salvata in {output_mesh_path}")
    
    return mesh

if __name__ == "__main__":
  try:
      image_directory = sys.argv[1] # "./input/HEIC_TO_JPG"  
      depth_directory = sys.argv[2] # "./input/JPG_TO_PNG" 
      output_point_cloud = sys.argv[3] # "./output/point_cloud2.ply"
      output_mesh = sys.argv[4] # "./output/mesh.ply"
      
      # Crea la point cloud
      point_cloud = create_point_cloud(image_directory, depth_directory, output_point_cloud)
      
      # Genera la mesh
      # if point_cloud is not None:
      #     mesh = generate_mesh(output_point_cloud, output_mesh)
      
  except FileNotFoundError as e:
      print(f"Errore: file non trovato - {e}")
  except ValueError as e:
      print(f"Errore di elaborazione: {e}")
  except Exception as e:
      print(f"Errore imprevisto durante l'elaborazione: {e}")
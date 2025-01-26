import os
import open3d as o3d
import sys

def process_images(input_folder, output_folder):
    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            # Carica l'immagine (puoi elaborarla come desideri)
            image = o3d.io.read_image(img_path)
        
            # Esempio: salva l'immagine elaborata nella cartella di output
            output_path = os.path.join(output_folder, f"processed_{filename}")
            o3d.io.write_image(output_path, image)

if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_images(input_folder, output_folder)

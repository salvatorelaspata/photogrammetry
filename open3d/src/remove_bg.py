import os
import sys
import logging
from rembg import remove
from PIL import Image
import cv2


logger = logging.getLogger(__name__)


def remove_bg_from_directory(input_dir, output_dir):
    """
    Processa le immagini rimuovendo lo sfondo e generando mappe di profondità.

    Args:
        input_dir (str): Percorso della directory con le immagini di input
        output_dir (str): Percorso della directory per le immagini senza sfondo
    """

    # verifica se la directory di output esiste
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory {output_dir} non trovata")

    image_files = [
        f
        for f in os.listdir(input_dir)  # <-- input_dir è original_jpg
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # remove background
    i = 1
    for image_file in image_files:
        try:

            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file + ".png")
            # logger.info(f"Input path {input_path}")
            # logger.info(f"Output path {output_path}")
            # Carica l'immagine
            input_image = Image.open(input_path)
            logger.info(f"Processing {input_path}")
            # Rimuovi lo sfondo
            output = remove(input_image)
            logger.info(f"Background removed in {output_path}")
            # Salva le immagini
            output.save(output_path)
            logger.info(f"Remove bg: {i}/{len(image_files)}: {image_file}")
            i += 1
            # Libera memoria
            del input_image, output
        except Exception as e:
            logger.info(f"Error removing background using PIL {input_path}: {e}")
            # try:
            #     cv2_image = cv2.imread(input_path)
            #     logger.info(f"Processing {input_path}")
            #     output = remove(cv2_image)
            #     cv2.imwrite(output_path, output)
            #     logger.info(f"Background removed {input_path}")
            # except Exception as e:
            #     logger.info(f"Error removing background using cv2 {input_path}: {e}")


if __name__ == "__main__":
    remove_bg_from_directory(sys.argv[1], sys.argv[2])

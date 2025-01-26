from PIL import Image
import pillow_heif
import os
import sys

def heic_to_jpg(input_path, output_path):
  """Converte un'immagine HEIC in JPG.

  Args:
    input_path: Il percorso del file HEIC.
    output_path: Il percorso del file JPG di output.
  """
  try:
    heif_file = pillow_heif.read_heif(input_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    )
    image.save(output_path, "JPEG")
    print(f"Convertito {input_path} in {output_path}")
  except Exception as e:
    print(f"Errore durante la conversione di {input_path}: {e}")


def convert_heic_files_in_directory(input_dir, output_dir):
    """Converte tutti i file HEIC in una directory in JPG.

    Args:
        input_dir: La directory contenente i file HEIC.
        output_dir: La directory in cui salvare i file JPG.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".heic"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_dir, output_filename)
            heic_to_jpg(input_path, output_path)

if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    convert_heic_files_in_directory(input_folder, output_folder)
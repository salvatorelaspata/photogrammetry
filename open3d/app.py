from flask import Flask, request, jsonify
import os
import time
import psutil
import logging
from datetime import datetime

app = Flask(__name__)

import src.heic_to_jpg as heic_to_jpg
import src.remove_bg as remove_bg
import src.depth_map as depth_map

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione sistema
UPLOAD_FOLDER = "input"
OUTPUT_FOLDER = "output"
PROCESSED_FOLDERS = {
    "original_jpg": "original_jpg",
    "no_background": "no_background",
    "depth_map": "depth_map",
}


@app.route("/upload", methods=["POST"])
def upload_images():
    if "images" not in request.files:
        return jsonify({"error": "Nessun file trovato"}), 400

    files = request.files.getlist("images")
    if len(files) == 0:
        return jsonify({"error": "Nessuna immagine fornita"}), 400

    # Crea job
    job_id = str(time.time())
    # folder input
    folder_input = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(folder_input, exist_ok=True)
    # folder output
    folder_output = os.path.join(OUTPUT_FOLDER, job_id)
    os.makedirs(folder_output, exist_ok=True)
    # folder original_jpg
    folder_original_jpg = os.path.join(folder_output, PROCESSED_FOLDERS["original_jpg"])
    os.makedirs(folder_original_jpg, exist_ok=True)
    # folder no_background
    folder_no_background = os.path.join(
        folder_output, PROCESSED_FOLDERS["no_background"]
    )
    os.makedirs(folder_no_background, exist_ok=True)
    # folder depth_map
    folder_depth_map = os.path.join(folder_output, PROCESSED_FOLDERS["depth_map"])
    os.makedirs(folder_depth_map, exist_ok=True)

    # Salva le immagini in input
    image_paths = []
    for file in files:
        if file.filename == "":
            continue
        path = os.path.join(folder_input, file.filename)
        file.save(path)
        image_paths.append(path)

    # Avvia il processo
    logger.info(f"Avvio job {job_id}")
    heic_to_jpg.convert_heic_files_in_directory(folder_input, folder_original_jpg)
    logger.info("convert heic files in directory")
    remove_bg.remove_bg_from_directory(folder_original_jpg, folder_no_background)
    logger.info("remove bg from directory")
    # time execitopm
    depth_map.process_images(folder_original_jpg, folder_depth_map)
    logger.info("process images")

    return (
        jsonify(
            {
                "message": "Job accettato",
                "job_id": job_id,
                # "status": job.status,
                # "queue_position": job.queue_position,
                "num_images": len(image_paths),
            }
        ),
        202,
    )


@app.route("/system-status", methods=["GET"])
def get_system_status():
    return jsonify(
        {
            # "active_jobs": len(job_queue.active_jobs),
            # "queued_jobs": job_queue.queued_jobs.qsize(),
            # "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            # "max_queued_jobs": MAX_QUEUED_JOBS,
            # "workers_per_job": NUM_WORKERS,
            "system_cpu": psutil.cpu_percent(),
            "system_memory": psutil.virtual_memory().percent,
            "system_time": datetime.now().isoformat(),
            # "queue": [asdict(job) for job in list(job_queue.queued_jobs.queue)],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

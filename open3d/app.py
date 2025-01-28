from flask import Flask, request, jsonify
import os
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from queue import Queue, Full
import psutil
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

app = Flask(__name__)

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione sistema
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
MAX_CONCURRENT_JOBS = 2  # Numero massimo di job contemporanei
MAX_QUEUED_JOBS = 5     # Numero massimo di job in coda
NUM_WORKERS = max(1, cpu_count() - 1)  # Worker per job

@dataclass
class JobStatus:
    """Classe per tracciare lo stato del job"""
    job_id: str
    status: str
    start_time: float
    num_images: int
    folder: str
    output_folder: str
    queue_position: Optional[int] = None
    completion_time: Optional[float] = None
    error: Optional[str] = None
    progress: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class JobQueue:
    """Gestisce la coda dei job e il loro stato"""
    def __init__(self, max_concurrent: int, max_queued: int):
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.active_jobs: Dict[str, JobStatus] = {}
        self.queued_jobs: Queue = Queue(maxsize=max_queued)
        self.lock = threading.Lock()
        
        # Avvia il thread di monitoraggio risorse
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()

    def add_job(self, job: JobStatus) -> bool:
        """Aggiunge un nuovo job alla coda"""
        with self.lock:
            if len(self.active_jobs) < self.max_concurrent:
                self.active_jobs[job.job_id] = job
                return True
            try:
                self.queued_jobs.put_nowait(job)
                job.queue_position = self.queued_jobs.qsize()
                return True
            except Full:
                return False

    def job_completed(self, job_id: str):
        """Gestisce il completamento di un job e avvia il successivo"""
        with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                
                # Avvia il prossimo job in coda se presente
                if not self.queued_jobs.empty():
                    next_job = self.queued_jobs.get()
                    self.active_jobs[next_job.job_id] = next_job
                    
                    # Aggiorna le posizioni in coda
                    self._update_queue_positions()
                    
                    # Avvia il processing del nuovo job
                    return next_job

    def _update_queue_positions(self):
        """Aggiorna le posizioni dei job in coda"""
        queue_list = list(self.queued_jobs.queue)
        for i, job in enumerate(queue_list, 1):
            job.queue_position = i

    def _monitor_resources(self):
        """Monitora l'utilizzo delle risorse per ogni job attivo"""
        while True:
            with self.lock:
                for job in self.active_jobs.values():
                    process = psutil.Process(os.getpid())
                    job.memory_usage = process.memory_percent()
                    job.cpu_usage = process.cpu_percent()
            time.sleep(1)

class ImageProcessor:
    """Gestisce il processing delle immagini con un pool di processi"""
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=NUM_WORKERS)
        
    def process_single_image(self, image_path: str, output_path: str):
        """Processa una singola immagine"""
        try:
            if image_path.lower().endswith('.heic'):
                jpg_path = heic_to_jpg.convert_single_file(image_path, output_path)
                return depth_map.process_single_image(jpg_path, output_path)
            return depth_map.process_single_image(image_path, output_path)
        except Exception as e:
            logger.error(f"Errore nel processing dell'immagine {image_path}: {str(e)}")
            raise

    async def process_job(self, job: JobStatus, image_paths: List[str]):
        """Processa un job completo utilizzando il pool di processi"""
        try:
            total_images = len(image_paths)
            futures = []
            
            # Submitting all images for processing
            for img_path in image_paths:
                future = self.pool.submit(
                    self.process_single_image,
                    img_path,
                    job.output_folder
                )
                futures.append(future)
            
            # Monitoring progress
            completed = 0
            for future in futures:
                try:
                    result = future.result()
                    completed += 1
                    job.progress = (completed / total_images) * 100
                except Exception as e:
                    logger.error(f"Error processing image in job {job.job_id}: {str(e)}")
                    
            job.status = 'completed'
            job.completion_time = time.time()
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            logger.error(f"Job {job.job_id} failed: {str(e)}")

# Inizializzazione delle componenti globali
job_queue = JobQueue(MAX_CONCURRENT_JOBS, MAX_QUEUED_JOBS)
image_processor = ImageProcessor()

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({"error": "Nessun file trovato"}), 400

    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({"error": "Nessuna immagine fornita"}), 400

    # Crea job
    job_id = str(time.time())
    folder = os.path.join(UPLOAD_FOLDER, job_id)
    folder_output = os.path.join(OUTPUT_FOLDER, job_id, 'depth')
    
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_output, exist_ok=True)

    # Salva le immagini
    image_paths = []
    for file in files:
        if file.filename == '':
            continue
        path = os.path.join(folder, file.filename)
        file.save(path)
        image_paths.append(path)

    # Crea il job status
    job = JobStatus(
        job_id=job_id,
        status='pending',
        start_time=time.time(),
        num_images=len(image_paths),
        folder=folder,
        output_folder=folder_output
    )

    # Prova ad aggiungere il job alla coda
    if not job_queue.add_job(job):
        return jsonify({
            "error": "Sistema sovraccarico, riprova più tardi",
            "queued_jobs": job_queue.queued_jobs.qsize(),
            "active_jobs": len(job_queue.active_jobs)
        }), 503

    # Se il job è stato accettato, avvia il processing se è attivo
    if job.queue_position is None:  # Job è attivo
        job.status = 'processing'
        threading.Thread(
            target=lambda: image_processor.process_job(job, image_paths)
        ).start()

    return jsonify({
        "message": "Job accettato",
        "job_id": job_id,
        "status": job.status,
        "queue_position": job.queue_position,
        "num_images": len(image_paths)
    }), 202

@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    # Cerca il job tra quelli attivi
    job = job_queue.active_jobs.get(job_id)
    if job:
        return jsonify(asdict(job))
    
    # Cerca il job tra quelli in coda
    for queued_job in list(job_queue.queued_jobs.queue):
        if queued_job.job_id == job_id:
            return jsonify(asdict(queued_job))
    
    return jsonify({"error": "Job non trovato"}), 404

@app.route('/system-status', methods=['GET'])
def get_system_status():
    return jsonify({
        "active_jobs": len(job_queue.active_jobs),
        "queued_jobs": job_queue.queued_jobs.qsize(),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "max_queued_jobs": MAX_QUEUED_JOBS,
        "workers_per_job": NUM_WORKERS,
        "system_cpu": psutil.cpu_percent(),
        "system_memory": psutil.virtual_memory().percent,
        "system_time": datetime.now().isoformat(),
        "queue": [asdict(job) for job in list(job_queue.queued_jobs.queue)]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
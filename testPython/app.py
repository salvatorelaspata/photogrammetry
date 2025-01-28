from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

# Cartella per salvare le immagini caricate
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_images():
    # Verifica che ci siano file nella richiesta
    if 'images' not in request.files:
        return jsonify({"error": "Nessun file trovato"}), 400

    files = request.files.getlist('images')  # Ottieni tutti i file con il nome 'images'
    if len(files) == 0:
        return jsonify({"error": "Nessuna immagine fornita"}), 400

    # Salva le immagini nella cartella UPLOAD_FOLDER
    # UPLOAD_FOLDER + timestamp
    folder = UPLOAD_FOLDER + '/' + str(time.time())
    # create folder if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in files:
        if file.filename == '':
            continue
        file.save(os.path.join(folder, file.filename))

    # Risposta di conferma
    return jsonify({
        "message": "Immagini prese in carico per il processamento",
        "num_images": len(files)
    }), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Benvenuto su ab"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
#!/bin/sh

# Verifica se la directory è vuota
if [ -z "$(ls -A /source/output)" ]; then
  echo "Directory vuota, creo i file..."
else
  echo "Directory già popolata:"
  ls -l /source/output
fi

# install python dependencies
pip install -r /source/requirements.txt

python3 /source/src/heic_to_jpg.py /source/input/HEIC /source/output/HEIC_TO_JPG

python3 /source/src/depth_map.py /source/output/HEIC_TO_JPG /source/output/DEPTH_MAP

# Mantieni il container attivo
exec tail -f /dev/null

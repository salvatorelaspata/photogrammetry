#!/bin/sh

# Verifica se la directory è vuota
if [ -z "$(ls -A /app/testFolder)" ]; then
  echo "Directory vuota, creo i file..."
  echo "Questo è un file di test" > /app/testFolder/testFile.txt
  echo "Questo è un altro file di test" > /app/testFolder/asd.txt
else
  echo "Directory già popolata:"
  ls -l /app/testFolder
fi

# Mantieni il container attivo
exec tail -f /dev/null

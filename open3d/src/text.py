# write file in /source/output/text.txt

if __name__ == "__main__":
  # write file in /source/output/text.txt
  with open("./sdasdasd.txt", "w") as f:
    f.write("Hello, World!")
  print("File scritto con successo!")
else:
  print("Errore: il main non è stato eseguito")
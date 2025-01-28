FROM alpine:latest

# Create the directory
RUN mkdir -p /app/testFolder

# Copia uno script d'avvio nel container
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


# Set the working directory
WORKDIR /app/testFolder
# Comando di default
CMD ["/entrypoint.sh"]
FROM alpine:latest

# Create the directory
RUN mkdir -p /app/testFolder

# Write a simple file in the directory
RUN echo "This is a test file" > /app/testFolder/testFile.txt

# Set the working directory
WORKDIR /app/testFolder

# Default command
CMD ["tail", "-f","/dev/null"]
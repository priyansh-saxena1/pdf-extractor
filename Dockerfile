FROM python:3.11-alpine

# Install system dependencies including Tesseract OCR and language packs
RUN apk add --no-cache tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
    tesseract-ocr-jpn tesseract-ocr-ara tesseract-ocr-hin

# Create app directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models directory
COPY src/models/ ./models/

# Copy the rest of the application
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "process_pdfs.py"] 
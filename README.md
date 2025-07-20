# PDF Outline Extractor

A **vision‑first**, **multilingual** PDF outline extractor that processes PDF documents to extract structured outlines including titles and headings. It uses computer vision and NLP techniques to identify and classify text blocks.

## 🗂 Repository Structure

```
.
├── Dockerfile              # Container definition
├── README.md               # This file
├── models/                 # ONNX model files
│   ├── yolo_nano.onnx      # YOLOv5-Nano heading detector (4 MB)
│   ├── langid_quant.onnx   # FastText language identifier (1 MB)
│   └── blockclf_quant.onnx # MiniLM block classifier (12 MB)
├── process_pdfs.py         # Main pipeline script
├── requirements.txt        # Python dependencies
├── schema/                 # JSON schema definitions
│   └── output_schema.json  # Output schema for JSON files
├── src/                    # Source code modules
│   ├── pdf_processor.py    # PDF extraction and rendering
│   ├── heading_detector.py # YOLO-based heading detection
│   ├── language_identifier.py # Language identification
│   ├── ocr_processor.py    # Tesseract OCR processing
│   ├── block_classifier.py # Heading level classification
│   ├── json_exporter.py    # JSON output formatting
│   └── train/              # Model training scripts
│       ├── train_detector.py  # YOLO training
│       ├── train_langid.py    # FastText training
│       └── train_blockclf.py  # MiniLM training
├── tests/                  # Test scripts
│   └── test_pipeline.py    # Pipeline tests
├── inputs/                 # Input PDF files
└── outputs/                # Output JSON files
```

## 🔍 Architecture Overview

The PDF Outline Extractor uses a multi-stage pipeline approach:

1. **Page Ingestion & Preprocessing**
   * PyMuPDF extracts text spans and renders pages at 300 DPI
   * Images are resized to 640×640 for the detector

2. **Heading Detection (YOLOv5-Nano)**
   * Tiny object detector (4 MB ONNX) spots heading-like regions
   * Processes each page in ~1 ms

3. **Language Identification (FastText)**
   * Predicts script (Latin, CJK, Arabic, Indic) for each region
   * Selects appropriate Tesseract language pack

4. **OCR Processing**
   * Tesseract extracts text from detected regions
   * Regex pattern matching for numbered headings

5. **Block Classification (MiniLM)**
   * 4-layer transformer with bbox/meta embeddings
   * Classifies regions as Title, H1-H4, or Other

6. **Post-Processing & JSON Output**
   * Aggregates results, sorts by page and position
   * Validates against schema and exports JSON

## ⚙️ Installation & Usage

### Requirements

* Python 3.8+
* Tesseract OCR with language packs
* ONNX Runtime

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-outline-extractor.git
cd pdf-outline-extractor

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if not included)
mkdir -p models
# Download models from release page or train your own
```

### Usage

```bash
# Process all PDFs in the input directory
python process_pdfs.py --input-dir inputs --output-dir outputs

# Run in dry-run mode to compare with ground truth
python process_pdfs.py --input-dir inputs --output-dir outputs --dry-run

# Run tests
python -m unittest tests/test_pipeline.py
```

### Docker

```bash
# Build the Docker image
docker build -t pdf-outline-extractor .

# Run the container
docker run --rm \
  -v $(pwd)/inputs:/app/input:ro \
  -v $(pwd)/outputs:/app/output \
  pdf-outline-extractor
```

## 🧪 Training Models

The repository includes scripts for training each model:

### YOLOv5-Nano Heading Detector

```bash
python src/train/train_detector.py --data path/to/data.yaml --epochs 100
```

### FastText Language Identifier

```bash
python src/train/train_langid.py --train-file path/to/train.txt --epochs 25
```

### MiniLM Block Classifier

```bash
python src/train/train_blockclf.py --train-file path/to/train.json --epochs 10
```

## 📈 Performance & Footprint

| Metric             | Value           |
| ------------------ | --------------- |
| Container Size     | ~147 MB         |
| Processing Speed   | ~0.4 s/page     |
| Peak RAM Usage     | < 1 GB          |
| CPU Usage          | 8 threads       |

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

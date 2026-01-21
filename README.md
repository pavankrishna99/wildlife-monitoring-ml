# Wildlife Monitoring System

A web-based AI-powered wildlife monitoring system that detects animals in images and classifies their species using computer vision and deep learning.

## Features

- 🐾 **Animal Detection**: Uses MegaDetector (YOLOv5) for accurate animal detection
- 🧠 **Species Classification**: ResNet18-based classifier for species identification
- 🔍 **Novelty Detection**: Autoencoder-based anomaly detection for unknown species
- 🌐 **Web Interface**: Modern, responsive web application
- 📊 **Real-time Results**: Instant analysis with confidence scores and visualization

## Architecture

The system uses a 3-stage pipeline:

1. **Stage 1 - Detection**: MegaDetector identifies animals in images
2. **Stage 2 - Classification**: ResNet classifier predicts species
3. **Stage 3 - Novelty Detection**: Autoencoder detects unknown species

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**:
   ```bash
   cd wildlife-monitoring
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Download or place model files**:
   Place the following model files in the `backend/models/` directory:
   - `md_v5a.0.0.pt` - MegaDetector model
   - `stage2_species_finetuned.pth` - Species classifier
   - `stage3_autoencoder.pth` - Autoencoder for novelty detection

### Running the Application

1. **Start the backend server**:
   ```bash
   cd backend
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload an image** by dragging & dropping or clicking "Choose File"
2. **Click "Analyze Image"** to process the wildlife image
3. **View results** including:
   - Detection status (Known/Unknown species)
   - Species classification
   - Confidence score
   - Novelty score
   - Visualized bounding box and metrics

## API Endpoints

- `POST /predict` - Upload and analyze an image
- `GET /results/<filename>` - Retrieve processed result images
- `GET /` - Serve the web interface

## Model Files Required

The application requires three trained model files:

1. **MegaDetector** (`md_v5a.0.0.pt`): Pre-trained animal detection model
2. **Species Classifier** (`stage2_species_finetuned.pth`): Fine-tuned for specific species
3. **Autoencoder** (`stage3_autoencoder.pth`): Trained for novelty detection

## Configuration

Key parameters can be adjusted in `backend/app.py`:

- `NOVELTY_THRESHOLD`: Threshold for novelty detection (default: 0.007806)
- `CONF_THRESHOLD`: Minimum confidence for known species (default: 0.6)

## Technologies Used

- **Backend**: Flask, PyTorch, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: YOLOv5, ResNet, Autoencoders
- **Deployment**: Local web server

## Troubleshooting

**Common Issues:**

1. **Model files not found**: Ensure all model files are in `backend/models/`
2. **CUDA out of memory**: The app will automatically use CPU if CUDA is unavailable
3. **Port already in use**: Change the port in `app.py` if 5000 is occupied

**Performance Tips:**

- Use GPU for faster processing (if available)
- Process images one at a time for best performance
- Ensure adequate RAM (4GB+ recommended)

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source. Please check individual model licenses for usage restrictions.
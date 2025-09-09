# TreeInventorization

A comprehensive street-level tree detection and inventory system that processes Google Street View panoramas to identify, segment, and geolocate trees in urban environments.

## Overview

This project provides an end-to-end pipeline for automated tree detection from street-level imagery. It combines computer vision models for tree segmentation, depth estimation, and quality assessment to create accurate tree inventories with precise geolocation data.

## Features

- **Tree Detection**: YOLO-based tree segmentation from street view panoramas
- **Depth Estimation**: DepthAnything V2 model for accurate depth mapping
- **Quality Assessment**: Mask quality verification to filter false positives
- **Geolocation**: Precise tree positioning using depth maps and camera parameters
- **Batch Processing**: Asynchronous processing of multiple panoramas
- **Evaluation Tools**: Comprehensive evaluation against ground truth data
- **Visualization**: Interactive maps showing detected trees

## Project Structure

```
TreeInventorization/
├── main.py                 # Main pipeline entry point
├── config.py              # Configuration management
├── cli.py                 # Command-line interface
├── src/                   # Source code
│   ├── inference/         # Model inference modules
│   │   ├── depth.py       # Depth estimation
│   │   ├── mask.py        # Mask quality verification
│   │   └── segment.py     # Tree segmentation
│   ├── pipeline/          # Main processing pipeline
│   │   └── pano_async.py  # Asynchronous panorama processing
│   ├── utils/             # Utility functions
│   │   ├── depth_calibration.py
│   │   ├── geodesic.py
│   │   ├── masks.py
│   │   ├── transformation.py
│   │   └── unwrap.py
│   └── notebooks/         # Testing notebooks (not used in pipeline)
├── models/                # Model weights (gitignored - must be downloaded)
│   ├── TreeModelV3/       # Trunk segmentation model
│   ├── TreeModel/         # Tree segmentation model
│   ├── DepthAnything/     # Depth estimation model
│   ├── CalibrateDepth/    # Depth calibration model
│   └── MaskQuality/       # Mask quality assessment model
├── data/                  # Data storage
│   ├── full/              # Full panorama images
│   ├── views/             # Perspective views
│   ├── depth_maps/        # Generated depth maps
│   ├── masks/             # Generated masks
│   └── logs/              # Processing logs
├── streetviews/           # Panorama metadata
│   └── *.csv              # Panorama ID lists
├── eval/                  # Evaluation tools
│   ├── eval.py            # Model evaluation script
│   └── 28_29_groundtruth.csv  # Ground truth data
├── annotations/           # Training annotations (ignored)
└── old/                   # Legacy code (ignored)
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utk984/TreeInventorization
   cd TreeInventorization
   ```

2. **Download model weights:**
   The `models/` directory is gitignored and must be populated with pre-trained model weights:
   - TreeModelV3 weights
   - DepthAnything V2 checkpoints
   - CalibrateDepth model weights
   - MaskQuality model weights

   Place the model files in their respective directories as specified in `config.py`.

## Usage

### Basic Usage

Run the main pipeline with default settings:

```bash
python3 main.py --input_csv path/to/panorama_ids.csv
```

### Command Line Options

```bash
python3 main.py --input_csv path/to/panorama_ids.csv --output_csv path/to/output.csv --fov 90 --width 1024 --height 720
```

**Parameters:**
- `--input_csv, -i`: Path to panorama ID CSV (default: from config.py)
- `--output_csv, -o`: Output CSV path (default: from config.py)
- `--fov`: Horizontal field of view in degrees (default: 90)
- `--width`: Perspective view width in pixels (default: 1024)
- `--height`: Perspective view height in pixels (default: 720)

### Evaluation

Evaluate model predictions against ground truth:

```bash
python3 eval/eval.py path/to/predictions.csv --plot True
```

**Parameters:**
- `predictions_csv_path`: Path to predictions CSV file
- `--plot`: Generate visualization map (True/False)

## Configuration

The system is configured through `config.py`. Key settings include:

- **Model paths**: Locations of pre-trained model weights
- **Data directories**: Input/output folder paths
- **Processing parameters**: Image dimensions, batch size, FOV
- **Device settings**: CUDA/CPU selection
- **Logging**: Log file configuration

## Data Format

### Input CSV Format
Panorama ID CSV should contain:
- `pano_id`: Google Street View panorama identifier

### Output CSV Format
Generated tree data CSV contains:
- `pano_id`: Source panorama identifier
- `tree_lat`, `tree_lng`: Tree coordinates
- `conf`: Detection confidence score
- `image_path`: Path to source image
- Additional metadata fields

## Pipeline Workflow

1. **Panorama Fetching**: Download street view panoramas and depth maps
2. **Perspective Generation**: Create multiple perspective views from panoramas
3. **Tree Detection**: Apply YOLO model for tree segmentation
4. **Depth Estimation**: Generate depth maps using DepthAnything V2
5. **Quality Assessment**: Filter detections using mask quality model
6. **Geolocation**: Calculate precise tree coordinates using depth and camera parameters
7. **Output Generation**: Save results to CSV format

## Models

- **TreeModelV3**: YOLO-based trunk segmentation model
- **TreeModel**: YOLO-based tree segmentation model
- **DepthAnything V2**: Vision Transformer for depth estimation
- **CalibrateDepth**: Random Forest model for depth calibration
- **MaskQuality**: Quality assessment for segmentation masks

## Evaluation Metrics

The evaluation system provides:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Matches %**: Percentage of correct panorama matches
- **Distance thresholds**: Configurable matching distances (3m, 5m)
- **Duplicate removal**: Configurable duplicate detection (2m, 5m)
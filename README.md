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
- **Interactive Visualization**: Real-time interactive maps with tree and street view markers

## Project Structure

```
TreeInventorization/
├── run.sh                 # Unified script for pipeline, eval, and plot
├── main.py                # Main pipeline entry point
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
│   │   ├── plot.py        # Interactive plotting and visualization
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
├── outputs/               # Generated outputs
│   └── *.csv              # Tree detection results
├── eval/                  # Evaluation tools
│   └── eval.py            # Model evaluation script (plotting removed)
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

### Quick Start

The project now includes a unified script `run.sh` that provides three main operations:

```bash
# Make the script executable (if not already)
chmod +x run.sh

# Run the tree detection pipeline
./run.sh --pipeline

# Run evaluation against ground truth
./run.sh --eval

# Run interactive plotting and visualization
./run.sh --plot
```

### Detailed Usage

#### 1. Pipeline (Tree Detection)

Run the main pipeline with custom parameters:

```bash
./run.sh --pipeline --input_csv path/to/panorama_ids.csv --output_csv path/to/output.csv --fov 90 --width 1024 --height 720
```

**Pipeline Parameters:**
- `--input_csv, -i`: Path to panorama ID CSV (default: `./streetviews/chandigarh_streets.csv`)
- `--output_csv, -o`: Output CSV path (default: `./outputs/chandigarh_trees.csv`)
- `--fov`: Horizontal field of view in degrees (default: 90)
- `--width`: Perspective view width in pixels (default: 1024)
- `--height`: Perspective view height in pixels (default: 720)
- `--save_depth_maps`: Save depth maps (default: False)
- `--save_mask_json`: Save mask JSON (default: False)

#### 2. Evaluation

Evaluate model predictions against ground truth:

```bash
./run.sh --eval path/to/predictions.csv
```

**Evaluation Parameters:**
- `predictions_csv_path`: Path to predictions CSV file (default: `./outputs/chandigarh_trees.csv`)

#### 3. Interactive Plotting

Generate interactive maps with tree and street view visualization:

```bash
./run.sh --plot --tree-csv ./outputs/tree_data.csv --streetview-csv ./streetviews/chandigarh_streets.csv --port 5000
```

**Plotting Parameters:**
- `--tree-csv`: Path to tree data CSV (default: `outputs/tree_data.csv`)
- `--streetview-csv`: Path to street view CSV (default: `streetviews/chandigarh_streets.csv`)
- `--data-dir`: Data directory path (default: `data`)
- `--server-url`: Server URL for images (default: `http://localhost:8000`)
- `--distance-threshold`: Distance threshold for duplicate removal in meters (default: 3.0)
- `--port`: Port to serve the map (default: 5000)

### Help

Get detailed help for any operation:

```bash
./run.sh --help
```

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
- `lat`, `lng`: Panorama coordinates (for street view plotting)

### Output CSV Format
Generated tree data CSV contains:
- `pano_id`: Source panorama identifier
- `tree_lat`, `tree_lng`: Tree coordinates
- `conf`: Detection confidence score
- `image_path`: Path to source image
- `stview_lat`, `stview_lng`: Street view coordinates
- Additional metadata fields

### Ground Truth Format
Ground truth CSV should contain:
- `tree_lat`, `tree_lng`: Ground truth tree coordinates
- `pano_id`: Associated panorama identifier

## Pipeline Workflow

1. **Panorama Fetching**: Download street view panoramas and depth maps
2. **Perspective Generation**: Create multiple perspective views from panoramas
3. **Tree Detection**: Apply YOLO model for tree segmentation
4. **Depth Estimation**: Generate depth maps using DepthAnything V2
5. **Quality Assessment**: Filter detections using mask quality model
6. **Geolocation**: Calculate precise tree coordinates using depth and camera parameters
7. **Duplicate Removal**: Remove duplicate trees within configurable distance threshold

## Visualization Features

The interactive plotting system provides:

- **Interactive Maps**: Real-time folium-based maps with zoom and pan
- **Tree Markers**: Green circle markers showing detected tree locations
- **Street View Markers**: Blue circle markers showing panorama locations
- **Connection Lines**: Red lines connecting trees to their source panoramas
- **Image Popups**: Click markers to view associated images
- **Duplicate Removal**: Automatic removal of trees within 3m distance threshold
- **Responsive Design**: Full-screen maps with no scrollbars
- **Live Streaming**: Maps served via Flask without saving HTML files

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
- **Distance thresholds**: Configurable matching distances (3m, 5m)
- **Duplicate removal**: Configurable duplicate detection (2m, 5m)

## TODO

Future improvements and planned features:

- [x] **Flag-based saving for depth maps and segmentation masks** - Add config flags to optionally save intermediate processing results (depth maps, segmentation masks) for debugging and analysis purposes.

- [x] **Interactive visualization system** - Real-time interactive maps with tree and street view markers, connection lines, and image popups.

- [x] **Unified command-line interface** - Single script for pipeline, evaluation, and visualization operations.

- [ ] **Multiview triangulation** - Implement triangulation algorithms to improve tree localization accuracy by combining detections from multiple panorama viewpoints.

- [ ] **Improve tree detection model** - Enhance the current tree segmentation model with better training data, architecture improvements, or ensemble methods for higher accuracy.

- [ ] **Redo depth anything + regression pipeline** - Refactor the depth estimation pipeline to improve the integration between DepthAnything model and the regression-based depth calibration system.

## Acknowledgment

Developers: [Utkarsh Agarwal](https://github.com/Utk984), [Malhar Bhise](https://github.com/coolperson111)

This project was undertaken in collaboration with the [Geospatial Computer Vision Group](https://anupamsobti.github.io/geospatial-computer-vision/) led by [Dr. Anupam Sobti](https://anupamsobti.github.io/). We are grateful for the support and guidance provided throughout the development of this project.
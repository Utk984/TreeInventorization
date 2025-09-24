#!/bin/bash

# Tree Inventorization Pipeline Runner Script
# This script provides two main options: pipeline and plot

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION] [ADDITIONAL_ARGS...]"
    echo ""
    echo "Options:"
    echo "  --pipeline    Run the tree detection pipeline (main.py)"
    echo "  --plot        Run interactive plotting (src/pipeline/plot.py)"
    echo "  --view3d      Run 3D panorama viewer with tree detection visualization"
    echo ""
    echo "Additional arguments are passed through to the respective Python scripts."
    echo ""
    echo "Examples:"
    echo "  $0 --pipeline"
    echo "  $0 --plot"
    echo "  $0 --plot --distance-threshold 5.0 --no-save"
    echo "  $0 --view3d --tree-csv outputs/chandigarh_trees.csv"
    echo ""
    echo "Pipeline (main.py):"
    echo "  The pipeline uses configuration from config.py. No command line arguments are supported."
    echo "  To modify settings, edit the Config class in config.py"
    echo ""
    echo "Plot arguments (src/pipeline/plot.py):"
    echo "  --distance-threshold    Distance threshold for duplicate removal in meters (default: 3.0)"
    echo "  --no-save               Do not save map to HTML file"
    echo "  --no-connections        Do not show connection lines between trees and street views"
    echo ""
    echo "3D Viewer arguments (run_3d_viewer_fixed.py):"
    echo "  --tree-csv              Path to tree detection CSV file (default: outputs/chandigarh_trees.csv)"
    echo "  --streetview-csv        Path to street view CSV file (default: streetviews/chandigarh_streets.csv)"
    echo "  --data-dir              Data directory path (default: data)"
    echo "  --port                  Port to serve the viewer (default: 5002)"
    echo "  --host                  Host to bind to (default: 0.0.0.0)"
}

# Function to check Python dependencies
check_dependencies() {
    print_status "Checking Python dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check basic dependencies
    python3 -c "import pandas, numpy, geopy" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some basic Python packages might be missing."
        print_status "Installing basic dependencies..."
        pip3 install pandas numpy geopy
        if [ $? -ne 0 ]; then
            print_error "Failed to install basic packages"
            exit 1
        fi
    fi
    
    print_success "Basic dependencies are available"
}

# Function to run pipeline
run_pipeline() {
    print_status "Running tree detection pipeline..."
    print_status "Arguments: $@"
    
    # Check for additional dependencies for pipeline
    python3 -c "import torch, transformers, PIL, cv2" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Pipeline dependencies might be missing. Please install them if needed."
    fi
    
    python3 main.py "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Pipeline completed successfully!"
    else
        print_error "Pipeline failed with exit code: $?"
        exit 1
    fi
}


# Function to run plotting
run_plot() {
    print_status "Running interactive plotting..."
    print_status "Arguments: $@"
    
    # Check for additional dependencies for plotting
    python3 -c "import pydeck, sklearn" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_status "Installing plotting dependencies..."
        pip3 install pydeck scikit-learn
        if [ $? -ne 0 ]; then
            print_error "Failed to install plotting dependencies"
            exit 1
        fi
    fi
    
    # Run the plotting script
    python3 src/pipeline/plot.py "$@"
    PLOT_EXIT_CODE=$?
    
    if [ $PLOT_EXIT_CODE -eq 0 ]; then
        print_success "Plotting completed successfully!"
    else
        print_error "Plotting failed with exit code: $PLOT_EXIT_CODE"
        exit 1
    fi
}


# Main script logic
if [ $# -eq 0 ]; then
    print_error "No option provided"
    show_usage
    exit 1
fi

# Parse the first argument
case "$1" in
    --pipeline)
        shift  # Remove --pipeline from arguments
        check_dependencies
        run_pipeline "$@"
        ;;
    --plot)
        shift  # Remove --plot from arguments
        check_dependencies
        run_plot "$@"
        ;;
    --view3d)
        shift  # Remove --view3d from arguments
        print_status "Starting 3D panorama viewer..."
        python run_3d_viewer_fixed.py "$@"
        ;;
    --help|-h)
        show_usage
        exit 0
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

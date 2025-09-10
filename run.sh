#!/bin/bash

# Tree Inventorization Pipeline Runner Script
# This script provides three main options: pipeline, eval, and plot

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
    echo "  --eval        Run evaluation against ground truth (eval/eval.py)"
    echo "  --plot        Run interactive plotting (src/utils/plot.py)"
    echo ""
    echo "Additional arguments are passed through to the respective Python scripts."
    echo ""
    echo "Examples:"
    echo "  $0 --pipeline --input_csv ./streetviews/test.csv --output_csv ./outputs/test_trees.csv"
    echo "  $0 --eval ./outputs/my_predictions.csv"
    echo "  $0 --plot"
    echo ""
    echo "Pipeline arguments (main.py):"
    echo "  --input_csv, -i     Path to panorama ID CSV (default: ./streetviews/chandigarh_streets.csv)"
    echo "  --output_csv, -o    Where to save tree data CSV (default: ./outputs/chandigarh_trees.csv)"
    echo "  --fov               Horizontal field of view in degrees (default: 90)"
    echo "  --width             Perspective view width in pixels (default: 1024)"
    echo "  --height            Perspective view height in pixels (default: 720)"
    echo "  --save_depth_maps   Save depth maps (default: False)"
    echo "  --save_mask_json    Save mask JSON (default: False)"
    echo ""
    echo "Eval arguments (eval/eval.py):"
    echo "  predictions_csv_path    Path to predictions CSV (default: ./outputs/chandigarh_trees.csv)"
    echo ""
    echo "Plot arguments (src/utils/plot.py):"
    echo "  --tree-csv              Path to tree data CSV (default: outputs/tree_data.csv)"
    echo "  --streetview-csv        Path to street view CSV (default: streetviews/chandigarh_streets.csv)"
    echo "  --data-dir              Data directory path (default: data)"
    echo "  --server-url            Server URL for images (default: http://localhost:8000)"
    echo "  --distance-threshold    Distance threshold for duplicate removal in meters (default: 3.0)"
    echo "  --port                  Port to serve the map (default: 5000)"
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

# Function to run evaluation
run_eval() {
    print_status "Running evaluation..."
    print_status "Arguments: $@"
    
    # Check for additional dependencies for eval
    python3 -c "import prettytable" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_status "Installing prettytable for evaluation..."
        pip3 install prettytable
    fi
    
    python3 eval/eval.py "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed successfully!"
    else
        print_error "Evaluation failed with exit code: $?"
        exit 1
    fi
}

# Function to run plotting
run_plot() {
    print_status "Running interactive plotting..."
    print_status "Arguments: $@"
    
    # Check for additional dependencies for plotting
    python3 -c "import folium, flask" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_status "Installing plotting dependencies..."
        pip3 install folium flask
        if [ $? -ne 0 ]; then
            print_error "Failed to install plotting dependencies"
            exit 1
        fi
    fi
    
    # Start data server in background if plotting
    print_status "Starting data server for image serving..."
    cd data
    python3 -m http.server 8000 &
    SERVER_PID=$!
    cd - > /dev/null
    
    # Wait a moment for server to start
    sleep 2
    
    # Check if server is running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        print_error "Failed to start data server"
        exit 1
    fi
    
    print_success "Data server started (PID: $SERVER_PID)"
    print_status "Server URL: http://localhost:8000"
    
    # Run the plotting script
    python3 src/utils/plot.py "$@"
    PLOT_EXIT_CODE=$?
    
    # Clean up server
    kill $SERVER_PID 2>/dev/null
    print_status "Stopped data server (PID: $SERVER_PID)"
    
    if [ $PLOT_EXIT_CODE -eq 0 ]; then
        print_success "Plotting completed successfully!"
        print_status "Interactive map should be available at http://localhost:5000"
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
    --eval)
        shift  # Remove --eval from arguments
        check_dependencies
        run_eval "$@"
        ;;
    --plot)
        shift  # Remove --plot from arguments
        check_dependencies
        run_plot "$@"
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

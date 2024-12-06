# TreeInventorization


## Run 

1. Add the following env variables to a .env
```env
# Postgres DB
DB_URL="**********"

# Google Gemini
GEMINI_API_KEY="**********"

# AWS S3 Bucket
CLOUD_STORAGE_BUCKET="**********"
CLOUD_URL="**********"
```

2. Add the chandigarh_panoramas.csv file to ./data/input/chandigarh_panoramas.csv

3. Run the following commands
```bash
# After placing ./data/input/chandigarh_panoramas.csv

# activate virtual enviromnent, then
pip install -r requirements.txt

# run
python3 src/main.py
```

## Pipeline

1. extract all panoramas (street view images) within a given city's limits.
2. divide the panoramas (unwrap them) into 90deg views.
3. run a instance segmentation model to detect trees
4. save the images into a image file dump on the cloud.
5. extract a point on the image around the bottom of each tree
6. project this image back to the original image and then using the depth map, estimate the latlong of the tree
7. send the tree image to gemini api, and get the estimate of tree species, common_name and short description.
8. store all the below data into the database (postgres)


## Malhar's NOTES

1. follow chatgpt folder structure.
2. start with a connection to db, load_dotenv(), etc... dont repeat this too many times.
3. Do it batchwise, follow gpt code's guide (as in pipeline.py) (tqdm)
    - get 32 pano images, store in a list of pano objects.
    - for each pano - get 3 fovs
    - for each fov, get all trees
    - for each tree
        1. (??) check if tree already exists at latlong +- radius=5m?
        2. get its image, process it, send to s3 bucket
        3. send its image to gemini, get its prediction (figure out prompt caching)
        4. save all its info to DB (sqlalchemy + postgis hopefully)
            - common name
            - species name
            - short desc
            - link path
            - latlong
            - stview image id
            - orientation of capture (theta)
            - height(?)
            - diameter (?)
            - lat_offset=0,
            - lng_offset=0,
            - image_x=image_x,
            - image_y=image_y,


## file structure

urban-tree-inventory/
├── data/                                # Folder for all raw and processed data
│   ├── input/                           # Input files like panoramas, CSVs
│   │   └── chandigarh_panoramas.csv     # All panoramas in chandigarh preprocessed
│   ├── images/                          # images stored while processing or testing
│   │   └── panoramas/                   # whole panorama images
│   │   └── perspectives/                # FOV (-90/0/90) degree images
│   │   └── predict/                     # Tree predictions (cropped or with bounding box)
│   ├── logs/                            # Log files
│   └── temp/                            # Temporary files during processing
├── src/                                 # Source code
│   ├── __init__.py                      # Makes src a Python package
│   ├── main.py                          # Main entry point for running the pipeline
│   ├── config.py                        # Configuration and constants
│   ├── data_pipeline/
│   │   ├── __init__.py                  # Makes data_pipeline a package
│   │   ├── unwrapping.py                # Functions to unwrap panoramas
│   │   ├── segmentation.py              # Instance segmentation model prediction logic
│   │   ├── cloud_storage.py             # Logic for saving images to the cloud
│   │   ├── tree_extraction.py           # Extract tree latlon from images
│   │   ├── species_detection.py         # Interaction with Gemini API
│   │   └── database.py                  # Database insertion logic
│   ├── utils/                           # Utility functions
│   │   ├── __init__.py                  # Makes utils a package
│   │   ├── image_utils.py               # Image utilities
│   │   └── batch_processing.py          # Functions to handle batch processing
│   └── visualization/
│       ├── __init__.py                  # Makes visualization a package
│       ├── map_viewer.py                # Streamlit or Folium map viewer (not implemented)
│       └── dashboard.py                 # Streamlit dashboard code (not implemented)
├── tests/                               # various random unit tests, to check functions
├── requirements.txt                     # List of Python dependencies
├── README.md                            # Documentation for the project
└── .gitignore                           # Files and directories to ignore in Git


## Todo

1. Type casting for all functions
2. All functions need to have proper documentation strings
3. Add logging instead of print statements
4. Add visualisation into this repo itself
5. Better Database connection - Maybe use ORM such as sqlalchemy?
6. Long term - Improve efficiency by taking advantages of batch processing, parallel computing, etc.
7. Integrate Panorama extraction into this so that we dont have to extract panoramas before.

import os

from treemodel.detect import detect_trees
from utils.database import save_annotations
from utils.utils import (
    get_panorama_id,
    handle_model_output,
    move_in_heading,
    process_location,
)


def collect_panoramic_data_within_square(lat, lon, radius=0.05, step_distance=0.01):
    """
    Collect panoramic data within a square box around the specified latitude and longitude,
    process panoramas, and detect trees.
    
    Args:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        radius (float): Half the side length of the square box in degrees. Default is 0.05.
        step_distance (float): Step distance in degrees for the grid traversal. Default is 0.01.
    """
    # Define the bounds of the square box
    min_lat = lat - radius
    max_lat = lat + radius
    min_lon = lon - radius
    max_lon = lon + radius
    
    # Iterate over the grid within the square box
    current_lat = min_lat
    while current_lat <= max_lat:
        current_lon = min_lon
        while current_lon <= max_lon:
            # Process each grid point
            pano_id = get_panorama_id(current_lat, current_lon)
            if pano_id is not None and not os.path.exists(f"./panoramas/{pano_id}.jpg"):
                # Process the location and get panorama data
                panorama, depth_map, heading_degrees, pano_lat, pano_lon, pano_id = (
                    process_location(current_lat, current_lon)
                )

                print("1. Processing location: ", current_lat, current_lon)

                # Detect trees in the panorama
                box_coords, width, height = detect_trees(f"./panoramas/{pano_id}.jpg")

                print("2. Detected trees: ", len(box_coords))

                # Handle model output for detected trees
                for x, y, theta in box_coords:
                    lat, lng = handle_model_output(
                        x,
                        y,
                        theta,
                        depth_map,
                        heading_degrees,
                        width,
                        height,
                        pano_lat,
                        pano_lon,
                    )
                    # Optional: Save annotations
                    # if lat is not None and lng is not None:
                    #     save_annotations(
                    #         f"./panoramas/{pano_id}.jpg",
                    #         pano_id,
                    #         pano_lat,
                    #         pano_lon,
                    #         lat,
                    #         lng,
                    #         x,
                    #         y,
                    #     )
                # print("3. Saved annotations")
            
            # Move to the next longitude in the grid
            current_lon += step_distance
        
        # Move to the next latitude in the grid
        current_lat += step_distance


def main():
    print("hello")

    lat, lon = 43.644157832726854, -79.39373552799226
    collect_panoramic_data_within_square(lat, lon, radius=0.05)


if __name__ == "__main__":
    main()

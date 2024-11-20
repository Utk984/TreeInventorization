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

    num_images = 0

    # Iterate over the grid within the square box
    current_lat = min_lat
    while current_lat <= max_lat:
        current_lon = min_lon
        while current_lon <= max_lon:
            # if num_images > 0:
            #     return

            # Process each grid point
            pano_id = get_panorama_id(current_lat, current_lon)
            if (
                pano_id is not None
            ):  # and not os.path.exists(f"./images/panoramas/{pano_id}.jpg"):
                num_images += 1
                # Process the location and get panorama data
                panorama, depth_map, heading_degrees, pano_lat, pano_lon, pano_id = (
                    process_location(current_lat, current_lon)
                )

                print("1. Processing location: ", current_lat, current_lon)

                # Detect trees in the panorama
                box_coords, species, common_names, descriptions, width, height = (
                    detect_trees(f"./images/panoramas/{pano_id}.jpg", lat, lon)
                )

                print("len of species", len(species))
                print("len of common_names", len(common_names))
                print("len of descriptions", len(descriptions))
                print("2. Detected trees: ", len(box_coords))

                # Handle model output for detected trees
                for idx, (x, y, theta) in enumerate(box_coords):
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
                    # Optional: Save annotations with new data (species, common names, descriptions)
                    if lat is not None and lng is not None:
                        if idx < len(species):
                            save_annotations(
                                path="./panoramas/pano_id.jpg",
                                panorama_id=pano_id,
                                lat=pano_lat,
                                lng=pano_lon,
                                tree_lat=float(lat),
                                tree_lng=float(lng),
                                image_x=float(x),
                                image_y=float(y),
                                species=species[idx],
                                common_name=common_names[idx],
                                description=descriptions[idx],
                                theta=float(theta),  # Pass theta value
                            )
                        else:
                            save_annotations(
                                path="./panoramas/pano_id.jpg",
                                panorama_id=pano_id,
                                lat=pano_lat,
                                lng=pano_lon,
                                tree_lat=float(lat),
                                tree_lng=float(lng),
                                image_x=float(x),
                                image_y=float(y),
                                species="",
                                common_name="",
                                description="",
                                theta=float(theta),  # Pass theta value
                            )
                        print("3. Saved annotations")

            # Move to the next longitude in the grid
            current_lon += step_distance

        # Move to the next latitude in the grid
        current_lat += step_distance

    print("\n\n\nFinished processing!!")
    print(f"Processed {num_images} images")


def main():
    print("hello")

    lat, lon = 41.89116059582178, 12.491930488965147
    collect_panoramic_data_within_square(lat, lon, radius=0.05)


if __name__ == "__main__":
    main()

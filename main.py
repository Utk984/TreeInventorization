from treemodel.detect import detect_trees
from utils.database import Database
from utils.utils import (calculate, get_panorama_id, move_in_heading,
                         process_location)


def save_annotations(path, panorama_id, lat, lng, tree_lat, tree_lng, image_x, image_y):
    db = Database()
    db.insert_annotation(
        image_path=path,
        pano_id=panorama_id,
        stview_lat=lat,
        stview_lng=lng,
        tree_lat=tree_lat,
        tree_lng=tree_lng,
        lat_offset=0,
        lng_offset=0,
        image_x=image_x,
        image_y=image_y,
        height=0,
        diameter=0,
    )
    db.close()


def handle_model_output(
    image_x,
    image_y,
    theta,
    depth,
    heading,
    image_width,
    image_height,
    image_lat,
    image_lng,
):
    distance, direction, depth = calculate(
        image_x,
        image_y,
        abs(theta - 90),
        0,
        depth,
        image_width,
        image_height,
        heading,
    )

    if depth > 0 and distance > 0:
        lat, lng = move_in_heading(
            image_lat, image_lng, int(direction), distance / 1000
        )
        return lat, lng
    return None, None


def collect_panoramic_data_within_radius(lat, lon, radius=0.05, step_distance=0.01):
    panoramic_data = {}
    headings = range(0, 360, 30)
    box_coords = []

    for heading in headings:
        current_lat, current_lon = lat, lon
        distance_moved = 0.0

        while distance_moved <= radius:
            pano_id = get_panorama_id(current_lat, current_lon)
            if pano_id is not None:
                panorma, depth_map, heading_degrees, pano_lat, pano_lon, pano_id = (
                    process_location(current_lat, current_lon)
                )

                print("1. Processing location: ", current_lat, current_lon)

                box_coords, width, height = detect_trees(f"./panoramas/{pano_id}.jpg")

                print("2. Detected trees: ", len(box_coords))

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
                    if lat is not None and lng is not None:
                        save_annotations(
                            f"./panoramas/{pano_id}.jpg",
                            pano_id,
                            pano_lat,
                            pano_lon,
                            lat,
                            lng,
                            x,
                            y,
                        )

                print("3. Saved annotations")

                current_lat, current_lon = move_in_heading(
                    current_lat, current_lon, heading, step_distance
                )
                distance_moved += step_distance


lat, lon = 30.71979998667062, 76.72142742674824
collect_panoramic_data_within_radius(lat, lon, radius=0.05)

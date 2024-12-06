import csv
import os

from OSMPythonTools.overpass import Overpass
from streetlevel import streetview

# Configuration
SEARCH_RADIUS = 50  # Search radius for panoramas in meters
CSV_FILE = "chandigarh_panoramas.csv"

overpass = Overpass()


def get_osm_data(location):
    """Retrieve boundary coordinates of a city using OpenStreetMap."""
    query = f"""
    area[name="{location}"]->.searchArea;
    (
      relation["type"="boundary"]["name"="{location}"];
    );
    (._;>;);
    out body;
    """
    try:
        result = overpass.query(query)
        result = [i for i in result.toJSON()["elements"] if "tags" not in i]
        boundary_coords = [
            (element["lat"], element["lon"])
            for element in result
            if "lat" in element and "lon" in element
        ]
        if len(boundary_coords) < 3:
            return None
        return boundary_coords
    except Exception as e:
        print(f"Error fetching OSM data for {location}: {e}")
        return None


def is_point_inside_polygon(lat, lon, polygon):
    """Check if a point is inside a polygon using ray-casting."""
    num = len(polygon)
    j = num - 1
    odd_nodes = False
    for i in range(num):
        if polygon[i][1] < lon <= polygon[j][1] or polygon[j][1] < lon <= polygon[i][1]:
            if (
                polygon[i][0]
                + (lon - polygon[i][1])
                / (polygon[j][1] - polygon[i][1])
                * (polygon[j][0] - polygon[i][0])
                < lat
            ):
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes


def load_existing_panoramas(csv_file):
    """Load existing panorama IDs from a CSV file."""
    if not os.path.exists(csv_file):
        return set()
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        return {row["pano_id"] for row in reader}


def save_panoramas_to_csv(panoramas, csv_file):
    """Append new panoramas to the CSV file."""
    file_exists = os.path.exists(csv_file)
    fieldnames = [
        "pano_id",
        "lat",
        "lon",
        "date",
        "address",
        "elevation",
        "link",
        "heading",
        "country_code",
        "source",
        "copyright_message",
        "street_names",
    ]
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(panoramas)


def find_panoramas_within_boundary(boundary, search_radius, existing_panos):
    """Find panoramas within the city's boundary."""
    latitudes = [coord[0] for coord in boundary]
    longitudes = [coord[1] for coord in boundary]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    panoramas = []
    stride = 0.00025  # 0.001 degrees ~ 111 meters
    count = 0
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            if is_point_inside_polygon(lat, lon, boundary):
                try:
                    pano = streetview.find_panorama(lat, lon, radius=search_radius)
                    if pano and pano.id not in existing_panos:
                        panoramas.append(
                            {
                                "pano_id": pano.id,
                                "lat": pano.lat,
                                "lon": pano.lon,
                                "date": pano.date,
                                "address": (
                                    ", ".join([addr.value for addr in pano.address])
                                    if pano.address
                                    else None
                                ),
                                "elevation": pano.elevation,
                                "link": pano.permalink,
                                "heading": pano.heading,
                                "country_code": pano.country_code,
                                "source": pano.source,
                                "copyright_message": pano.copyright_message,
                                "street_names": (
                                    ", ".join(
                                        [
                                            street.name.value
                                            for street in pano.street_names
                                        ]
                                    )
                                    if pano.street_names
                                    else None
                                ),
                            }
                        )
                        existing_panos.add(pano.id)
                        count += 1
                        print(
                            f"{count}. Panorama found: {pano.id} at ({pano.lat}, {pano.lon}). Date {pano.date}"
                        )
                except Exception as e:
                    print(f"Error finding panorama at ({lat}, {lon}): {e}")
            lon += stride
        lat += stride
    return panoramas


def main():
    city_name = "Chandigarh"
    print(f"Fetching boundary data for {city_name}...")
    boundary = get_osm_data(city_name)
    if not boundary:
        print(f"Unable to fetch boundary data for {city_name}. Exiting.")
        return

    print(f"Loading existing panoramas from {CSV_FILE}...")
    existing_panoramas = load_existing_panoramas(CSV_FILE)
    print(f"Found {len(existing_panoramas)} existing panoramas.")

    print(f"Searching for panoramas within {city_name}'s boundary...")
    new_panoramas = find_panoramas_within_boundary(
        boundary, SEARCH_RADIUS, existing_panoramas
    )

    print(f"Saving {len(new_panoramas)} new panoramas to {CSV_FILE}...")
    save_panoramas_to_csv(new_panoramas, CSV_FILE)

    print("Panorama collection completed.")


if __name__ == "__main__":
    main()

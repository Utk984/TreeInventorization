import csv
import math
import os

from streetlevel import streetview

SEARCH_RADIUS = 50  # Search radius for panoramas in meters
CSV_FILE = "./data/input/cdg_st_v3_28_29_panoramas.csv"


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


def load_existing_panoramas(csv_file):
    """Load existing panorama IDs from a CSV file."""
    if not os.path.exists(csv_file):
        return set()
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        return {row["pano_id"] for row in reader}


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two geographic points in meters."""
    R = 6371000  # Earth's radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def interpolate_points(lat1, lon1, lat2, lon2, stride):
    """Generate points along the line between two geographic coordinates."""
    lat, lon = lat1, lon1
    points = []

    while True:
        points.append((lat, lon))
        delta_lat = lat2 - lat
        delta_lon = lon2 - lon

        if abs(delta_lat) < stride and abs(delta_lon) < stride:
            break

        angle = math.atan2(delta_lat, delta_lon)
        lat += stride * math.sin(angle)
        lon += stride * math.cos(angle)

    points.append((lat2, lon2))  # Ensure the endpoint is included
    return points


def find_panoramas_on_street(start_point, end_point, search_radius, existing_panos):
    """Find panoramas along a straight street given its start and end points."""
    stride = 0.00025  # Approx. 25 meters, adjust based on precision needed
    # stride = 0.1
    panoramas = []

    # Interpolate points along the street
    street_points = interpolate_points(
        start_point[0], start_point[1], end_point[0], end_point[1], stride
    )

    print(f"Interpolated {len(street_points)} points along the street.")
    # print(street_points)

    # return panoramas

    count = 0
    for lat, lon in street_points:
        try:
            pano = streetview.find_panorama(lat, lon, radius=search_radius)
            if pano and pano.id not in existing_panos:
                print(pano.id, pano.lat, pano.lon, pano.date)
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
                                [street.name.value for street in pano.street_names]
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
            else:
                print(f"No panorama found at ({lat}, {lon}).")
        except Exception as e:
            print(f"Error finding panorama at ({lat}, {lon}): {e}")

    return panoramas


def main():
    # Example usage
    # Replace with actual coordinates and parameters
    start_point = (30.710973905954493, 76.80092800043607)
    end_point = (30.716201113211884, 76.79570430226612)
    search_radius = 50  # Search radius in meters

    print(f"Loading existing panoramas from {CSV_FILE}...")
    existing_panos = load_existing_panoramas(CSV_FILE)
    print(f"Found {len(existing_panos)} existing panoramas.")

    print(f"Searching for panoramas on v3 between sector 28 and 29 street...")
    panoramas = find_panoramas_on_street(
        start_point, end_point, search_radius, existing_panos
    )
    print(f"Found {len(panoramas)} panoramas on the street.")

    print(f"Saving {len(panoramas)} new panoramas to {CSV_FILE}...")
    save_panoramas_to_csv(panoramas, CSV_FILE)


if __name__ == "__main__":
    main()

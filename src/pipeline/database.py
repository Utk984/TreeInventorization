import csv
import os

import psycopg2
from psycopg2 import sql


class Database:
    def __init__(self, db_url):
        # Establish connection to the database
        try:
            self.conn = psycopg2.connect(db_url)
            self.cur = self.conn.cursor()
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def get_or_create_streetview_image(
        self, pano_id, stview_lat, stview_lng, address, elevation, heading
    ):
        # Check if the pano_id already exists in streetview_images
        check_query = sql.SQL(
            "SELECT image_id FROM streetview_images WHERE pano_id = %s"
        )
        insert_query = sql.SQL(
            "INSERT INTO streetview_images (pano_id, lat, lng, address, elevation, heading) VALUES (%s, %s, %s, %s, %s, %s) RETURNING image_id"
        )

        try:
            self.cur.execute(check_query, (pano_id,))
            result = self.cur.fetchone()

            if result:
                image_id = result[0]
            else:
                self.cur.execute(
                    insert_query,
                    (pano_id, stview_lat, stview_lng, address, elevation, heading),
                )
                image_id = self.cur.fetchone()[0]
                self.conn.commit()

            return image_id

        except Exception as e:
            print(f"Error in get_or_create_streetview_image: {e}")
            self.conn.rollback()
            return None

    def insert_annotation(
        self,
        image_path,
        pano_id,
        stview_lat,
        stview_lng,
        tree_lat,
        tree_lng,
        lat_offset=None,
        lng_offset=None,
        image_x=None,
        image_y=None,
        height=None,
        diameter=None,
        species=None,
        common_name=None,
        description=None,
        theta=None,
        address=None,
        elevation=None,
        heading=None,
    ):
        image_id = self.get_or_create_streetview_image(
            pano_id, stview_lat, stview_lng, address, elevation, heading
        )

        if image_id is None:
            print("Error: Could not obtain image_id for the annotation.")
            return

        insert_query = sql.SQL(
            """
                INSERT INTO tree_details (
                    image_id, lat, lng, lat_offset, lng_offset, image_x, image_y, annotator_name, height, diameter, img_path, species, common_name, description, theta
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
        )
        try:
            self.cur.execute(
                insert_query,
                (
                    image_id,
                    tree_lat,
                    tree_lng,
                    lat_offset,
                    lng_offset,
                    image_x,
                    image_y,
                    "Model",
                    height,
                    diameter,
                    image_path,
                    species,
                    common_name,
                    description,
                    theta,
                ),
            )
            self.conn.commit()
        except Exception as e:
            print("error was heere")
            print(e)
            self.conn.rollback()

    def load_saved(self):
        query = sql.SQL("SELECT lat, lng, lat_offset, lng_offset FROM tree_details;")

        try:
            self.cur.execute(query)
            return self.cur.fetchall()
        except Exception as e:
            print(f"Error in load_saved: {e}")
            return []

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()


def save_to_csv(trees, csv_file):
    """
    Append new panoramas to a CSV file.
    For testing purposes, when we dont really want to store in the db
    """

    file_exists = os.path.exists(csv_file)
    fieldnames = [
        "image_path",
        "pano_id",
        "stview_lat",
        "stview_lng",
        "tree_lat",
        "tree_lng",
        "lat_offset",
        "lng_offset",
        "image_x",
        "image_y",
        "height",
        "diameter",
        "gem_species",
        "gem_common_name",
        "gem_description",
        "gpt_family",
        "gpt_genus",
        "gpt_species",
        "gpt_common_name",
        "gpt_description",
        "gpt_usage",
        "theta",
        "address",
        "elevation",
        "heading",
        "conf",
    ]
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(trees)

import io

import boto3
import cv2
from PIL import Image  # For converting OpenCV image to a file-like object

# AWS S3 Configuration
BUCKET_NAME = "treeinventory-images"
s3 = boto3.client("s3")


# OpenCV Image Processing
def process_and_upload_image(image_path, s3_key):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    print(type(image))

    # Convert OpenCV image to a file-like object (BytesIO) for S3
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buffer = io.BytesIO(buffer)

    # Upload to S3
    s3.upload_fileobj(
        io_buffer,
        BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "image/jpeg"},  # Set content type
    )
    print(f"Image uploaded to S3 at: s3://{BUCKET_NAME}/{s3_key}")


# Example usage
process_and_upload_image(
    "YHuImFJqNsGqB6W2tKxB8w_view0_tree0_box0.jpg",
    "YHuImFJqNsGqB6W2tKxB8w_view0_tree0_box0.jpg",
)

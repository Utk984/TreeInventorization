import io

import cv2


def cloud_save_image(im, image_path, s3, bucket_name):
    """
    Save image to cloud storage bucket
    """

    # Convert OpenCV image to a file-like object (BytesIO) for S3
    is_success, buffer = cv2.imencode(".jpg", im)
    io_buffer = io.BytesIO(buffer)

    # Upload to S3
    s3.upload_fileobj(
        io_buffer,
        bucket_name,
        image_path,
        ExtraArgs={
            "ContentType": "image/jpeg",
            # "ACL": "bucket-owner-full-control",
        },  # Set content type
    )


def local_save_image(im, image_dir, image_path):
    """
    Save image to local storage
    """

    cv2.imwrite(f"{image_dir}/{image_path}", im)
    # print(f"Saved image to {image_path}")

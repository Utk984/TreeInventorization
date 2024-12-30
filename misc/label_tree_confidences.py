import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def process_images(csv_file, output_folder):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    path = "./data/images/tree/street"

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        image_path = row["image_path"].split("/")[-1]
        confidence_tensor = row["conf"]
        species_name = row['gpt_common_name']

        # print(path + "/" + image_path.split("/")[-1])
        # return

        try:

            # Extract the float value from the tensor string
            match = re.search(r"tensor\(\[([\d.]+)\]", confidence_tensor)
            if match:
                confidence = float(match.group(1))
            else:
                raise ValueError(f"Invalid tensor format: {confidence_tensor}")

            # Open the image
            image = Image.open(path + "/" + image_path)

            # Get image dimensions
            width, height = image.size

            # Initialize a drawing context
            draw = ImageDraw.Draw(image)

            # Set font size and load a font
            font_size = int(height * 0.05)  # Font size is 5% of image height
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                # Default to a basic font if arial.ttf is not found
                font = ImageFont.load_default()

            text = f"Conf: {confidence:.4f}\nSpecies: {species_name}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )

            # Define the text position
            text_position = (width - text_width - 10, height - text_height - 10)

            # Add text to the image
            # draw.text(text_position, text, fill="white", font=font)
            draw.multiline_text(text_position, text, fill="white", font=font, align="right")

            # Save the modified image
            output_path = f"{output_folder}/{image_path}"
            image.save(output_path)
            print(f"Processed image saved at: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        # break


def main():
    csv_file = "./data/output/street_panoramas.csv"  # Replace with your CSV file path
    output_folder = (
        "./data/images/tree/street_w_conf"  # Replace with your desired output folder
    )
    process_images(csv_file, output_folder)


if __name__ == "__main__":
    main()

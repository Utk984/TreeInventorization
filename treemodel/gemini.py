import json
import os

import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()


def get_species(image_path, lat, lon):
    # Call external API to get species information
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
    )

    species_list = []
    common_names_list = []
    descriptions_list = []

    # for all images in path/tree
    for path in os.listdir(f"{image_path}/tree"):
        image = PIL.Image.open(f"{image_path}/tree/{path}")

        # prompt to get species information, along with a species description
        prompt = f"Identify the tree species in this image of a tree, given that it is in lat long coordinates {lat}, {lon}. Respond with a JSON containing name, common_name, and short_description."

        response = model.generate_content([prompt, "\n", image])

        response_dict = response.to_dict()
        print("\n\nRESPONSEDICT", response_dict, "\n\n")
        print("\n\nRESPONSEDICT_type", type(response_dict), "\n\n")


        # Parse the JSON response
        try:
            # Extract JSON text
            json_text = response_dict["candidates"][0]["content"]["parts"][0]["text"]
            print("json text extracted")
            cleaned_json_text = json_text.strip("```json").strip("```").strip()
            print("cleaned json text extracted")

            # Parse the cleaned JSON string
            tree_data = json.loads(cleaned_json_text)
            print("tree data extracted")

            species_list.append(tree_data.get("name"))
            common_names_list.append(tree_data.get("common_name"))
            descriptions_list.append(tree_data.get("short_description"))
        except (KeyError, IndexError, json.JSONDecodeError):
            print(f"Error processing response for image: {path}")

    return species_list, common_names_list, descriptions_list

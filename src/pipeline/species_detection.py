import json

import PIL.Image


def get_species(image, pano_address, model):
    """
    Identify the species of a tree from an image, given its latitude and longitude.

    Args:
        image (ndarray): Image of the tree in ndarray format.
        address (pano.address): address of image
        model (gemini model): gemini model (1.5 flash 8b

    Returns:
        tuple: A tuple containing the species name, common name, and a short description.
    """

    # Convert ndarray to PIL Image
    pil_image = PIL.Image.fromarray(image)

    address = ""
    if pano_address:
        address = ", ".join([addr.value for addr in pano_address])

    prompt = f"Identify the tree species in the bounding box in this image, given that it's located in {address}. Give your best guess. Respond with a JSON containing name, common_name, and short_description."

    try:
        # Generate content using the model
        response = model.generate_content([prompt, "\n", pil_image])
        response_dict = response.to_dict()

        # Extract and clean JSON response
        json_text = response_dict["candidates"][0]["content"]["parts"][0]["text"]
        cleaned_json_text = json_text.strip("```json").strip("```").strip()

        # Parse the JSON response
        tree_data = json.loads(cleaned_json_text)

        species_name = tree_data.get("name")
        common_name = tree_data.get("common_name")
        short_description = tree_data.get("short_description")

        return species_name, common_name, short_description
    except (KeyError, IndexError, json.JSONDecodeError):
        print("Error processing response")
        return None, None, None

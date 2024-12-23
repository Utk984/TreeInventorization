import base64
import json
from textwrap import dedent

import google.generativeai as genai
import PIL.Image


def get_species_gemini(image, pano_address, model):
    """
    Identify the species of a tree from an image, given its latitude and longitude.
    Using Gemini API

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


def get_species_gpt(image, pano_address, client):
    """
    Identify the species of a tree from an image, given its latitude and longitude.
    Using GPT API

    Args:
        image (ndarray): Image of the tree in ndarray format.
        address (pano.address): address of image
        model (gpt model): gpt model

    Returns:
        tuple: A tuple containing the species name, common name, and a short description.
    """

    # convert image to jpg (dont save) then base64 encode
    image = PIL.Image.fromarray(image)
    temp_path = "./data/images/temp.jpg"
    image.save(temp_path)
    with open(temp_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    address = ""
    if pano_address:
        address = ", ".join([addr.value for addr in pano_address])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tree_species",
                "schema": {
                    "type": "object",
                    "properties": {
                        "family": {
                            "description": "The scientific family of the tree",
                            "type": "string",
                        },
                        "genus": {
                            "description": "The scientific genus of the tree",
                            "type": "string",
                        },
                        "species": {
                            "description": "The species of the tree",
                            "type": "string",
                        },
                        "common_name": {
                            "description": "The common name of the tree. Normalize these values so they are similar accross multiple predictions",
                            "type": "string",
                        },
                        "short_description": {
                            "description": "A short description of the tree",
                            "type": "string",
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": dedent(
                    """
                    You are a tree expert who has to give best guess on the species of the tree in the image given its address. Note that you are detecting trees in Chandigarh, India, where the most popular trees are given below (in order of popularity)
                        Chakrasia
                        Mahogany
                        Mulberry
                        Arjun
                        Mango
                        Peepal
                        Pilkhan
                        Fig
                        Jamun
                        Siri
                        Gulmohar
                        Neem
                        Banyan
                        Dek
                        Ashoka
                        Ber
                    """
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Identify the tree species in this image, given that it's located in {address}. Give your best guess.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            },
        ],
    )

    response_dict = response.to_dict()
    # pretty_print(response_dict)
    print("response_dict: ", response_dict)

    content = response.choices[0].message.content

    # convert content to dict
    content = json.loads(content)

    family = content.get("family")
    genus = content.get("genus")
    species = content.get("species")
    common_name = content.get("common_name")
    short_description = content.get("short_description")

    # make a response dict to return
    gpt_response = {
        "family": family,
        "genus": genus,
        "species": species,
        "common_name": common_name,
        "description": short_description,
        "usage": str(response.usage),
    }

    return gpt_response
    # return species, common_name, short_description

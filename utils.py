import json
from pathlib import Path

from PIL import Image


def load_image(path):
    image_f = Image.open(path)
    image = image_f.copy()
    image_f.close()
    return image


def load_images_and_payloads(folder_path: str):
    folder_path = Path(folder_path)
    images = []
    payloads = []
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix == ".jpg":
            image = load_image(file_path)
            images.append(image)

            json_filename = file_path.stem + ".json"
            json_path = folder_path / json_filename
            payload = {}
            if json_path.exists():
                with open(json_path, "r") as f:
                    payload = json.load(f)
            payload["path"] = str(file_path)
            payloads.append(payload)

    return images, payloads

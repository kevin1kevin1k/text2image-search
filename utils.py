import json
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

from constants import IMAGES_PATH


class ImagePayloadDataset(Dataset):
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.file_paths = []
        self.payloads = []

        for file_path in self.folder_path.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix == ".jpg":
                self.file_paths.append(file_path)

                payload = {"file_stem": file_path.stem}

                self.payloads.append(payload)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = read_image(str(file_path))
        payload = self.payloads[idx]
        if image.shape == (1, 256, 256):
            image = image.repeat(3, 1, 1)
        return {"image": image, "payload": payload}


def load_json_dict(file_path: str) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    with open(file_path, "r") as f:
        return json.load(f)


def _transform_json_list(json_list: str):
    return json.loads(json_list.replace("'", '"'))


def load_image_and_payload(file_stem):
    image = to_pil_image(read_image(f"{IMAGES_PATH}/{file_stem}.jpg"))
    payload = load_json_dict(f"{IMAGES_PATH}/{file_stem}.json")
    payload["image_labels"] = _transform_json_list(payload["image_labels"])
    payload["bbox_labels"] = _transform_json_list(payload["bbox_labels"])
    return image, payload

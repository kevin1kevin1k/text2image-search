import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


def load_image(path):
    image_f = Image.open(path)
    image = image_f.copy()
    image_f.close()
    return image


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

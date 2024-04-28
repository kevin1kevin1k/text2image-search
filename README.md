# Text-to-image Search

## Introduction

Sample project for semantic text-to-image search running locally.

## Installation

```
pip install pipenv  # skip if already installed
pipenv sync --dev
```

## Dataset preparation

```
pipenv shell  # skip if already in the env
bash ./prepare_images.sh
```

## Dataset analysis

Label distribution and some examples can be viewed at the notebook `analyze_dataset.ipynb`.

## Usage

### Launch qdrant docker in one tab

```
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### In another tab, create a collection and upload vectors

Note that this may take a while.

```
pipenv shell
python create_collection.py
```

### Start the service

TODO

## Query evaluation

TODO

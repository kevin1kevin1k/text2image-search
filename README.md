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

Continue from the previous shell and run the following.
A browser window will open with the app.
If not, open http://localhost:8501/ in a browser or follow the link in the terminal.

```
streamlit run serve.py
```

Now one can search images with arbitrary queries.
The similarity score threshold can be adjusted; note that higher thresholds may lead to fewer results.
The resulting images are accompanied with corresponding similarity scores, as well as the labels from the dataset.
The labels are only for reference, and are not used in the search process.

## Query evaluation

### Qualitative evaluation

The evaluations with various examples can be viewed at the notebook `evaluation.ipynb`.

### Quantitative evaluation

- Data preparation
    - Select a set of representative queries from the dataset label space and manually annotate relevant images for each query. This set will serve as the test/evaluation queries. Instead of manual annotation, relevant images could potentially be determined automatically based on some heuristics or another pre-trained model.
    - The queries should be diverse enough to cover different parts of the label space. A balanced set of queries will make the evaluation more robust and generalizable.
- Evaluation metric
    - Since the task is about retrieval of relevant images, Recall@K should be a standard evaluation metric for this task. If one also cares about the precision, F1 score can also be measured to balance precision and recall.
    - In addition, if the ranking order is important, Mean Reciprocal Rank (MRR) or Normalized Discounted Cumulative Gain (NDCG) can be used to evaluate how well the truly relevant images are ranked at the top.


## Challenges and potential improvements

- The current dataset size is relatively small (~10k images). With a larger and more diverse dataset, the result is expected to be better.
- The current image and text embeddings are from a pre-trained CLIP model. Fine-tuning the model on the target dataset may improve the quality of embeddings and hence the retrieval accuracy.
- Some parameters are chosen arbitrarily, such as the distance measure (cosine) and the image cropping size (256). Altering these values may also lead to different results.
- The current codebase is not modular and lacks abstraction. Refactoring it into reusable modules/classes and adding unit tests would make it more maintainable and extensible.

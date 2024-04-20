#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def process_data(
    test_images_file, annotations_file, class_descriptions_file, nrows=None
):
    test_images_df = pd.read_csv(test_images_file, nrows=nrows)
    annotations_df = pd.read_csv(annotations_file)

    class_descriptions = pd.read_csv(class_descriptions_file)
    label_display_map = dict(
        zip(class_descriptions["LabelName"], class_descriptions["DisplayName"])
    )

    confident_annotations = annotations_df[annotations_df["Confidence"] == 1]

    merged_df = pd.merge(
        confident_annotations, test_images_df[["ImageID", "OriginalURL"]], on="ImageID"
    )
    merged_df["DisplayName"] = merged_df["LabelName"].map(label_display_map)

    return merged_df[["OriginalURL", "DisplayName"]]


def group_labels_and_rename_column(df: pd.DataFrame, new_col_name: str):
    return (
        df.groupby("OriginalURL")["DisplayName"]
        .apply(list)
        .reset_index()
        .rename(
            columns={
                "DisplayName": new_col_name,
                "OriginalURL": "url",
            }
        )
    )


def main():
    image_labels_10000 = process_data(
        "test-images-with-rotation.csv",
        "oidv7-test-annotations-human-imagelabels.csv",
        "oidv7-class-descriptions.csv",
        nrows=10000,
    )
    bbox_labels_10000 = process_data(
        "test-images-with-rotation.csv",
        "test-annotations-human-imagelabels-boxable.csv",
        "oidv7-class-descriptions-boxable.csv",
        nrows=10000,
    )
    joined_df = pd.merge(
        group_labels_and_rename_column(image_labels_10000, "image_labels"),
        group_labels_and_rename_column(bbox_labels_10000, "bbox_labels"),
        on="url",
    )
    joined_df.head(10000).to_csv(
        "urls.csv", index=False, columns=["url", "image_labels", "bbox_labels"]
    )


if __name__ == "__main__":
    main()

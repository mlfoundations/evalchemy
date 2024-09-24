import csv
import logging
import os

from datasets import Dataset


def count_characters(dataset: Dataset, columns_to_count: list[str] = None, prefix: str = "") -> Dataset:
    if columns_to_count is None:
        columns_to_count = dataset.column_names

    def count_chars_in_row(row):
        return {
            f"{prefix}{col}_char_count": len(str(row[col])) for col in columns_to_count if col in dataset.column_names
        }

    dataset = dataset.map(count_chars_in_row)
    return dataset


def save_to_csv(dataset: Dataset, filename: str = "example.csv") -> Dataset:
    dataset.to_csv(filename, index=False)
    logging.info(f"Dataset saved to {filename}")

    return dataset  #

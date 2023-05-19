import os
import numpy as np


def get_all_data_file_paths(data_folder_path):

    data_files = [
        os.path.join(data_folder_path, f)
        for f in os.listdir(data_folder_path)
        if os.path.isfile(os.path.join(data_folder_path, f))
    ]

    return data_files


def get_confusion_matrix (num_class, actual_labels, predicted_labels):

    confusion_matrix = np.zeros((num_class, num_class))
    for (actual_class, predicted_class) in zip(actual_labels, predicted_labels):
        confusion_matrix[actual_class, predicted_class] += 1

    class_count = np.zeros(num_class)
    for label in actual_labels:
        class_count[label] += 1

    for row_idx in range(num_class):
        confusion_matrix[row_idx] /= class_count[row_idx]

    return confusion_matrix


def quantize(arr: np.ndarray, levels: int, min_value_old: int, max_value_old: int, min_val_new: int) -> np.ndarray:
    level_size = (max_value_old - min_value_old)/levels

    for level_idx in range(levels):

        level_min_threshold = min_value_old + level_idx * level_size
        level_max_threshold = min_value_old + (level_idx + 1) * level_size

        level_value = min_val_new + level_idx

        arr[(level_min_threshold <= arr) & (arr < level_max_threshold)] = level_value

    return arr

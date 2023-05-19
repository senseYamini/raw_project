import preprocess
import utility

import numpy as np


def get_csi_multi_files (folder_path):

    data_files = utility.get_all_data_file_paths(folder_path)

    all_csi_amp_mat = []
    for file in data_files:
        csi_df = preprocess.get_csi_df_from_file(file)
        csi_amp_mat, _ = preprocess.get_filtered_csi_amp_and_phase_matrix(csi_df)
        all_csi_amp_mat.append(csi_amp_mat)

    csi_amp_mat = np.concatenate(all_csi_amp_mat)

    return csi_amp_mat


def get_csi_single_file (file_path):

    csi_df = preprocess.get_csi_df_from_file(file_path)
    csi_amp_mat, _ = preprocess.get_filtered_csi_amp_and_phase_matrix(csi_df)

    return csi_amp_mat


def get_csi_images (csi_mat, image_length):

    images = np.split(
        csi_mat,
        np.arange(image_length, len(csi_mat), image_length)
    )

    images = images[:len(images) - 1]

    return images


def flatten_csi_images (csi_images):

    images_flattened_mat = np.reshape(csi_images, (csi_images.shape[0], csi_images.shape[1] * csi_images.shape[2]))
    images_flattened_mat = images_flattened_mat.transpose()

    return images_flattened_mat

def get_low_dim_class_images (low_dim_ortho_space, csi_images_flattened, labels, class_label):

    flatten_class_images = np.squeeze(csi_images_flattened[:, np.where(labels==class_label)])
    class_label_low_dim_images = low_dim_ortho_space @ flatten_class_images

    return class_label_low_dim_images
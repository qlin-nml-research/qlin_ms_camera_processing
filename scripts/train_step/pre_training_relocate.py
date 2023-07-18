import os
import random
import shutil

import cv2
from tqdm import tqdm
import numpy as np


def run(
        total_original_img=None,
        total_augmented_img=None,
        raw_resized_path=None,
        mask_resized_path=None,
        raw_augmented_path=None,
        mask_augmented_path=None,
        dataset_path=None,
        dataset_mask_path=None,
):

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(dataset_mask_path, exist_ok=True)


    current_i = 0
    for i in tqdm(range(current_i, original_num_for_train), desc="original to train"):
        copy_files(raw_resized_path, mask_resized_path,
                   dataset_train_path, dataset_train_mask_path, raw_list[i] + 1, i + 1)
    print(str(original_num_for_train) + ' original images copied to train and train_mask folder!')
    current_i += original_num_for_train

    for i in tqdm(range(current_i, current_i + original_num_for_test), desc="original to test"):
        copy_files(raw_resized_path, mask_resized_path,
                   dataset_test_path, dataset_test_mask_path, raw_list[i] + 1,
                   i - original_num_for_train + 1)
    print(str(original_num_for_test) + ' original images copied to test and test_mask folder!')
    current_i += original_num_for_test

    for i in tqdm(range(current_i, current_i + original_num_for_val), desc="original to val"):
        copy_files(raw_resized_path, mask_resized_path,
                   dataset_val_path, dataset_val_mask_path, raw_list[i] + 1,
                   i - original_num_for_train - original_num_for_test + 1)
    print(str(original_num_for_val) + ' original images copied to val and val_mask folder!')
    current_i += original_num_for_val

    current_i = 0
    for i in tqdm(range(current_i, aug_num_for_train), desc="augmented to train"):
        copy_files(raw_augmented_path, mask_augmented_path,
                   dataset_train_path, dataset_train_mask_path,
                   aug_list[i] + total_original_img + 1, i + original_num_for_train + 1)
    print(str(aug_num_for_train) + ' augmented images copied to train and train_mask folder!')
    current_i += aug_num_for_train

    for i in tqdm(range(current_i, current_i + aug_num_for_test), desc="augmented to test"):
        copy_files(raw_augmented_path, mask_augmented_path,
                   dataset_test_path, dataset_test_mask_path,
                   aug_list[i] + total_original_img + 1,
                   i - aug_num_for_train + original_num_for_test + 1)
    print(str(aug_num_for_test) + ' augmented images copied to test and test_mask folder!')
    current_i += aug_num_for_test

    for i in tqdm(range(current_i, current_i + aug_num_for_val), desc="augmented to val"):
        copy_files(raw_augmented_path, mask_augmented_path,
                   dataset_val_path, dataset_val_mask_path,
                   aug_list[i] + total_original_img + 1,
                   i - aug_num_for_train - aug_num_for_test + original_num_for_val + 1)
    print(str(aug_num_for_val) + ' augmented images copied to val and val_mask folder!')
    current_i += aug_num_for_val


def copy_files(get_raw_dir, get_mask_dir, output_raw_dir, output_mask_dir, input_iteration, output_iteration):
    shutil.copyfile(os.path.join(get_raw_dir, str(input_iteration) + '.png'),
                    os.path.join(output_raw_dir, str(output_iteration) + '.png'))
    shutil.copyfile(os.path.join(get_mask_dir, str(input_iteration) + '.png'),
                    os.path.join(output_mask_dir, str(output_iteration) + '.png'))

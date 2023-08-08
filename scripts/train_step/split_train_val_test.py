import glob
import os
import random
import shutil

import cv2
from tqdm import tqdm
import numpy as np


def run(
        original_num_for_train=None,
        aug_num_for_train=None,

        raw_resized_path=None,
        mask_resized_path=None,
        raw_augmented_path=None,
        mask_augmented_path=None,

        dataset_train_path=None,
        dataset_test_path=None,
        dataset_val_path=None,
        dataset_train_mask_path=None,
        dataset_test_mask_path=None,
        dataset_val_mask_path=None,
        val_test_ratio=None,
):
    # test to val ratio
    if val_test_ratio is None:
        val_test_ratio = 0.8

    original_img_list = glob.glob(os.path.join(raw_resized_path, "*png"))
    augmented_img_list = glob.glob(os.path.join(raw_augmented_path, "*png"))
    original_mask_list = [os.path.join(mask_resized_path, os.path.basename(entry)) for entry in original_img_list]
    augmented_mask_list = [os.path.join(mask_augmented_path, os.path.basename(entry)) for entry in augmented_img_list]

    # sanity_check
    for raw_path, mask_path in zip(original_img_list, original_mask_list):
        assert os.path.isfile(raw_path), "raw_path not found for: " + raw_path
        assert os.path.isfile(mask_path), "mask_path not found for: " + mask_path
    for raw_path, mask_path in zip(augmented_img_list, augmented_mask_list):
        assert os.path.isfile(raw_path), "raw_path not found for: " + raw_path
        assert os.path.isfile(mask_path), "mask_path not found for: " + mask_path

    assert len(original_img_list) - 100 > original_num_for_train, "insufficient images:" \
                                                                  + " total original: " + str(len(original_img_list)) \
                                                                  + " train: " + str(original_num_for_train)
    assert len(augmented_img_list) - 100 > aug_num_for_train, "insufficient images:" \
                                                              + " total augment: " + str(len(augmented_img_list)) \
                                                              + " train: " + str(aug_num_for_train)

    original_num_for_val = int((len(original_img_list) - original_num_for_train) * val_test_ratio)
    original_num_for_test = len(original_img_list) - original_num_for_train - original_num_for_val

    aug_num_for_val = int((len(augmented_img_list) - aug_num_for_train) * val_test_ratio)
    aug_num_for_test = len(augmented_img_list) - aug_num_for_train - aug_num_for_val

    original_list = list(zip(original_img_list, original_mask_list))
    augmented_list = list(zip(augmented_img_list, augmented_mask_list))
    random.shuffle(original_list)
    random.shuffle(augmented_list)

    ori_train_list, ori_test_list, ori_val_list = split_list(original_list, original_num_for_train,
                                                             original_num_for_test, original_num_for_val)
    aug_train_list, aug_test_list, aug_val_list = split_list(augmented_list, aug_num_for_train,
                                                             aug_num_for_test, aug_num_for_val)

    train_list = ori_train_list + aug_train_list
    test_list = ori_test_list + aug_test_list
    val_list = ori_val_list + aug_val_list

    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(val_list)

    os.makedirs(dataset_train_path, exist_ok=True)
    os.makedirs(dataset_test_path, exist_ok=True)
    os.makedirs(dataset_val_path, exist_ok=True)
    os.makedirs(dataset_train_mask_path, exist_ok=True)
    os.makedirs(dataset_test_mask_path, exist_ok=True)
    os.makedirs(dataset_val_mask_path, exist_ok=True)

    for i, (raw_path, mask_path) in enumerate(tqdm(train_list, desc="original and augment to Train")):
        copy_files(raw_path, mask_path,
                   dataset_train_path, dataset_train_mask_path,
                   i + 1)

    for i, (raw_path, mask_path) in enumerate(tqdm(test_list, desc="original and augment to Test")):
        copy_files(raw_path, mask_path,
                   dataset_test_path, dataset_test_mask_path,
                   i + 1)

    for i, (raw_path, mask_path) in enumerate(tqdm(val_list, desc="original and augment to Val")):
        copy_files(raw_path, mask_path,
                   dataset_val_path, dataset_val_mask_path,
                   i + 1)


def split_list(input_list: list[str], num_train: int, num_test: int, num_val: int):
    conter = 0
    train_list = input_list[conter:num_train]
    conter += num_train
    test_list = input_list[conter:conter + num_test]
    conter += num_test
    val_list = input_list[conter:conter + num_val]
    assert num_train == len(train_list), "something went wrong in split"
    assert num_test == len(test_list), "something went wrong in split"
    assert num_val == len(val_list), "something went wrong in split"
    assert conter + num_val == len(input_list), "something went wrong in split"
    return train_list, test_list, val_list


def copy_files(input_raw_file_path, input_mask_file_path, output_raw_dir, output_mask_dir, output_numbering):
    shutil.copyfile(input_raw_file_path,
                    os.path.join(output_raw_dir, str(output_numbering) + '.png'))
    shutil.copyfile(input_mask_file_path,
                    os.path.join(output_mask_dir, str(output_numbering) + '.png'))

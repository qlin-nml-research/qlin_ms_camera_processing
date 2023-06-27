import os
import random
import cv2
from tqdm import tqdm

from tools import functions as fn


def augmentation(parameters=None):

    image_num_after_augmentation = parameters.image_num_after_augmentation

    original_num = int(len(os.listdir(parameters.json_and_raw_path)) / 2)

    augmentation_num = image_num_after_augmentation - original_num

    transforms = fn.get_augmentation()

    if not os.path.exists(parameters.raw_augmented_path):
        os.mkdir(parameters.raw_augmented_path)

    if not os.path.exists(parameters.mask_augmented_path):
        os.mkdir(parameters.mask_augmented_path)

    k = 0
    while k < augmentation_num:
        k = k + 1
        print('augmenting...: ' + str(k) + '/' + str(augmentation_num))
        raw_original, mask_original = fn.get_random_choice_images(parameters.raw_resized_path,
                                                                  parameters.mask_resized_path,
                                                                  original_num)
        augmented = transforms(image=raw_original, mask=mask_original)
        raw_augmented, mask_augmented = augmented['image'], augmented['mask']
        fn.save_augmented_images(parameters.raw_augmented_path, parameters.mask_augmented_path, raw_augmented,
                                 mask_original, k,
                                 original_num)
        del raw_augmented, mask_augmented, augmented, raw_original, mask_original


def test_train_validation_split(
        parameters=None,
        dataset_train_path=None,
        dataset_test_path=None,
        dataset_val_path=None,
        dataset_train_mask_path=None,
        dataset_test_mask_path=None,
        dataset_val_mask_path=None,
):
    image_num_after_augmentation = parameters.image_num_after_augmentation

    original_num = int(len(os.listdir(parameters.json_and_raw_path)) / 2)
    original_num_for_train = parameters.original_num_for_train
    original_num_for_test = int((original_num - original_num_for_train) / 2)
    original_num_for_val = original_num_for_test

    augmentation_num = image_num_after_augmentation - original_num
    aug_num_for_train = parameters.aug_num_for_train
    aug_num_for_test = int((augmentation_num - aug_num_for_train) / 2)
    aug_num_for_val = aug_num_for_test

    raw_list = list(range(original_num))
    random.shuffle(raw_list)

    aug_list = list(range(augmentation_num))
    random.shuffle(aug_list)

    os.makedirs(dataset_train_path, exist_ok=True)
    os.makedirs(dataset_test_path, exist_ok=True)
    os.makedirs(dataset_val_path, exist_ok=True)
    os.makedirs(dataset_train_mask_path, exist_ok=True)
    os.makedirs(dataset_test_mask_path, exist_ok=True)
    os.makedirs(dataset_val_mask_path, exist_ok=True)

    current_i = 0
    for i in tqdm(range(current_i, original_num_for_train), desc="original to train"):
        fn.copy_files(parameters.raw_resized_path, parameters.mask_resized_path,
                      parameters.dataset_train_path, dataset_train_mask_path, raw_list[i] + 1, i + 1)
    print(str(original_num_for_train) + ' original images copied to train and train_mask folder!')
    current_i += original_num_for_train

    for i in tqdm(range(current_i, current_i+original_num_for_test), desc="original to test"):
        fn.copy_files(parameters.raw_resized_path, parameters.mask_resized_path,
                      parameters.dataset_test_path, dataset_test_mask_path, raw_list[i] + 1,
                      i - original_num_for_train + 1)
    print(str(original_num_for_test) + ' original images copied to test and test_mask folder!')
    current_i += original_num_for_test

    for i in tqdm(range(current_i, current_i+original_num_for_val), desc="original to val"):
        fn.copy_files(parameters.raw_resized_path, parameters.mask_resized_path,
                      parameters.dataset_val_path, dataset_val_mask_path, raw_list[i] + 1,
                      i - original_num_for_train - original_num_for_test + 1)
    print(str(original_num_for_val) + ' original images copied to val and val_mask folder!')
    current_i += original_num_for_val

    current_i = 0
    for i in tqdm(range(current_i, aug_num_for_train), desc="augmented to train"):
        fn.copy_files(parameters.raw_augmented_path, parameters.mask_augmented_path,
                      parameters.dataset_train_path, dataset_train_mask_path,
                      aug_list[i] + original_num + 1, i + original_num_for_train + 1)
    print(str(aug_num_for_train) + ' augmented images copied to train and train_mask folder!')
    current_i += aug_num_for_train

    for i in tqdm(range(current_i, current_i+aug_num_for_test), desc="augmented to test"):
        fn.copy_files(parameters.raw_augmented_path, parameters.mask_augmented_path,
                      parameters.dataset_test_path, dataset_test_mask_path,
                      aug_list[i] + original_num + 1,
                      i - aug_num_for_train + original_num_for_test + 1)
    print(str(aug_num_for_test) + ' augmented images copied to test and test_mask folder!')
    current_i += aug_num_for_test

    for i in tqdm(range(current_i, current_i+aug_num_for_val), desc="augmented to val"):
        fn.copy_files(parameters.raw_augmented_path, parameters.mask_augmented_path,
                      parameters.dataset_val_path, dataset_val_mask_path,
                      aug_list[i] + original_num + 1,
                      i - aug_num_for_train - aug_num_for_test + original_num_for_val + 1)
    print(str(aug_num_for_val) + ' augmented images copied to val and val_mask folder!')
    current_i += aug_num_for_val


if __name__ == '__main__':
    augmentation()

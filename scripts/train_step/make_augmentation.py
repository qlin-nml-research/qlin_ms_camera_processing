import os
import random
import cv2
from tqdm import tqdm
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool


def aug_task(
        k, original_num, augmentation_preset,
        raw_resized_path, mask_resized_path, raw_augmented_path, mask_augmented_path,
):
    raw_original, mask_original = get_random_choice_images(raw_resized_path,
                                                           mask_resized_path,
                                                           original_num)
    augmented = augmentation_preset(image=raw_original, mask=mask_original)
    raw_augmented, mask_augmented = augmented['image'], augmented['mask']

    raw_output_path = os.path.join(raw_augmented_path, str(k + original_num) + '.png')
    cv2.imwrite(raw_output_path, raw_augmented)
    mask_output_path1 = os.path.join(mask_augmented_path, str(k + original_num) + '.png')
    cv2.imwrite(mask_output_path1, mask_augmented[:, :, 0])


def run(
        json_and_raw_path=None,
        raw_resized_path=None,
        raw_augmented_path=None,
        mask_resized_path=None,
        mask_augmented_path=None,
        image_num_after_augmentation=None,
        augmentation_preset=None,
):
    original_num = int(len(os.listdir(json_and_raw_path)) / 2)
    # print(os.listdir(json_and_raw_path))
    # print(json_and_raw_path)

    augmentation_num = image_num_after_augmentation - original_num

    os.makedirs(raw_augmented_path, exist_ok=True)
    os.makedirs(mask_augmented_path, exist_ok=True)
    os.makedirs(raw_resized_path, exist_ok=True)
    os.makedirs(mask_resized_path, exist_ok=True)

    # for k in tqdm(range(0, augmentation_num), desc="Augmentation Progress"):
    print("Starting parallel augmentation: ")
    # pool = Pool(processes=30)
    with Pool(processes=30) as pool:
        for k in range(0, augmentation_num + 1):
            pool.apply_async(
                aug_task, args=(
                    k, original_num, augmentation_preset,
                    raw_resized_path, mask_resized_path,
                    raw_augmented_path, mask_augmented_path,
                )
            )
        pool.close()
        pool.join()

    print("Done Augmentation")


def get_random_choice_images(input_raw_dir, input_mask_dir, total_image_num):
    n = np.random.choice(total_image_num - 1) + 1
    raw = cv2.imread(os.path.join(input_raw_dir, str(n) + '.png'))
    shape = np.shape(raw)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    mask = np.zeros([shape[0], shape[1], 1])
    mask[:, :, 0] = cv2.imread(os.path.join(input_mask_dir, str(n) + '.png'), 0)
    return raw, mask


if __name__ == '__main__':
    import sys

    sys.path.append("E:/Github/qlin2023/Moonshot/catkin_ws/src/qlin_camera_processing/scripts")
    from config import training_param

    train_param = training_param.TrainingParameters()
    run(
        json_and_raw_path=train_param.json_and_raw_path,
        raw_resized_path=train_param.raw_resized_path,
        raw_augmented_path=train_param.raw_augmented_path,
        mask_resized_path=train_param.mask_resized_path,
        mask_augmented_path=train_param.mask_augmented_path,
        image_num_after_augmentation=train_param.image_num_after_augmentation,
        augmentation_preset=train_param.augmentation_preset
    )

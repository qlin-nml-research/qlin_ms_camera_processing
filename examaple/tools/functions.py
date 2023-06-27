import matplotlib.pyplot as plt
import albumentations as albu
import numpy as np
import cv2
import shutil
from labelme import utils
from scipy.ndimage.filters import minimum_filter
import json
import torch
import os


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def get_gaussian_confidence_map(w, h, tip_x, tip_y, sigma):
    width = np.arange(0, w, 1)
    height = np.arange(0, h, 1)
    X, Y = np.meshgrid(width, height)
    mu = np.array([tip_x, tip_y])
    S = np.array([[sigma, 0], [0, sigma]])

    x_norm = (np.array([X, Y]) - mu[:, None, None]).transpose(1, 2, 0)
    heatmap = np.exp(- x_norm[:, :, None, :] @ np.linalg.inv(S)[None, None, :, :] @ x_norm[:, :, :, None] / 2.0) * 255

    return heatmap[:, :, 0, 0]


def make_mask_image_confidence_map(w, h, x, y, sigma, path):
    img = get_gaussian_confidence_map(w, h, x, y, sigma)
    cv2.imwrite(path, img)
    return img


def make_empty_mask_image_confidence_map(w, h, path):
    img = np.zeros([h, w])
    cv2.imwrite(path, img)


def get_drill_tip_from_json(image_path):
    with open(image_path, "r", encoding="utf-8") as f:
        dj = json.load(f)

    if len(dj['shapes']) != 0:
        if dj['shapes'][0]['label'] == 'drill_tip':
            points = dj['shapes'][0]['points']
        elif dj['shapes'][1]['label'] == 'drill_tip':
            points = dj['shapes'][1]['points']
        else:
            points = dj['shapes'][2]['points']
        xy = [tuple(point) for point in points][0]
    else:
        xy = (None, None)

    return xy


def get_image_size(image_path):
    with open(image_path, "r", encoding="utf-8") as f:
        dj = json.load(f)
    w = dj['imageWidth']
    h = dj['imageHeight']
    return w, h





def save_augmented_images(save_raw_dir, save_mask_dir, raw_image, mask_image, iteration, first_num):
    raw_output_path = os.path.join(save_raw_dir, str(iteration + first_num) + '.png')
    cv2.imwrite(raw_output_path, raw_image)

    mask_output_path1 = os.path.join(save_mask_dir, str(iteration + first_num) + '.png')
    cv2.imwrite(mask_output_path1, mask_image[:, :, 0])


def get_random_choice_images(input_raw_dir, input_mask_dir, total_image_num):
    n = np.random.choice(total_image_num - 1) + 1
    raw = cv2.imread(os.path.join(input_raw_dir, str(n) + '.png'))
    shape = np.shape(raw)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    mask = np.zeros([shape[0], shape[1], 1])
    mask[:, :, 0] = cv2.imread(os.path.join(input_mask_dir, str(n) + '.png'), 0)
    return raw, mask


def copy_files(get_raw_dir, get_mask_dir, output_raw_dir, output_mask_dir, input_iteration, output_iteration):
    shutil.copyfile(os.path.join(get_raw_dir, str(input_iteration) + '.png'),
                    os.path.join(output_raw_dir, str(output_iteration) + '.png'))
    shutil.copyfile(os.path.join(get_mask_dir, str(input_iteration) + '.png'),
                    os.path.join(output_mask_dir, str(output_iteration) + '.png'))


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def predict_and_mark(in_img, best_model, preprocessing, model_dev, output_size, postive_treshold):
    x_img = preprocessing(image=in_img)['image']
    x_tensor = torch.from_numpy(x_img).to(model_dev).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)  # torch.tensor
    pr_mask = (pr_mask.squeeze().cpu().numpy()) * 255
    pr_mask = np.transpose(pr_mask, (0, 1))

    tip_drill = np.array(np.unravel_index(np.argmax(pr_mask[:, :]), pr_mask[:, :].shape))

    if pr_mask[tuple(tip_drill)] >= postive_treshold:
        cv2.circle(in_img, (tip_drill[1], tip_drill[0]), 5, (255, 0, 0), -1)
    else:
        tip_drill = None

    return tip_drill, cv2.resize(in_img, output_size), cv2.resize(pr_mask, output_size)

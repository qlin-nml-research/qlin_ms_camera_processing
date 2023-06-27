import glob
import cv2
import os
import numpy as np
from tools import functions as fn


def make_labeled_image(
        parameters=None,
        json_and_raw_path=None,
        debug=None,
        debug_path=None,
        binary=None,
        raw_resized_path=None,
        mask_resized_path=None,
        resize_target=None,
        mask_sigma=None,
        threshold=None,
):

    image_list = glob.glob1(parameters.json_and_raw_path, "*.png")

    if parameters.debug:
        if not os.path.exists(parameters.debug_path):
            os.mkdir(parameters.debug_path)

    if not os.path.exists(parameters.raw_resized_path):
        os.mkdir(parameters.raw_resized_path)

    if not os.path.exists(parameters.mask_resized_path):
        os.mkdir(parameters.mask_resized_path)

    for img_name in image_list:
        num_str = img_name.replace(".png", "")
        k = int(num_str)

        # define save path
        input_img_path = os.path.join(parameters.json_and_raw_path, str(k) + '.png')
        output_heatmap_path = os.path.join(parameters.mask_resized_path, str(k) + '.png')
        output_resized_img_path = os.path.join(parameters.raw_resized_path, str(k) + '.png')
        output_debug_img_path = os.path.join(parameters.debug_path, str(k) + '-debug' + '.png')
        output_original_img_path = os.path.join(parameters.raw_resized_path, str(k) + '-original.png')

        print("processing:", k)
        if parameters.binary:
            image = cv2.imread(input_img_path, 0)
            image_original = cv2.imread(input_img_path)
        else:
            image = cv2.imread(input_img_path)

        input_json_path = os.path.join(parameters.json_and_raw_path, str(k) + '.json')

        w, h = fn.get_image_size(input_json_path)
        resized_w, resized_h = parameters.resize_target
        x_tip, y_tip = fn.get_drill_tip_from_json(input_json_path)

        print("resized: " + str(int(h)) + "×" + str(int(h)) + " -> " + str(parameters.resize_target))
        # resizing image
        image_resized = cv2.resize(image, parameters.resize_target)
        debug_image = image_resized.copy()
        if parameters.binary:
            image_original_resized = cv2.resize(image_original, parameters.resize_target)

        # point does not exist
        if x_tip is None or y_tip is None:
            fn.make_empty_mask_image_confidence_map(resized_w, resized_h, output_heatmap_path)

        else:
            x_tip = int(float(x_tip) / w * resized_w)
            y_tip = int(float(y_tip) / h * resized_h)

            tip_drill = np.array([x_tip, y_tip])

            if parameters.debug:
                cv2.circle(debug_image, (int(x_tip), int(y_tip)), 10, (255, 0, 0), -1)

            if parameters.binary:
                ret, image_resized = cv2.threshold(image_resized, parameters.threshold, 255, cv2.THRESH_BINARY)

            fn.make_mask_image_confidence_map(resized_w, resized_h,
                                              x_tip, y_tip,
                                              parameters.mask_sigma, output_heatmap_path)

        # save image
        cv2.imwrite(output_resized_img_path, image_resized)
        if parameters.debug:
            cv2.imwrite(output_debug_img_path, debug_image)
        if parameters.binary:
            cv2.imwrite(output_original_img_path, image_original_resized)

if __name__ == '__main__':
    make_labeled_image()

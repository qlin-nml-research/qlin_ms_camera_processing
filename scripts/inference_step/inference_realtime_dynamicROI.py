import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import time
from tqdm import tqdm
from enum import Enum, auto

from .network_common import get_preprocessing
from .inference_common import predict_and_mark

import logging

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class InferencerDROI:
    class ROIMode(Enum):
        MODE_1 = auto()  # x0.7 network w
        MODE_2 = auto()  # x1 network w
        MODE_3 = auto()  # x2 network w
        MODE_FULL = auto()  # full picture

    def _get_roi_scaling(self, mode):
        if mode is self.ROIMode.MODE_1:
            return 0.9
        elif mode is self.ROIMode.MODE_2:
            return 1
        elif mode is self.ROIMode.MODE_3:
            return 2
        else:
            return 10000

    def __init__(self, model_path, network_img_size, inference_param, show_img=False):
        self.model_path = model_path
        self.network_img_size = network_img_size
        self.inference_param = inference_param
        self.show_img = show_img

        self.best_model = torch.load(model_path, map_location=torch.device(inference_param['DEVICE']))
        logger.info("model loaded!")
        ENCODER = inference_param['ENCODER']
        ENCODER_WEIGHTS = inference_param['ENCODER_WEIGHTS']

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        self.preprocessing = get_preprocessing(preprocessing_fn)

        self.initialized = False
        self.roi_center = None
        self.debug_roi = None
        self.roi_mode = self.ROIMode.MODE_FULL

    def process_frame(self, frame_raw, debug=False, show_img=False, show_img_size=[960, 540]):
        h, w, c = frame_raw.shape
        if not self.initialized:
            self.debug_roi = ([0, 0], [w, h])
            self.roi_center = np.array([int(w / 2), int(h / 2)])
            self.initialized = True

        tip_drill, mask_output = self._run_predict(frame_raw, debug=debug)

        if show_img:
            frame_labeled = frame_raw.copy()
            if tip_drill is not None:
                cv2.circle(frame_labeled, (int(tip_drill[0]), int(tip_drill[1])), 10, (0, 0, 255), -1)
            if debug:
                cv2.rectangle(frame_labeled, self.debug_roi[0],
                              np.array(self.debug_roi[0]) + np.array(self.debug_roi[1]),
                              (0, 255, 0), 2)

            cv2.imshow('label', cv2.resize(frame_labeled, show_img_size))
            cv2.imshow('mask', cv2.resize(mask_output / 255.0, show_img_size))
            cv2.waitKey(1)

        return tip_drill

    def _increment_roi_mode(self):
        if self.roi_mode is self.ROIMode.MODE_1:
            self.roi_mode = self.ROIMode.MODE_2
        elif self.roi_mode is self.ROIMode.MODE_2:
            self.roi_mode = self.ROIMode.MODE_3
        elif self.roi_mode is self.ROIMode.MODE_3:
            self.roi_mode = self.ROIMode.MODE_FULL
        else:
            pass

    def _decrement_roi_mode(self):
        if self.roi_mode is self.ROIMode.MODE_FULL:
            self.roi_mode = self.ROIMode.MODE_3
        elif self.roi_mode is self.ROIMode.MODE_3:
            self.roi_mode = self.ROIMode.MODE_2
        elif self.roi_mode is self.ROIMode.MODE_2:
            self.roi_mode = self.ROIMode.MODE_1
        else:
            pass

    def _pad_mask(self, mask_in, crop_wh, crop_xy, target_wh):
        # h, w = mask_in.shape
        return np.pad(mask_in, ((crop_xy[1], target_wh[1] - crop_wh[1] - crop_xy[1]),
                                (crop_xy[0], target_wh[0] - crop_wh[0] - crop_xy[0])))

    def _run_predict(self, frame_raw, mask_output=None, expand_roi=False, debug=False):
        if expand_roi:
            self._increment_roi_mode()

        h, w, _ = frame_raw.shape

        cropped_frame, crop_xy, crop_wh = self._get_roi_image(self.roi_mode, self.roi_center.astype(int), frame_raw)
        self.debug_roi = (crop_xy, crop_wh)
        if debug:
            cv2.imshow("debug roi crop", cropped_frame)

        tip_drill, _mask_output = predict_and_mark(
            in_frame=cropped_frame,
            best_model=self.best_model,
            preprocessing=self.preprocessing,
            model_dev=self.inference_param['DEVICE'],
            network_img_size=self.network_img_size,
            postive_treshold=self.inference_param['postive_detect_threshold'],
        )
        if tip_drill is None:
            # recursive element
            if self.roi_mode is self.ROIMode.MODE_FULL:
                mask_output = self._pad_mask(_mask_output, crop_wh, crop_xy, [w, h])
                return tip_drill, mask_output
            if debug:
                logger.info("Target not found, expanding ROI")
            tip_drill, _mask_output = self._run_predict(frame_raw, _mask_output, expand_roi=True, debug=debug)
        else:
            tip_drill = np.array([tip_drill[0] + crop_xy[0], tip_drill[1] + crop_xy[1]])
            _mask_output = self._pad_mask(_mask_output, crop_wh, crop_xy, [w, h])
            self.roi_center = tip_drill.copy()
            self._decrement_roi_mode()  # shrink ROI in following run

        return tip_drill, _mask_output

    def _get_roi_image(self, roi_mode, roi_center, in_frame):
        # Extracting the input image dimensions
        image_height, image_width = in_frame.shape[:2]

        nn_w, nn_h = self.network_img_size

        scaling = self._get_roi_scaling(roi_mode)
        out_w = int(min(scaling * nn_w, image_width))
        out_h = int(min(scaling * nn_h, image_height))

        # Calculating the top-left and bottom-right coordinates of the RoI
        roi_center_x, roi_center_y = roi_center
        roi_half_width = out_w // 2
        roi_half_height = out_h // 2

        roi_center_x = max(roi_half_width, min(image_width - roi_half_width - 1, roi_center_x))
        roi_center_y = max(roi_half_height, min(image_height - roi_half_height - 1, roi_center_y))

        roi_top_left_x = roi_center_x - roi_half_width
        roi_top_left_y = roi_center_y - roi_half_height
        roi_bottom_right_x = roi_center_x + roi_half_width
        roi_bottom_right_y = roi_center_y + roi_half_height

        # Cropping the image based on the RoI coordinates
        cropped_image = in_frame[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x, :]

        return cropped_image, [roi_top_left_x, roi_top_left_y], [out_w, out_h]

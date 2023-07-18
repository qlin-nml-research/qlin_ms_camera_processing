#!/usr/bin/python3
import cv2
import os
import scipy.io as scio
import numpy as np

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from run_stream_cv2 import realtime_stream_main_cv2

cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ))
vid_path = "E:/ExperimentData/MSCameraAutomation/microlens_original/7-5_constraints.mp4"
# vid_path = "E:/ExperimentData/MSCameraAutomation/microlens_original/egg1-cropped.mp4"
if __name__ == '__main__':
    _show_img = True
    _debug = False
    cam_mat_file_path = "../config/cv2_correction_param.mat"
    cam_data = scio.loadmat(cam_mat_file_path)

    intrinsic_mat_ = cam_data['intrinsicMatrix']
    distortion_coeff_ = cam_data['distortionCoefficients']
    focal_length = cam_data['focalLength']
    cell_size_ = cam_data['cellSize']
    sensor_res = cam_data['sensorResolution']
    cam_param_ = {
        "intrinsic": intrinsic_mat_,
        "distort_coff": distortion_coeff_,
        # made up number
        "focal_length": focal_length.reshape(-1),
        "object_plane_h": 0.003,
        "object_plane_w": 0.005,
        "sensor_cell_size": cell_size_.reshape(-1),
        "native_resolution": sensor_res.reshape(-1),
    }

    # _device_id = 4  # Mooonshot Master PC
    _device_id = vid_path  # file

    _inference_param = {
        # "model_path": os.path.join(cw_base_path, "model", 'best_model_v2.pth'),
        "model_path": os.path.join(cw_base_path, "model", 'best_model_v1.pth'),
        "network_img_size": [768, 768],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 40
        },
    }

    realtime_stream_main_cv2(
        inference_param=_inference_param,
        cam_param=cam_param_,
        device_id=_device_id,
        show_img=_show_img,
        debug=_debug,
        port=21039,
        ip="10.198.113.138",
    )

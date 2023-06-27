import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import time
from tqdm import tqdm

from .network_common import get_preprocessing
from .inference_common import predict_and_mark


class Inferencer:
    def __init__(self, model_path, network_img_size, inference_param, show_img=False):
        self.model_path = model_path
        self.network_img_size = network_img_size
        self.inference_param = inference_param
        self.show_img = show_img

        self.best_model = torch.load(model_path)
        print("model loaded!")
        ENCODER = inference_param['ENCODER']
        ENCODER_WEIGHTS = inference_param['ENCODER_WEIGHTS']

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        self.preprocessing = get_preprocessing(preprocessing_fn)

    def process_frame(self, img_in):
        tip_drill, frame_output, mask_output = predict_and_mark(
            in_frame=img_in,
            best_model=self.best_model,
            preprocessing=self.preprocessing,
            model_dev=self.inference_param['DEVICE'],
            network_img_size=self.network_img_size,
            postive_treshold=self.inference_param['postive_detect_threshold'],
        )
        if self.show_img:
            cv2.imshow('image', frame_output)
            cv2.imshow('mask', mask_output / 255.0)
            cv2.waitKey(1)
        return tip_drill


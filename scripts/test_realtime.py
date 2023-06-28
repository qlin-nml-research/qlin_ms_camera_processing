import cv2
import os

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from run_realtime_cv2 import realtime_main_cv2

cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ))
vid_path = "E:/ExperimentData/MSCameraAutomation/microlens_original/mouse1-cropped.mp4"
# vid_path = "E:/ExperimentData/MSCameraAutomation/microlens_original/egg1-cropped.mp4"
if __name__ == '__main__':
    _show_img = True
    _debug = False

    # _device_id = 4  # Mooonshot Master PC
    _device_id = vid_path  # file

    _inference_param = {
        "model_path": os.path.join(cw_base_path, "model", 'best_model.pth'),
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

    realtime_main_cv2(
        inference_param=_inference_param,
        device_id=_device_id,
        show_img=_show_img,
        debug=_debug,
    )

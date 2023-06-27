import cv2
import os

from inference_step.inference_realtime import Inferencer


def realtime_main(inference_param, device_id):
    inference_h = Inferencer(**inference_param)

    cap = cv2.VideoCapture(device_id)

    if cap.isOpened():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("current", frame)

                tip_pos = inference_h.process_frame(frame)

                print(tip_pos)

    except KeyboardInterrupt:
        print("Exit on Interrupt")

    cap.release()


cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ))
if __name__ == '__main__':
    show_img = True

    _device_id = 4  # Mooonshot Master PC

    _inference_param = {
        "model_path": os.path.join(cw_base_path, "model", 'best_model_4K_960.pth'),
        # "model_path": os.path.join(cw_base_path, "model", 'best_model_4K.pth'),
        "network_img_size": [960, 544],
        # "network_img_size": [1920, 1088],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 80
        },
        "show_img": show_img,
    }

    realtime_main(
        inference_param=_inference_param,
        device_id=_device_id,
    )

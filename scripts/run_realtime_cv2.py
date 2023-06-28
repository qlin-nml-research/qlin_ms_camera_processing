import cv2
import os

from inference_step.inference_realtime_dynamicROI import InferencerDROI


def realtime_main_cv2(inference_param, device_id, show_img, debug):
    inference_h = InferencerDROI(**inference_param)

    cap = cv2.VideoCapture(device_id)

    if cap.isOpened():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("current", frame)

                tip_pos = inference_h.process_frame(frame, debug=debug, show_img=show_img)

                print(tip_pos)
            else:
                break
    except KeyboardInterrupt:
        print("Exit on Interrupt")

    cap.release()


cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ))
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

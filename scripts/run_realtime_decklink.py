import cv2
import os

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from decklink.AR_CoppeliaSim_Py.decklink_interface import DeckLinkInterface

dll_path = "E:/Github/qlin2023/Moonshot/catkin_ws/src/qlin_camera_processing/scripts/decklink" \
           "/DecklinkInterface/x64/Release/DecklinkInterface.dll"
def realtime_main_decklink(inference_param, deck_link_index, display_mode_index, show_img, debug):
    inference_h = InferencerDROI(**inference_param)

    cap = DeckLinkInterface(dll_location=dll_path)
    cap.start_capture_thread(deck_link_index=deck_link_index, display_mode_index=display_mode_index)

    if cap.is_capture_running():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    try:
        while cap.is_capture_running():
            frame = cap.get_image()
            if frame is not None:
                cv2.imshow("current", frame)

                tip_pos = inference_h.process_frame(frame)

                print(tip_pos)

    except KeyboardInterrupt:
        print("Exit on Interrupt")

    cap.stop_capture_thread()


cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..",))
if __name__ == '__main__':

    _show_img = True
    _debug = False

    _inference_param = {
        "model_path": os.path.join(cw_base_path, "model", 'best_model_4K_960.pth'),
        "network_img_size": [960, 544],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 255.0 / 2
        },
        "show_img": show_img,
    }

    realtime_main_decklink(
        inference_param=_inference_param,
        deck_link_index=2,
        display_mode_index=1,
        show_img=_show_img,
        debug=_debug,
    )

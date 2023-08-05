import cv2
import numpy as np
import os
from functools import partial

from tqdm import tqdm
import sys
sys.path.append("..")

from inference_step.inference_realtime_dynamicROI import InferencerDROI

TARGET_DISPLAY_SIZE = [960, 540]


def key_point_processing_main(inference_param, file_path):
    timestamp_f = open(file_path + "_time.txt", 'r')

    start_time = float(timestamp_f.readline().strip())
    reader_timestamps = np.array([0.0] + [float(time_str.strip()) - start_time for time_str in timestamp_f])

    keypoint_dict = {}

    vid_reader = cv2.VideoCapture(file_path + ".mp4")
    if not vid_reader.isOpened():
        raise RuntimeError("video file failled to open")

    vid_w, vid_h = vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH), vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("video resolution is:", vid_w, vid_h)

    reader_frame_count = 0
    reader_total_frames = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(reader_timestamps) != reader_total_frames:
        print("beware of mismatch timestamp file", len(reader_timestamps), reader_total_frames)

    inference_h = InferencerDROI(**inference_param)

    ind = -1
    while vid_reader.isOpened():
        success, frame_raw = vid_reader.read()
        if not success:
            # raise RuntimeError("something went wrong")
            print("something went wrong, exiting....")
            break
        else:
            ind += 1

        tip_pos = inference_h.process_frame(frame_raw, debug=False, show_img=False,
                                            show_img_size=TARGET_DISPLAY_SIZE)

        frame = frame_raw.copy()
        if tip_pos is not None:
            tip_pos = np.round(tip_pos).astype(np.int32).tolist()
            print(ind, tip_pos)
            cv2.circle(frame, tip_pos, 20, (0, 0, 255), -1)
            if ind >= len(reader_timestamps):
                break
            keypoint_dict[ind] = [reader_timestamps[ind], (tip_pos[0], tip_pos[1])]

        cv2.imshow('Video', frame)

        key = cv2.waitKey(2)
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == 13:  # Press 'Enter' to pause/resume
            print("proceeding next frame")

    with open(file_path + "_auto_keypoint.txt", 'w') as keypoint_writer_f:
        ind_list = keypoint_dict.keys()
        sorted_ind_list = sorted(ind_list)
        for ind in sorted_ind_list:
            time = keypoint_dict[ind][0]
            x, y = keypoint_dict[ind][1]
            keypoint_writer_f.write('{:d},{:.6f},{:d},{:d}\n'.format(ind, time, x, y))

    vid_reader.release()
    cv2.destroyAllWindows()


cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "../..", ))
if __name__ == '__main__':
    dir_root = "/Users/quentinlin/Desktop/Moonshot/2023_08_04_experiment_recording"
    filename = "adapt_0804_exp1_vid_original"
    # filename = "adapt_lock_R1_0804_exp1_vid_original"
    # filename = "no_adapt_0804_exp1_vid_original"

    _inference_param = {
        # "model_path": os.path.join(cw_base_path, "model", 'best_model_960.pth'),
        "model_path": os.path.join(cw_base_path, "model", 'best_model.pth'),
        # "network_img_size": [960, 544],
        "network_img_size": [768, 768],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cpu",
            "postive_detect_threshold": 70
        },
    }

    key_point_processing_main(
        inference_param=_inference_param,
        file_path=os.path.join(dir_root, filename),
    )

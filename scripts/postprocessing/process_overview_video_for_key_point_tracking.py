import cv2
import numpy as np
import os
from functools import partial

from tqdm import tqdm
import sys

sys.path.append("..")

from inference_step.inference_realtime_dynamicROI import InferencerDROI

TARGET_DISPLAY_SIZE = [960, 540]


def key_point_processing_main(inference_param, file_path, output_file_path, crop_region):
    keypoint_dict = {}

    vid_reader = cv2.VideoCapture(file_path)
    print(file_path)
    if not vid_reader.isOpened():
        raise RuntimeError("video file failled to open")

    vid_w, vid_h = vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH), vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("video resolution is:", vid_w, vid_h)
    crop_w = (vid_w * np.array(crop_region[:2])).astype(np.int32)
    crop_h = (vid_h * np.array(crop_region[2:])).astype(np.int32)
    fps = vid_reader.get(cv2.CAP_PROP_FPS)

    reader_frame_count = 0
    reader_total_frames = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_h = InferencerDROI(**inference_param)

    ind = -1
    time_stamp = 0.0
    with tqdm(total=reader_total_frames) as progress_bar:
        while vid_reader.isOpened():
            success, frame_raw = vid_reader.read()
            if not success:
                # raise RuntimeError("something went wrong")
                print("something went wrong, exiting....")
                break
            else:
                ind += 1
            # print(frame_raw.shape)
            # print(vid_w, crop_w[0], crop_w[1])
            # print(vid_h, crop_h[0], crop_h[1])
            frame_cropped = frame_raw[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], :]

            tip_pos = inference_h.process_frame(frame_cropped, debug=False, show_img=False,
                                                show_img_size=TARGET_DISPLAY_SIZE)

            frame = frame_raw.copy()
            if tip_pos is not None:
                tip_pos[0] += crop_w[0]
                tip_pos[1] += crop_h[0]
                tip_pos = np.round(tip_pos).astype(np.int32).tolist()
                # print(ind, tip_pos)
                cv2.circle(frame, tip_pos, 6, (0, 0, 255), -1)

                keypoint_dict[ind] = [time_stamp, (tip_pos[0], tip_pos[1])]

            cv2.imshow('Video', cv2.resize(frame, TARGET_DISPLAY_SIZE))
            cv2.imshow('Video_cropped', cv2.resize(frame_cropped, TARGET_DISPLAY_SIZE))
            progress_bar.update(1)
            time_stamp += 1 / fps

            key = cv2.waitKey(1)
            if key == 27:  # Press 'Esc' to exit
                break

    with open(output_file_path + "_auto_keypoint.txt", 'w') as keypoint_writer_f:
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
    # dir_root = "E:/ExperimentData/MSCameraAutomation/8_14_experiment"
    dir_root = "C:/Users/qlin1/Public Box/8_14_experiment"
    filename = "adapt_2023-08-15_11-07-02.mkv"
    # filename = "adapt_lock_R1_0804_exp1_vid_original"
    # filename = "no_adapt_0804_exp1_vid_original"

    output_f_name = "adapt_overview_"

    _inference_param = {
        # "model_path": os.path.join(cw_base_path, "model", 'best_model_960.pth'),
        # "model_path": os.path.join(cw_base_path, "model", 'best_model.pth'),
        "model_path": os.path.join(cw_base_path, "model", 'best_model_576.pth'),
        # "network_img_size": [960, 544],
        # "network_img_size": [768, 768],
        "network_img_size": [576, 576],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 70
        },
    }

    key_point_processing_main(
        # x_start, x_end, y_start, y_end
        crop_region=[0, 0.7, 0, 1],
        inference_param=_inference_param,
        file_path=os.path.join(dir_root, filename),
        output_file_path=os.path.join(dir_root, output_f_name),
    )

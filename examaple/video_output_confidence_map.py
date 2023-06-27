import os
import torch
import numpy as np
import cv2
from tools import functions
import segmentation_models_pytorch as smp
import time
from tools import functions as fn
from tqdm import tqdm


def video_output_confidence_map(
        video_dir=None,
        video_list=None,
        model_path=None,
        output_dir=None,
        network_img_size=None,
        inference_param=None,
        print_time=False, show_img=False,
):
    best_model = torch.load(model_path)
    print("loaded!")
    ENCODER = inference_param['ENCODER']
    ENCODER_WEIGHTS = inference_param['ENCODER_WEIGHTS']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = functions.get_preprocessing(preprocessing_fn)

    # VIDEO ABSOLUTE OR RELATIVE PATH
    os.makedirs(output_dir, exist_ok=True)

    video_f_paths = [os.path.join(video_dir, video_id) for video_id in video_list]
    output_f_paths = [os.path.join(output_dir, os.path.splitext(video_id)[0]+'.mp4') for video_id in video_list]
    output_mask_f_paths = [os.path.join(output_dir, os.path.splitext(video_id)[0]+'_mask.mp4')
                           for video_id in video_list]

    for video_number in range(len(video_f_paths)):
        print('Opening video...')
        cap = cv2.VideoCapture(video_f_paths[video_number])

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resized_w, resized_h = network_img_size

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_recorder = cv2.VideoWriter(output_f_paths[video_number], fourcc, fps,
                                         (original_w, original_h))
        video_mask_recorder = cv2.VideoWriter(output_mask_f_paths[video_number], fourcc, fps,
                                              (original_w, original_h))

        if cap.isOpened():
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("Total Frame number is " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            print("Video is not opened")
            break

        for i in tqdm(range(int(total_frame)), desc=video_f_paths[video_number]+" progress"):
            # print(str(int(i / total_frame * 100)) + ' %')
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    start = time.time()

                    frame_resized = cv2.resize(frame, (resized_w, resized_h))

                    tip_drill, frame_output, mask_output = fn.predict_and_mark(
                        in_img=frame_resized,
                        best_model=best_model,
                        preprocessing=preprocessing,
                        model_dev=inference_param['DEVICE'],
                        output_size=(original_w, original_h),
                        postive_treshold=inference_param['postive_detect_threshold'],
                    )

                    # if print_time:
                        # print(time.time() - start)

                    if show_img:
                        cv2.imshow('image', frame_output)
                        cv2.imshow('mask', mask_output / 255.0)
                        cv2.waitKey(1)

                    video_recorder.write(frame_output)
                    video_mask_recorder.write(mask_output)

        print(str(output_f_paths[video_number]) + ' is completed!')

        video_recorder.release()
        # Release video
        cap.release()
        # Close windows
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    video_output_confidence_map()

import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np

def run(
        frame_interval=1,
        video_list=None,
        video_path=None,
        output_frames_path=None,
        starting_number=None,
        overwriting=False,
        rotate_by_90=False,
):
    print("video list:", video_list)
    video_full_path_list = [os.path.join(video_path, video_f) for video_f in video_list]

    total_image_num = 0
    # Open Video
    print('Opening video...')
    for video_f in video_full_path_list:
        cap = cv2.VideoCapture(video_f)
        if cap.isOpened():
            total_image_num = total_image_num + int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_interval + 1)
        else:
            print("Video is not opened")

        cap.release()

    # if os.path.isdir(output_frames_path):
    #     shutil.rmtree(output_frames_path)
    # os.makedirs(output_frames_path, exist_ok=True)

    print('Total output image number is ' + str(total_image_num) + ' images.')

    total_image_num = 0
    frame_counter = 0
    for video_f in video_full_path_list:
        cap = cv2.VideoCapture(video_f)
        if cap.isOpened():
            total_image_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("Total Frame number is " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            print("Video is not opened")

        # Skip useless frames
        for i in tqdm(range(0, int(total_image_num))):
            if cap.isOpened():
                ret, frame = cap.read()
                if i % frame_interval == 0:
                    frame_counter += 1
                    # OUTPUT IMAGE ABSOLUTE OR RELATIVE PATH
                    output_image_path = os.path.join(output_frames_path, str(frame_counter+starting_number) + '.png')
                    if not overwriting and os.path.isfile(output_image_path):
                        raise RuntimeError("Old frame will be overritten")
                    if rotate_by_90:
                        frame = np.transpose(frame, (1, 0, 2))
                    cv2.imwrite(output_image_path, frame)

        print(video_f + ' extraction is completed!')
        # Release video
        cap.release()

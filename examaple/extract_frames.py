import os
import cv2
import shutil


def extract_frames(
        frame_interval=1,
        video_list=None,
        video_path=None,
        output_frames_path=None,
):
    print(video_list)
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

    shutil.rmtree(output_frames_path)
    os.mkdir(output_frames_path)
    extracted_num = 0
    # extracted_num = len(os.listdir(output_frames_path))
    start_png_num = extracted_num
    print('Total output image number is ' + str(total_image_num) + ' images.')

    total_image_num = 0
    for video_f in video_full_path_list:
        cap = cv2.VideoCapture(video_f)
        if cap.isOpened():
            total_image_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("Total Frame number is " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            print("Video is not opened")

        # Skip useless frames
        for i in range(0, int(total_image_num)):
            if cap.isOpened():
                ret, frame = cap.read()
                if i % frame_interval == 0:
                    start_png_num = start_png_num + 1
                    # OUTPUT IMAGE ABSOLUTE OR RELATIVE PATH
                    output_image_path = os.path.join(output_frames_path, str(start_png_num) + '.png')
                    cv2.imwrite(output_image_path, frame)

        print(video_f + ' is completed!')
        # Release video
        cap.release()

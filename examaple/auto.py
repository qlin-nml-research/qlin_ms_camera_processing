import os

import extract_frames
import rename
import make_labeled_image
import augmentation
import train_confidence_map
import video_output_confidence_map

from tools.Parameters import Parameters

if __name__ == '__main__':
    param = Parameters()

    # extract_frames.extract_frames(
    #     frame_interval=param.frame_interval,
    #     video_path=param.video_path,
    #     video_list=['real_micro_teleop.mp4', 'real_micro_traj.mp4'],
    #     output_frames_path=param.extracted_frames_path,
    # )

    # rename.rename(param)

    # make_labeled_image.make_labeled_image(parameters=param)

    # augmentation.augmentation(parameters=param)

    # augmentation.test_train_validation_split(
    #     parameters=param,
    #     dataset_train_path=param.dataset_train_path,
    #     dataset_test_path=param.dataset_test_path,
    #     dataset_val_path=param.dataset_val_path,
    #     dataset_train_mask_path=param.dataset_train_mask_path,
    #     dataset_test_mask_path=param.dataset_test_mask_path,
    #     dataset_val_mask_path=param.dataset_val_mask_path,
    # )

    # train_confidence_map.train_confidence_map(
    #     training_param={
    #         "CLASSES": ['drill_tip'],
    #         "ENCODER": 'resnet18',
    #         "ENCODER_WEIGHTS": 'imagenet',
    #         "ACTIVATION": 'sigmoid',
    #         "DEVICE": 'cuda',
    #         "batch_size": 6,
    #         "epoch": 30,
    #     },
    #     network_img_size=param.resize_target,
    #     dataset_path=param.dataset_path,
    #     result_path=param.result_path,
    # )

    video_output_confidence_map.video_output_confidence_map(
        video_dir=param.video_path,
        video_list=['real_micro_teleop.mp4', 'real_micro_traj.mp4'],
        model_path=os.path.join(param.result_path, "best_model.pth"),
        output_dir=os.path.join(param.result_path, 'video'),
        network_img_size=param.resize_target,
        inference_param={
            "CLASSES": ['drill_tip'],
            "ENCODER": 'resnet18',
            "ENCODER_WEIGHTS": 'imagenet',
            "ACTIVATION": 'sigmoid',
            "DEVICE": 'cuda',
            "postive_detect_threshold": 100,
        },
        print_time=False, show_img=False,
    )

import os
import warnings

# Disable warnings from the Albumentations package
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

from train_step import frame_extration
from train_step import re_locate_file
from train_step import process_image_set
from train_step import make_augmentation
from train_step import split_train_validation_test
from train_step import train_network
from inference_step import inference_on_vid_file


from config.training_param import TrainingParameters
from config.network_param import NetworkParameter

if __name__ == '__main__':
    train_param = TrainingParameters()
    nn_param = NetworkParameter()

    # frame_extration.run(
    #     frame_interval=train_param.frame_interval,
    #     video_path=train_param.video_path,
    #     video_list=['real_micro_teleop.mp4', 'real_micro_traj.mp4'],
    #     output_frames_path=train_param.extracted_frames_path,
    # )

    # re_locate_file.run(
    #     extracted_frames_path=train_param.extracted_frames_path,
    #     json_and_raw_path=train_param.json_and_raw_path
    # )
    #
    # process_image_set.run(
    #     json_and_raw_path=train_param.json_and_raw_path,
    #     raw_resized_path=train_param.raw_resized_path,
    #     mask_resized_path=train_param.mask_resized_path,
    #     network_img_size=train_param.network_img_size,
    #     make_binary_img_in=train_param.binary,
    #     mask_sigma=train_param.mask_sigma,
    #     binary_threshold=train_param.binary_threshold,
    #     make_debug_img=True, debug_img_path=train_param.debug_path,
    # )

    # make_augmentation.run(
    #     json_and_raw_path=train_param.json_and_raw_path,
    #     raw_resized_path=train_param.raw_resized_path,
    #     raw_augmented_path=train_param.raw_augmented_path,
    #     mask_resized_path=train_param.mask_resized_path,
    #     mask_augmented_path=train_param.mask_augmented_path,
    #     image_num_after_augmentation=train_param.image_num_after_augmentation,
    #     augmentation_preset=train_param.augmentation_preset
    # )
    #
    # split_train_validation_test.run(
    #     total_original_img=int(len(os.listdir(train_param.json_and_raw_path)) / 2),
    #     original_num_for_train=train_param.original_num_for_train,
    #     aug_num_for_train=train_param.aug_num_for_train,
    #     image_num_after_augmentation=train_param.image_num_after_augmentation,
    #     raw_resized_path=train_param.raw_resized_path,
    #     mask_resized_path=train_param.mask_resized_path,
    #     raw_augmented_path=train_param.raw_augmented_path,
    #     mask_augmented_path=train_param.mask_augmented_path,
    #     dataset_train_path=train_param.dataset_train_path,
    #     dataset_test_path=train_param.dataset_test_path,
    #     dataset_val_path=train_param.dataset_val_path,
    #     dataset_train_mask_path=train_param.dataset_train_mask_path,
    #     dataset_test_mask_path=train_param.dataset_test_mask_path,
    #     dataset_val_mask_path=train_param.dataset_val_mask_path,
    # )

    train_network.run(
        training_param={
            "CLASSES": nn_param.CLASSES,
            "ENCODER": nn_param.ENCODER,
            "ENCODER_WEIGHTS": nn_param.ENCODER_WEIGHTS,
            "ACTIVATION": nn_param.ACTIVATION,
            "DEVICE": nn_param.DEVICE,
            "batch_size": train_param.train_batch_size,
            "epoch": train_param.epoch,
        },
        network_img_size=train_param.network_img_size,
        dataset_path=train_param.dataset_path,
        result_path=train_param.result_path,
    )

    inference_on_vid_file.run(
        video_dir=train_param.video_path,
        video_list=['real_micro_teleop.mp4', 'real_micro_traj.mp4'],
        model_path=os.path.join(train_param.result_path, "best_model.pth"),
        output_dir=os.path.join(train_param.result_path, 'video'),
        network_img_size=train_param.network_img_size,
        inference_param={
            "CLASSES": nn_param.CLASSES,
            "ENCODER": nn_param.ENCODER,
            "ENCODER_WEIGHTS": nn_param.ENCODER_WEIGHTS,
            "ACTIVATION": nn_param.ACTIVATION,
            "DEVICE": nn_param.DEVICE,
            "postive_detect_threshold": 255.0/2
        },
        print_time=False, show_img=False,
    )

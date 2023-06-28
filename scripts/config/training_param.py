import os
import albumentations as albu


def get_augmentation_preset():
    train_transform = [
        albu.Blur(blur_limit=21, p=1),
        albu.MotionBlur(blur_limit=21),
        albu.GaussianBlur(blur_limit=21),
        albu.GlassBlur(),
        # albu.GaussNoise(p=0.2),
        # albu.ImageCompression(),
        albu.ISONoise(),
        # albu.MultiplicativeNoise(),
        albu.Downscale(),
        albu.Rotate(),
        # albu.OpticalDistortion(),
        # albu.GridDistortion(),
        # albu.ElasticTransform(),
        # albu.RandomGridShuffle(),
        # albu.HueSaturationValue(),
        albu.RGBShift(),
        # albu.ChannelDropout(),
        # albu.Normalize(),
        albu.RandomGamma(),
        albu.RandomBrightnessContrast(),
        # albu.RandomContrast(),
        albu.Compose(
            [
                albu.Rotate(),
                albu.OneOf(
                    [
                        albu.Blur(blur_limit=21, p=1),
                        albu.MotionBlur(blur_limit=21),
                        albu.GaussianBlur(blur_limit=21),
                        # albu.GlassBlur(),
                        # albu.GaussNoise(p=0.2),
                        # albu.ImageCompression(),
                        albu.ISONoise(),
                        # albu.MultiplicativeNoise(),
                        albu.Downscale(),
                        # albu.OpticalDistortion(),
                        # albu.GridDistortion(),
                        # albu.ElasticTransform(),
                        # albu.RandomGridShuffle(),
                        # albu.HueSaturationValue(),
                        albu.RGBShift(),
                        # albu.ChannelDropout(),
                        # albu.Normalize(),
                        albu.RandomGamma(),
                        # albu.RandomBrightness(),
                        # albu.RandomContrast(),
                        albu.RandomBrightnessContrast(),
                    ]
                )
            ]
        ),
    ]
    return albu.OneOf(train_transform)


class TrainingParameters:
    # image_quality = '4K_ver1'
    # network_img_size = [1920, 1088]

    # image_quality = '4K_960_ver1'
    # network_img_size = [960, 544]

    image_quality = '4K_768_ver2'
    network_img_size = [768, 768]

    # image_quality = '4K_1280_ver1'
    # network_img_size = [1280, 704]

    model = 'MSCamControlTip_model'
    original_video_quality = '4K'
    # Path
    dir_base_path = 'C:/Users/qlin1/Public Box/tmp_workspace/MS_camera_adaptive_cal'
    dir_base_path = 'E:/ExperimentData/MSCameraAutomation'

    video_path = os.path.join(dir_base_path, "microlens_original")
    extracted_frames_path = os.path.join(dir_base_path, "workspace/extracted_frames/")

    json_and_raw_path = os.path.join(dir_base_path, "workspace", model, image_quality, "json_and_raw")
    debug_path = os.path.join(dir_base_path, "workspace", model, image_quality, "debug")
    raw_resized_path = os.path.join(dir_base_path, "workspace", model, image_quality, "raw_resized")
    mask_resized_path = os.path.join(dir_base_path, "workspace", model, image_quality, "mask_resized")
    raw_augmented_path = os.path.join(dir_base_path, "workspace", model, image_quality, "raw_augmented")
    mask_augmented_path = os.path.join(dir_base_path, "workspace", model, image_quality, "mask_augmented")

    dataset_path = os.path.join(dir_base_path, "dataset", model, image_quality)
    dataset_train_path = os.path.join(dir_base_path, "dataset", model, image_quality, "train")
    dataset_test_path = os.path.join(dir_base_path, "dataset", model, image_quality, "test")
    dataset_val_path = os.path.join(dir_base_path, "dataset", model, image_quality, "val")
    dataset_train_mask_path = os.path.join(dir_base_path, "dataset", model, image_quality, "train_mask")
    dataset_test_mask_path = os.path.join(dir_base_path, "dataset", model, image_quality, "test_mask")
    dataset_val_mask_path = os.path.join(dir_base_path, "dataset", model, image_quality, "val_mask")

    result_path = os.path.join(dir_base_path, "result", model, image_quality)

    # 0_extract_frames
    frame_interval = 20

    # 1_rename
    # start_num = 0
    # total_image = 973

    # 2_make_labeled_image
    debug = True
    binary = False
    binary_threshold = 100
    mask_sigma = 400  # 800  #1500

    # 3_augmentation
    image_num_after_augmentation = 14000
    original_num_for_train = 1200
    aug_num_for_train = 11000
    augmentation_preset = get_augmentation_preset()

    # Training
    epoch = 10
    train_batch_size = 24

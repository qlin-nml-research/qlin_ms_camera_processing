import os


class Parameters:
    # Path
    model = 'MSCamControlTip_model'
    image_quality = '4K_256_ver1'
    original_video_quality = '4K'
    dir_base_path = 'E:/ExperimentData/MSCameraAutomation/'

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
    start_num = 0
    total_image = 973

    # 2_make_labeled_image
    debug = True
    binary = False
    threshold = 100
    # resize_size = (512, 512)  # 256  #512
    resize_target = [1920, 1088]

    # resize_size_w = 256
    margin = 70  # 100  #100
    mask_specific_image = False
    specific_image_num = 239
    mask_sigma = 400  # 800  #1500
    slide = 40  # 40  #80


    # 3_augmentation
    image_num_after_augmentation = 8000
    original_num_for_train = 600
    aug_num_for_train = 6000

    # 4_train_confidence_map
    CLASSES = ['drill_tip']
    ENCODER = 'resnet18'  # 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'
    epoch = 10
    train_batch_size = 5

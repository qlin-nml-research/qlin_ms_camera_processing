import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

from .tools import network_common as nn_common
from .tools.Dataset_confidence_map import Dataset


def run(
        training_param=None,
        network_img_size=None,
        dataset_path=None, result_path=None,
        output_test_image=False,
):
    image_size = network_img_size
    CLASSES = training_param['CLASSES']
    ENCODER = training_param['ENCODER']
    ENCODER_WEIGHTS = training_param['ENCODER_WEIGHTS']
    ACTIVATION = training_param['ACTIVATION']
    DEVICE = training_param['DEVICE']

    x_train_dir = os.path.join(dataset_path, 'train')
    y_train_dir = os.path.join(dataset_path, 'train_mask')
    x_valid_dir = os.path.join(dataset_path, 'val')
    y_valid_dir = os.path.join(dataset_path, 'val_mask')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        raise Exception("GPU not available. CPU training will be too slow.")
    print("device name", torch.cuda.get_device_name(0))

    # build model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        encoder_depth=5,
        classes=len(CLASSES),
        activation=ACTIVATION
    )

    # data normalization function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(x_train_dir, y_train_dir, image_size,
                            preprocessing=nn_common.get_preprocessing(preprocessing_fn),
                            classes=CLASSES)

    valid_dataset = Dataset(x_valid_dir, y_valid_dir, image_size,
                            preprocessing=nn_common.get_preprocessing(preprocessing_fn),
                            classes=CLASSES)

    print('the number of image/label in the train: ', len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=training_param['batch_size'], shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12)

    loss = smp_utils.losses.MSELoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=DEVICE,
                                             verbose=True)
    valid_epoch = smp_utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

    max_score = 1

    x_epoch_data = []
    train_mse_loss = []
    valid_mse_loss = []

    train_dataset_vis = Dataset(x_train_dir, y_train_dir, image_size, classes=CLASSES)
    valid_dataset_vis = Dataset(x_valid_dir, y_valid_dir, image_size, classes=CLASSES)

    if output_test_image:
        # shutil.rmtree(result_path + 'process_image')
        os.makedirs(os.path.join(result_path, 'process_image'), exist_ok=True)

    for i in range(0, training_param['epoch']):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        x_epoch_data.append(i)
        train_mse_loss.append(train_logs['mse_loss'])
        valid_mse_loss.append(valid_logs['mse_loss'])

        torch.save(model, os.path.join(result_path, 'current.pth'))

        if output_test_image:
            current_model = torch.load(os.path.join(result_path, 'current.pth'))

            n_train = np.random.choice(len(train_dataset))
            n_valid = np.random.choice(len(valid_dataset))

            image_train, mask_train = train_dataset[n_train]
            image_vis_train = train_dataset_vis[n_train][0].astype('uint8')
            mask_vis_train = train_dataset_vis[n_train][1]

            cv2.imwrite(os.path.join(result_path, 'process_image', str(i + 1) + '-train' + '.png'),
                        image_vis_train)
            cv2.imwrite(os.path.join(result_path, 'process_image', str(i + 1) + '-train_mask' + '.png'),
                        mask_vis_train[:, :, 0] * 255)

            x_tensor = torch.from_numpy(image_train).to(DEVICE).unsqueeze(0)
            pr_mask_train = current_model.predict(x_tensor)
            pr_mask_train = (pr_mask_train.squeeze().cpu().numpy())
            pr_mask_train = pr_mask_train * 255
            pr_mask_train = np.transpose(pr_mask_train, (0, 1))
            cv2.imwrite(os.path.join(result_path, 'process_image', str(i + 1) + '-train_predict' + '.png'),
                        pr_mask_train)

            image_valid, mask_valid = valid_dataset[n_valid]
            image_vis_valid = valid_dataset_vis[n_valid][0].astype('uint8')
            mask_vis_valid = valid_dataset_vis[n_valid][1]

            cv2.imwrite(result_path + 'process_image/' + str(i + 1) + '-valid' + '.png', image_vis_valid)
            cv2.imwrite(os.path.join(result_path, 'process_image', str(i + 1) + '-valid_mask' + '.png'),
                        mask_vis_valid * 255)

            x_tensor = torch.from_numpy(image_valid).to(DEVICE).unsqueeze(0)
            pr_mask_valid = current_model.predict(x_tensor)
            pr_mask_valid = (pr_mask_valid.squeeze().cpu().numpy())
            pr_mask_valid = pr_mask_valid * 255
            pr_mask_valid = np.transpose(pr_mask_valid, (0, 1))
            cv2.imwrite(os.path.join(result_path, 'process_image', str(i + 1) + '-valid_predict' + '.png'),
                        pr_mask_valid)

        # do something (save model, change lr, etc.)
        if max_score > valid_logs['mse_loss']:
            max_score = valid_logs['mse_loss']
            torch.save(model, os.path.join(result_path, 'best_model.pth'))
            print('Model saved!')

        if i % 10 == 0 and i != 0:
            optimizer.param_groups[0]['lr'] = \
                optimizer.param_groups[0]['lr'] * training_param['learning_rate_stepping']
            print(f'Decrease decoder learning rate to {optimizer.param_groups[0]["lr"]}!')

    fig = plt.figure(figsize=(7, 5))
    plt.plot(x_epoch_data, train_mse_loss, label='train')
    plt.plot(x_epoch_data, valid_mse_loss, label='validation')
    fig.suptitle("MSE loss", size="xx-large", color="blue", weight="bold")
    fig.supxlabel('epoch')
    fig.supylabel('MSE_loss')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(result_path, 'learning_curve.png'))
    # plt.show()

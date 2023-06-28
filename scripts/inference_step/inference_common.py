import torch
import numpy as np
import cv2


def predict_and_mark(in_frame, best_model, preprocessing, model_dev, network_img_size, postive_treshold):
    # get input image size
    h, w, c = in_frame.shape
    frame_resized = cv2.resize(in_frame, network_img_size)

    x_img = preprocessing(image=frame_resized)['image']
    x_tensor = torch.from_numpy(x_img).to(model_dev).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)  # torch.tensor
    pr_mask = (pr_mask.squeeze().cpu().numpy()) * 255
    pr_mask = np.transpose(pr_mask, (0, 1))

    tip_drill_ind = np.array(np.unravel_index(np.argmax(pr_mask[:, :]), pr_mask[:, :].shape))

    if pr_mask[tip_drill_ind[0], tip_drill_ind[1]] >= postive_treshold:
        # tip_drill in format (x, y)
        tip_drill = np.array([tip_drill_ind[1], tip_drill_ind[0]], dtype=float)
        tip_drill[0] = tip_drill[0] / network_img_size[0] * w
        tip_drill[1] = tip_drill[1] / network_img_size[1] * h
    else:
        tip_drill = None

    return tip_drill, cv2.resize(pr_mask, (w, h))

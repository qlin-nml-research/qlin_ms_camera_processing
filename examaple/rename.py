import os
import shutil
import glob


def rename(parameters):
    ids = glob.glob1(parameters.extracted_frames_path, "*.json")
    extracted_fps = [os.path.join(parameters.extracted_frames_path, img_id) for img_id in ids]

    if not os.path.exists(parameters.json_and_raw_path):
        os.makedirs(parameters.json_and_raw_path, exist_ok=True)

    # image_number = len(os.listdir(parameters.json_and_raw_path))

    for i, id in enumerate(ids):
        shutil.copyfile(extracted_fps[i], os.path.join(parameters.json_and_raw_path, id))
        shutil.copyfile(extracted_fps[i].replace(".json", ".png"),
                        os.path.join(parameters.json_and_raw_path, id.replace(".json", ".png")))


if __name__ == '__main__':
    rename()

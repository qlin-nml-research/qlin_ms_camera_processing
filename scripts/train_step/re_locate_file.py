import os
import shutil
import glob


def run(extracted_frames_path, json_and_raw_path):
    ids = glob.glob1(extracted_frames_path, "*.json")
    extracted_fps = [os.path.join(extracted_frames_path, img_id) for img_id in ids]

    if not os.path.exists(json_and_raw_path):
        os.makedirs(json_and_raw_path, exist_ok=True)

    # image_number = len(os.listdir(parameters.json_and_raw_path))

    for i, id in enumerate(ids):
        shutil.copyfile(extracted_fps[i], os.path.join(json_and_raw_path, id))
        shutil.copyfile(extracted_fps[i].replace(".json", ".png"),
                        os.path.join(json_and_raw_path, id.replace(".json", ".png")))

import os
import shutil
import glob
from tqdm import tqdm

def run(extracted_frames_path, json_and_raw_path):
    ids = glob.glob1(extracted_frames_path, "*.png")
    extracted_fps = [os.path.join(extracted_frames_path, img_id) for img_id in ids]

    if not os.path.exists(json_and_raw_path):
        os.makedirs(json_and_raw_path, exist_ok=True)

    # image_number = len(os.listdir(parameters.json_and_raw_path))

    for i, id in enumerate(tqdm(ids)):
        # also include renumbering
        shutil.copyfile(extracted_fps[i], os.path.join(json_and_raw_path, str(i+1)+".png"))
        if os.path.isfile(extracted_fps[i].replace(".png", ".json")):
            shutil.copyfile(extracted_fps[i].replace(".png", ".json"),
                            os.path.join(json_and_raw_path, str(i+1)+".json"))

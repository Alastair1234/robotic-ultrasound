from datasets import Dataset, Features, Array3D, Sequence, Value
import json
import cv2
import numpy as np
import os

def load_data(json_path, images_dir, target_frame='capture308.jpg'):
    with open(json_path, 'r') as file:
        data = json.load(file)["PointInfos"]

    target_info = next(d for d in data if d["FileName"] == target_frame)
    target_pos = np.array([
        target_info["Position"]["x"],
        target_info["Position"]["y"],
        target_info["Position"]["z"]
    ])
    target_rot = np.array([
        target_info["RotationEuler"]["x"],
        target_info["RotationEuler"]["y"],
        target_info["RotationEuler"]["z"]
    ])

    images, labels = [], []
    for d in data:
        img_path = os.path.join(images_dir, d["FileName"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        pos = np.array([d["Position"]["x"], d["Position"]["y"], d["Position"]["z"]])
        rot = np.array([d["RotationEuler"]["x"], d["RotationEuler"]["y"], d["RotationEuler"]["z"]])

        label = (target_pos - pos).tolist() + (target_rot - rot).tolist()

        images.append(img)
        labels.append(label)

    return {"image": images, "label": labels}

def create_hf_dataset(json_path, images_dir):
    data = load_data(json_path, images_dir)
    features = Features({
        'image': Array3D(dtype='float32', shape=(224, 224, 3)),
        'label': Sequence(Value('float32'), length=6)
    })
    dataset = Dataset.from_dict(data, features=features)
    return dataset

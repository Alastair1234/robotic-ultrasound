from datasets import Dataset, Features, Array3D, Sequence, Value
import json
import cv2
import numpy as np
import os

def normalize_labels(labels):
    labels = np.array(labels)
    mean = labels.mean(axis=0)
    std = labels.std(axis=0)
    normalized_labels = (labels - mean) / std
    return normalized_labels, mean.tolist(), std.tolist()

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

    images, raw_labels = [], []
    for d in data:
        img_path = os.path.join(images_dir, d["FileName"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        pos = np.array([
            d["Position"]["x"],
            d["Position"]["y"],
            d["Position"]["z"]
        ])
        rot = np.array([
            d["RotationEuler"]["x"],
            d["RotationEuler"]["y"],
            d["RotationEuler"]["z"]
        ])

        # Position difference
        pos_diff = target_pos - pos

        # Rotation difference normalized between [-180, 180]
        rot_diff = (target_rot - rot + 180) % 360 - 180
        rot_diff_rad = np.deg2rad(rot_diff)

        # Sine and cosine components for each rotation clearly
        rot_sin_cos = np.hstack([np.sin(rot_diff_rad), np.cos(rot_diff_rad)])

        # Concatenate position and rotation sin/cos clearly
        label = np.concatenate([pos_diff, rot_sin_cos])

        images.append(img)
        raw_labels.append(label)

    labels, label_mean, label_std = normalize_labels(raw_labels)

    # Save mean and std clearly
    np.save('label_mean.npy', label_mean)
    np.save('label_std.npy', label_std)

    return {"image": images, "label": labels.tolist()}

def create_hf_dataset(json_path, images_dir):
    data = load_data(json_path, images_dir)
    features = Features({
        'image': Array3D(dtype='float32', shape=(224, 224, 3)),
        'label': Sequence(Value('float32'), length=9)  # 3 positions + 6 rotation sin/cos
    })
    dataset = Dataset.from_dict(data, features=features)
    return dataset

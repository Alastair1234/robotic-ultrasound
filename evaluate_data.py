import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_rotation(json_path, output_path='rotation_analysis.png'):
    with open(json_path, 'r') as file:
        data = json.load(file)["PointInfos"]

    rot_x, rot_y, rot_z = [], [], []

    # Extract rotation data clearly
    for d in data:
        rot_x.append(d["RotationEuler"]["x"])
        rot_y.append(d["RotationEuler"]["y"])
        rot_z.append(d["RotationEuler"]["z"])

    rot_x, rot_y, rot_z = np.array(rot_x), np.array(rot_y), np.array(rot_z)

    # Plot histograms and scatter plots clearly
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Histograms
    axs[0, 0].hist(rot_x, bins=50, color='red', alpha=0.7)
    axs[0, 0].set_title('Rotation X Histogram')
    axs[0, 0].set_xlabel('Degrees')
    axs[0, 0].set_ylabel('Count')

    axs[1, 0].hist(rot_y, bins=50, color='green', alpha=0.7)
    axs[1, 0].set_title('Rotation Y Histogram')
    axs[1, 0].set_xlabel('Degrees')
    axs[1, 0].set_ylabel('Count')

    axs[2, 0].hist(rot_z, bins=50, color='blue', alpha=0.7)
    axs[2, 0].set_title('Rotation Z Histogram')
    axs[2, 0].set_xlabel('Degrees')
    axs[2, 0].set_ylabel('Count')

    # Scatter plots to observe relationships
    axs[0, 1].scatter(rot_x, rot_y, c='purple', alpha=0.6, s=10)
    axs[0, 1].set_title('Rotation X vs Rotation Y')
    axs[0, 1].set_xlabel('Rot X')
    axs[0, 1].set_ylabel('Rot Y')

    axs[1, 1].scatter(rot_y, rot_z, c='orange', alpha=0.6, s=10)
    axs[1, 1].set_title('Rotation Y vs Rotation Z')
    axs[1, 1].set_xlabel('Rot Y')
    axs[1, 1].set_ylabel('Rot Z')

    axs[2, 1].scatter(rot_x, rot_z, c='cyan', alpha=0.6, s=10)
    axs[2, 1].set_title('Rotation X vs Rotation Z')
    axs[2, 1].set_xlabel('Rot X')
    axs[2, 1].set_ylabel('Rot Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# Adjust the path clearly if necessary
analyze_rotation('data/imageInfo.json')

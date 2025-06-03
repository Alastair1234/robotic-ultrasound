import torch
from torch.utils.data import DataLoader
from dataset_hf import create_hf_dataset
from model import DinoV2Regressor
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    images = torch.stack([torch.tensor(example['image']).permute(2, 0, 1) for example in batch])
    labels = torch.stack([torch.tensor(example['label']) for example in batch])
    return images, labels

def rotation_diff(true, pred):
    diff = np.abs(true - pred) % 360
    diff = np.minimum(diff, 360 - diff)
    return diff

def recover_angles_from_sin_cos(sin_vals, cos_vals):
    angles_rad = np.arctan2(sin_vals, cos_vals)
    angles_deg = np.rad2deg(angles_rad)
    return angles_deg

def combined_visualization(true_values, predicted_values, epoch, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'epoch_{epoch}_combined.pdf')

    with PdfPages(pdf_path) as pdf:
        # Position X Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(true_values[:, 0], predicted_values[:, 0], alpha=0.7, color='blue', edgecolors='k', label='Predictions')

        min_val = min(true_values[:, 0].min(), predicted_values[:, 0].min())
        max_val = max(true_values[:, 0].max(), predicted_values[:, 0].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

        plt.xlabel('True Position X')
        plt.ylabel('Predicted Position X')
        plt.title(f'Epoch {epoch} - True vs Predicted Position X')
        plt.legend()
        plt.grid(True)

        rmse_pos = np.sqrt(mean_squared_error(true_values[:, 0], predicted_values[:, 0]))
        r2_pos = r2_score(true_values[:, 0], predicted_values[:, 0])

        plt.text(0.05, 0.95, f'RMSE: {rmse_pos:.6f}\nR²: {r2_pos:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        pdf.savefig()
        plt.close()

        # Rotation X Plot
        true_rot_x = recover_angles_from_sin_cos(true_values[:, 3], true_values[:, 6])
        pred_rot_x = recover_angles_from_sin_cos(predicted_values[:, 3], predicted_values[:, 6])

        angle_differences = rotation_diff(true_rot_x, pred_rot_x)

        plt.figure(figsize=(8, 8))
        plt.scatter(true_rot_x, pred_rot_x, alpha=0.7, color='green', edgecolors='k', label='Predictions')
        plt.plot([-180, 180], [-180, 180], 'r--', label='Ideal Prediction')

        plt.xlabel('True Rotation X (degrees)')
        plt.ylabel('Predicted Rotation X (degrees)')
        plt.title(f'Epoch {epoch} - True vs Predicted Rotation X')
        plt.legend()
        plt.grid(True)

        rmse_rot = np.sqrt(np.mean(angle_differences ** 2))
        r2_rot = r2_score(true_rot_x, pred_rot_x)

        plt.text(0.05, 0.95, f'RMSE: {rmse_rot:.4f}\nR²: {r2_rot:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        pdf.savefig()
        plt.close()

def main():
    dataset = create_hf_dataset('data/imageInfo.json', 'data/images')
    dataset = dataset.train_test_split(test_size=0.1)

    batch_size = 16
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=False)

    label_mean = torch.tensor(np.load('label_mean.npy')).to(device)
    label_std = torch.tensor(np.load('label_std.npy')).to(device)
    print(f"[Normalization Check] Label Mean: {label_mean.cpu().numpy()}")
    print(f"[Normalization Check] Label Std: {label_std.cpu().numpy()}")

    model = DinoV2Regressor().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Training] Epoch {epoch+1} Loss (Normalized): {avg_loss:.6f}")

        model.eval()
        eval_loss = 0.0
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} Evaluation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                eval_loss += loss.item()

                outputs_denorm = outputs * label_std + label_mean
                labels_denorm = labels * label_std + label_mean

                all_predictions.append(outputs_denorm.cpu().numpy())
                all_labels.append(labels_denorm.cpu().numpy())

        avg_eval_loss = eval_loss / len(test_loader)
        print(f"[Evaluation] Epoch {epoch+1} Loss (Normalized): {avg_eval_loss:.6f}")

        predictions_array = np.vstack(all_predictions)
        ground_truth_array = np.vstack(all_labels)

        # Explicit detailed logs for Rotation X:
        true_rot_x_sin = ground_truth_array[:, 3]
        true_rot_x_cos = ground_truth_array[:, 6]
        pred_rot_x_sin = predictions_array[:, 3]
        pred_rot_x_cos = predictions_array[:, 6]

        true_rot_x_angles = recover_angles_from_sin_cos(true_rot_x_sin, true_rot_x_cos)
        pred_rot_x_angles = recover_angles_from_sin_cos(pred_rot_x_sin, pred_rot_x_cos)

        print(f"[Rotation Debugging] True sin values (sample): {true_rot_x_sin[:5]}")
        print(f"[Rotation Debugging] True cos values (sample): {true_rot_x_cos[:5]}")
        print(f"[Rotation Debugging] True angles recovered (degrees, sample): {true_rot_x_angles[:5]}")
        print(f"[Rotation Debugging] Pred sin values (sample): {pred_rot_x_sin[:5]}")
        print(f"[Rotation Debugging] Pred cos values (sample): {pred_rot_x_cos[:5]}")
        print(f"[Rotation Debugging] Pred angles recovered (degrees, sample): {pred_rot_x_angles[:5]}")

        combined_visualization(
            ground_truth_array,
            predictions_array,
            epoch=epoch + 1,
            save_dir='plots'
        )

        torch.save(model.state_dict(), f'dinov2_regressor_epoch{epoch+1}.pth')

    torch.save(model.state_dict(), 'dinov2_regressor_final.pth')

if __name__ == '__main__':
    main()

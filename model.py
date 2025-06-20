import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DinoV2Regressor(nn.Module):
    def __init__(self, output_dim=9, model_name='facebook/dinov2-base'):
        super(DinoV2Regressor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)  # explicitly outputting 9 dimensions
        )

    def forward(self, images):
        inputs = self.processor(images, return_tensors="pt").to(images.device)
        outputs = self.backbone(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        return self.regressor(pooled_output)

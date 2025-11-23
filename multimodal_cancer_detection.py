# multimodal_cancer_detection.py
# Python 3.14 compatible
# Demonstration of multimodal AI model combining image and clinical/genomic data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------------
# Image Encoder (CNN)
# -------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(ImageEncoder, self).__init__()
        base_model = models.resnet18(weights=None)
        # remove final classification layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, output_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

# -------------------------------
# Clinical/Genomic Data Encoder (Tabular NN)
# -------------------------------
class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=10, output_dim=128):
        super(ClinicalEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

# -------------------------------
# Fusion Model
# -------------------------------
class MultimodalCancerModel(nn.Module):
    def __init__(self, img_feat_dim=256, clinical_feat_dim=128, num_classes=2):
        super(MultimodalCancerModel, self).__init__()
        self.image_encoder = ImageEncoder(img_feat_dim)
        self.clinical_encoder = ClinicalEncoder(output_dim=clinical_feat_dim)
        
        # fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + clinical_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical_data):
        img_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical_data)
        fused = torch.cat((img_features, clinical_features), dim=1)
        output = self.classifier(fused)
        return output

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Simulated input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)   # RGB medical images
    dummy_clinical = torch.randn(batch_size, 10)          # e.g., age, gene markers, etc.

    # Initialize model
    model = MultimodalCancerModel()
    predictions = model(dummy_images, dummy_clinical)

    print("Output shape:", predictions.shape)
    print("Predicted class scores:\n", predictions)


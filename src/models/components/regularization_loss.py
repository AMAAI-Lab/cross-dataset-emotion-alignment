import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizationLossV2(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarities
        sim_matrix = torch.matmul(features, features.T)

        # Create negative mask
        negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

        # Exclude self-similarities
        negative_mask.fill_diagonal_(0)

        # Compute negative similarities
        negative_similarities = (sim_matrix * negative_mask).sum(1) / negative_mask.sum(1).clamp(
            min=1e-8
        )

        # Compute loss to push away negative examples
        loss = self.alpha * F.relu(negative_similarities + 1)

        return loss.mean()


class RegularizationLossV3(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarities
        sim_matrix = torch.matmul(features, features.T)

        # Create positive mask
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Exclude self-similarities
        positive_mask.fill_diagonal_(0)

        # Compute positive similarities
        positive_similarities = (sim_matrix * positive_mask).sum(1) / positive_mask.sum(1).clamp(
            min=1e-8
        )

        # Compute loss to pull together positive examples
        loss = self.alpha * (1 - positive_similarities)

        return loss.mean()


class RegularizationLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarities
        sim_matrix = torch.matmul(features, features.T)

        # Compute triplet loss
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        negative_mask = 1 - positive_mask

        # Exclude self-similarities
        positive_mask.fill_diagonal_(0)
        negative_mask.fill_diagonal_(0)

        # Compute positive and negative similarities
        positive_similarities = (sim_matrix * positive_mask).sum(1) / positive_mask.sum(1).clamp(
            min=1e-8
        )
        negative_similarities = (sim_matrix * negative_mask).sum(1) / negative_mask.sum(1).clamp(
            min=1e-8
        )

        # Compute triplet loss with margin
        loss = self.alpha * F.relu(negative_similarities - positive_similarities + 1)

        return loss.mean()

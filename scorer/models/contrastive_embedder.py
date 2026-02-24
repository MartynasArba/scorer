import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConSleepCNN(nn.Module):
    """
    wraps EmbeddingCNN with a projection head for contrastive learning
    """
    def __init__(self, base_model, embedding_dim=128):
        super().__init__()
        self.encoder = base_model
        # projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        # Ephys only slicing
        x = x[:, :2, :]
        
        # Convolutions
        x = F.relu(self.encoder.conv1(x))
        x = self.encoder.pool(x)
        x = F.relu(self.encoder.conv2(x))
        x = self.encoder.pool(x)
        x = F.relu(self.encoder.conv3(x))
        x = self.encoder.pool(x)
        x = F.relu(self.encoder.conv4(x))
        x = self.encoder.pool(x)
        
        # Features
        features = self.encoder.global_pool(x).view(x.size(0), -1) 
        
        # 2. Project and Normalize (Essential for cosine similarity)
        projected = self.projection_head(features)
        return F.normalize(projected, p=2, dim=1)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()
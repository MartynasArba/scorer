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
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
            # Let the updated EphysSleepCNN handle the BatchNorms, pooling, and flattening
            features = self.encoder.get_features(x)
            
            # project and normalize for cosine similarity
            projected = self.projection_head(features)
            return F.normalize(projected, p=2, dim=1)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)    #mask marks all positive pairs as 1

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),         #pairwise cosine similarity (-1/t to 1/t) because features were normalized previously
            self.temperature
        )
        
        # subtracting max logit for numerical stability (preventing exp operations on high numbers)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        #make sure anchor is not compared to itself - add 0s to the diagonal of the mask
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        #compute log_prob like cross-entropy
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        #get count of positive samples (called cardinality for some reason?)
        pos_counts = mask.sum(dim=1)
        #safeguard against zero division for isolated minority samples, should be irrelevant in balanced datasets
        pos_counts[pos_counts == 0] = 1.0 
        #aggregate the log-probabilities over valid positive pairs and normalize
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_counts
        # return the minimized negative log-likelihood
        return -mean_log_prob_pos.mean()
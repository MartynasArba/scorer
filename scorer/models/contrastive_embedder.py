import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConSleepCNN(nn.Module):
    """
    wraps EmbeddingCNN with a projection head for contrastive learning
    """
    def __init__(self, base_model, input_dim = 512, embedding_dim=128):
        super().__init__()
        self.encoder = base_model
        
        # projection head for contrastive learning only
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embedding_dim)
        )

    def forward(self, x):
            # get embeddings
            features = self.encoder.get_features(x)
            # project to 128d and normalize for cosine similarity
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
    
    
class SimCLRLoss(nn.Module):
    """
    unsupervised contrastive learning (NT-Xent Loss): https://arxiv.org/abs/2002.05709
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        #crossentropy backend
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        z1: [batch_size, embedding_dim] view1 of batch 
        z2: [batch_size, embedding_dim] view2 of batch 
        """
        device = z1.device
        batch_size = z1.size(0)
        
        #stack features into single [2N, D] tensor, N is view1 and 2
        z = torch.cat((z1, z2), dim=0) 
        
        # compute pairwise cosine similarity (-1/t to 1/t), result shape [2N, 2N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # mask self similarity 
        sim_matrix.fill_diagonal_(-1e9)
        
        # create targets: for view 1 (z[0]), positive pair is view 2 (x[batch_size]), and vice versa
        targets = torch.arange(batch_size, device=device)
        targets = torch.cat([targets + batch_size, targets], dim=0)
        
        # compute crossentropy
        return self.criterion(sim_matrix, targets)
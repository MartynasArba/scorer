import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareSleepScorer(nn.Module):
    def __init__(self, pretrained_cnn, embedding_dim=512, hidden_dim=64, num_classes=3, num_layers=2):
        super().__init__()
        
        # feature extractor is contrastive-pretrained SCDS CNN
        self.encoder = pretrained_cnn
        
        # freeze CNN
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # remove classification layer if it's there
        # CNN outputs 'embedding_dim' directly before fc layers (as feature_proj)
        if hasattr(self.encoder, 'fc1'):
            self.encoder.fc1 = nn.Identity()
        if hasattr(self.encoder, 'fc2'):
            self.encoder.fc2 = nn.Identity()
        
        #put in eval to disable dropout
        self.encoder.eval()

        # context wrapper (bi-directional GRU, default from torch)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # classification head
        # hidden_dim * 2 because it's bi-directional
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x expected shape: [batch_size, seq_length, chs, win_len]
        batch_size, seq_len, channels, win_len = x.size()
        
        # flatten batch and sequence to push through pretrained CNN to [batch * seq, chs, win_len]
        x_flat = x.view(batch_size * seq_len, channels, win_len)
        
        # extract features
        with torch.no_grad(): # ensure no gradients leak into frozen encoder
            embeddings = self.encoder(x_flat) # Shape: [batch * seq, embedding_dim (128)]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # reshape back to sequence format for RNN: [batch, seq, embedding_dim]
        rnn_input = embeddings.view(batch_size, seq_len, -1)
        
        # pass through Bi-GRU
        # rnn_out shape: [batch, seq, hidden_dim * 2]
        rnn_out, _ = self.rnn(rnn_input)
        
        # classify every step in the sequence: [batch, seq, num_classes]
        logits = self.classifier(rnn_out)
        
        # for standard crossentropy transpose to: [batch, num_classes, seq_len]
        return logits.transpose(1, 2)
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # handles class imbalance
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        #standard crossentropy
        ce_loss = F.cross_entropy(inputs, targets, weight = self.weight, reduction='none', label_smoothing = 0.1)
        
        #extract prob of true class, CE = -log(pt), so pt = exp(-CE)
        pt = torch.exp(-ce_loss)
        # applied focal scaling factor gamma, loss = (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # final reduction (mean or sum)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
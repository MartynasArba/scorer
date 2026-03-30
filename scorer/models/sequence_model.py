import torch
import torch.nn as nn

class ContextAwareSleepScorer(nn.Module):
    def __init__(self, pretrained_cnn, embedding_dim=128, hidden_dim=64, num_classes=3, num_layers=2):
        super().__init__()
        
        # 1. The Frozen Feature Extractor
        self.encoder = pretrained_cnn
        
        # Freeze the CNN parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Remove the old classification head from the CNN if it's still attached
        # (Assuming your base_cnn outputs 'embedding_dim' directly before the fc layers)
        if hasattr(self.encoder, 'fc1'):
            self.encoder.fc1 = nn.Identity()
        if hasattr(self.encoder, 'fc2'):
            self.encoder.fc2 = nn.Identity()

        # 2. The Context Wrapper (Bi-directional GRU)
        # Bi-directional means it looks at past AND future context to score the present
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # 3. The Classification Head
        # hidden_dim * 2 because it's bi-directional
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x expected shape: [Batch_size, Sequence_Length, Channels, Window_Length]
        # Example: [32, 5, 2, 1000] (A batch of 32 sequences, each sequence has 5 windows)
        
        batch_size, seq_len, channels, win_len = x.size()
        
        # Step 1: Flatten batch and sequence to push through the CNN
        # Shape becomes: [160, 2, 1000]
        x_flat = x.view(batch_size * seq_len, channels, win_len)
        
        # Step 2: Extract features using your Contrastive Pretrained CNN
        with torch.no_grad(): # Ensure no gradients leak into the frozen encoder
            embeddings = self.encoder(x_flat) # Shape: [160, 128]
            
        # Step 3: Reshape back to sequence format for the RNN
        # Shape becomes: [32, 5, 128]
        rnn_input = embeddings.view(batch_size, seq_len, -1)
        
        # Step 4: Pass through the Bi-GRU
        # rnn_out shape: [32, 5, hidden_dim * 2]
        rnn_out, _ = self.rnn(rnn_input)
        
        # Step 5: Classify every step in the sequence
        # Shape becomes: [32, 5, 3]
        logits = self.classifier(rnn_out)
        
        # In standard PyTorch CrossEntropyLoss for sequences, we usually transpose to:
        # [Batch, Classes, Sequence_Length] -> [32, 3, 5]
        return logits.transpose(1, 2)